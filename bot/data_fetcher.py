# data_fetcher.py - Grok Elite Signal Bot v27.8.0 - Data Fetching Layer
"""
Handles all external data fetching:
- OHLCV data with VWAP computation
- Ticker prices (batch & individual)
- Order flow / orderbook data
- Open interest data
- Binance cross-exchange data (v27.8.0)
- Combined funding rates (v27.8.0)
- Exchange divergence analysis (v27.8.0)
- Volume comparison (v27.8.0)

v27.8.0: Full Binance integration for cross-exchange analysis
"""
import asyncio
import time
import logging
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt.async_support as ccxt

from bot.config import (
    SYMBOLS, EXCHANGE_CONFIG, FUTURES_EXCHANGE_CONFIG,
    CACHE_TTL, HTF_CACHE_TTL, TICKER_CACHE_TTL, ORDER_FLOW_CACHE_TTL,
    FETCH_SEMAPHORE_LIMIT, USE_BINANCE_DATA, EXCHANGE_DIVERGENCE_SIGNIFICANT_PCT,
    FUNDING_EXTREME_PCT
)
from bot.utils import BanManager, evict_if_full, cache_get

# ============================================================================
# EXCHANGE INSTANCES (Perpetual Futures)
# ============================================================================
exchange = ccxt.bybit(EXCHANGE_CONFIG)
futures_exchange = ccxt.bybit(FUTURES_EXCHANGE_CONFIG)
binance_exchange = None
binance_futures = None

# ============================================================================
# CACHE STORAGE
# ============================================================================
ohlcv_cache: OrderedDict = OrderedDict()
ticker_cache: OrderedDict = OrderedDict()
order_flow_cache: OrderedDict = OrderedDict()
binance_ticker_cache: OrderedDict = OrderedDict()
binance_funding_cache: OrderedDict = OrderedDict()
divergence_cache: OrderedDict = OrderedDict()

last_oi: Dict[str, float] = {}
fetch_sem = asyncio.Semaphore(FETCH_SEMAPHORE_LIMIT)

# ============================================================================
# BINANCE INITIALIZATION
# ============================================================================
async def init_binance():
    """Initialize Binance spot exchange (lazy init)"""
    global binance_exchange
    if binance_exchange is None:
        binance_exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 1200,
            'options': {'defaultType': 'spot'},
            'timeout': 30000
        })
    return binance_exchange

async def init_binance_futures():
    """Initialize Binance futures exchange (lazy init)"""
    global binance_futures
    if binance_futures is None:
        binance_futures = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 1200,
            'options': {'defaultType': 'future'},
            'timeout': 30000
        })
    return binance_futures

# ============================================================================
# VWAP COMPUTATION
# ============================================================================
def compute_vwap_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VWAP safely with proper DatetimeIndex handling."""
    if len(df) < 20 or 'date' not in df.columns:
        df['vwap'] = np.nan
        return df
    
    try:
        df_indexed = df.set_index('date').sort_index().copy()
        
        total_vol = df_indexed['volume'].sum()
        if total_vol == 0:
            df['vwap'] = np.nan
            return df
        
        vwap_series = ta.vwap(
            df_indexed['high'],
            df_indexed['low'],
            df_indexed['close'],
            df_indexed['volume']
        )
        
        vwap_dict = dict(zip(df_indexed.index, vwap_series))
        df['vwap'] = df['date'].map(vwap_dict)
        
        return df
        
    except Exception as e:
        logging.error(f"VWAP computation failed: {e}")
        df['vwap'] = np.nan
        return df

# ============================================================================
# OHLCV FETCHING
# ============================================================================
async def fetch_ohlcv(symbol: str, tf: str, limit: int = 200, since: Optional[int] = None) -> pd.DataFrame:
    """Fetch OHLCV data with proper VWAP computation and caching."""
    async with fetch_sem:
        if await BanManager.check_and_sleep():
            cached = cache_get(ohlcv_cache, f"{symbol}{tf}")
            return cached.get('df', pd.DataFrame()) if cached else pd.DataFrame()
        
        cache_key = f"{symbol}{tf}"
        now = time.time()
        ttl = HTF_CACHE_TTL if tf in ['1d', '1w'] else CACHE_TTL
        
        cached = cache_get(ohlcv_cache, cache_key)
        if cached and now - cached['timestamp'] < ttl:
            return cached['df']
        
        evict_if_full(ohlcv_cache)
        backoff = 1
        
        for attempt in range(4):
            try:
                norm_tf = tf.lower()
                params = {'limit': limit}
                if since:
                    params['since'] = since
                
                data = await exchange.fetch_ohlcv(symbol, norm_tf, **params)
                
                df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['ts'], unit='ms')
                df = compute_vwap_safe(df)
                
                ohlcv_cache[cache_key] = {'df': df, 'timestamp': now}
                logging.debug(f"OHLCV fetched for {symbol} {tf}")
                break
                
            except Exception as e:
                logging.warning(f"OHLCV fetch attempt {attempt+1} failed for {symbol} {tf}: {e}")
                if attempt < 3:
                    await asyncio.sleep(backoff * 3)
                    backoff *= 3
        
        await asyncio.sleep(3)
        
        cached = cache_get(ohlcv_cache, cache_key)
        return cached.get('df', pd.DataFrame()) if cached else pd.DataFrame()

# ============================================================================
# TICKER FETCHING (BYBIT)
# ============================================================================
async def fetch_ticker_batch() -> Dict[str, Optional[float]]:
    """Fetch current prices for all symbols in batch."""
    async with fetch_sem:
        now = time.time()
        
        cache_hits = sum(1 for s in SYMBOLS if s in ticker_cache and now - ticker_cache[s]['timestamp'] < TICKER_CACHE_TTL)
        
        if cache_hits == len(SYMBOLS):
            return {s: ticker_cache[s]['price'] for s in SYMBOLS}
        
        if await BanManager.check_and_sleep():
            return {s: cache_get(ticker_cache, s).get('price') if cache_get(ticker_cache, s) else None for s in SYMBOLS}
        
        evict_if_full(ticker_cache)
        backoff = 1
        prices = {}
        
        for attempt in range(3):
            try:
                tickers = await exchange.fetch_tickers(SYMBOLS)
                prices = {s: tickers[s]['last'] if s in tickers else None for s in SYMBOLS}
                
                for s in SYMBOLS:
                    if prices[s]:
                        ticker_cache[s] = {'price': prices[s], 'timestamp': now}
                
                break
                
            except Exception as e:
                logging.debug(f"Batch ticker poll attempt {attempt+1} failed: {e}")
                if "429" in str(e):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    if attempt == 2:
                        prices = {s: cache_get(ticker_cache, s).get('price') if cache_get(ticker_cache, s) else None for s in SYMBOLS}
        
        return prices

async def fetch_ticker(symbol: str) -> Optional[float]:
    """Fetch current price for a single symbol."""
    prices = await fetch_ticker_batch()
    return prices.get(symbol)

# ============================================================================
# BINANCE TICKER FETCHING (v27.8.0)
# ============================================================================
async def fetch_binance_ticker(symbol: str) -> Optional[float]:
    """
    Fetch current price from Binance for divergence analysis.
    
    Args:
        symbol: Trading pair
    
    Returns:
        Price or None
    """
    if not USE_BINANCE_DATA:
        return None
    
    now = time.time()
    
    cached = cache_get(binance_ticker_cache, symbol)
    if cached and now - cached['timestamp'] < TICKER_CACHE_TTL:
        return cached['price']
    
    try:
        exchange_obj = await init_binance()
        ticker = await exchange_obj.fetch_ticker(symbol)
        price = ticker['last'] if ticker else None
        
        if price:
            binance_ticker_cache[symbol] = {'price': price, 'timestamp': now}
        
        return price
    except Exception as e:
        logging.debug(f"Binance ticker fetch error for {symbol}: {e}")
        return None

# ============================================================================
# EXCHANGE DIVERGENCE ANALYSIS (v27.8.0)
# ============================================================================
async def calculate_exchange_divergence(symbol: str) -> Optional[Dict]:
    """
    Calculate price divergence between Bybit and Binance.
    
    Positive = Bybit premium (bearish - futures overleveraged)
    Negative = Binance premium (bullish - spot accumulation)
    
    Args:
        symbol: Trading pair
    
    Returns:
        Dict with divergence data or None
    """
    if not USE_BINANCE_DATA:
        return None
    
    try:
        bybit_price = await fetch_ticker(symbol)
        binance_price = await fetch_binance_ticker(symbol)
        
        if bybit_price and binance_price and bybit_price > 0 and binance_price > 0:
            divergence_pct = (bybit_price - binance_price) / binance_price * 100
            
            # Determine signal
            if divergence_pct > EXCHANGE_DIVERGENCE_SIGNIFICANT_PCT:
                signal = 'bearish'  # Bybit premium = futures overleveraged
            elif divergence_pct < -EXCHANGE_DIVERGENCE_SIGNIFICANT_PCT:
                signal = 'bullish'  # Binance premium = spot accumulation
            else:
                signal = 'neutral'
            
            significant = abs(divergence_pct) > EXCHANGE_DIVERGENCE_SIGNIFICANT_PCT
            
            logging.debug(f"{symbol} exchange divergence: {divergence_pct:+.3f}% ({signal})")
            
            return {
                'bybit_price': bybit_price,
                'binance_price': binance_price,
                'divergence_pct': divergence_pct,
                'signal': signal,
                'significant': significant
            }
        
        return None
        
    except Exception as e:
        logging.debug(f"Exchange divergence calc error for {symbol}: {e}")
        return None

async def get_all_exchange_divergences() -> Dict[str, Dict]:
    """
    Get exchange divergences for all symbols.
    
    Returns:
        Dict mapping symbol -> divergence data
    """
    if not USE_BINANCE_DATA:
        return {}
    
    results = {}
    
    for symbol in SYMBOLS:
        try:
            div_data = await calculate_exchange_divergence(symbol)
            if div_data:
                results[symbol] = div_data
            await asyncio.sleep(0.2)  # Rate limit
        except Exception as e:
            logging.debug(f"Divergence fetch error for {symbol}: {e}")
            continue
    
    return results

# ============================================================================
# BINANCE FUNDING RATE (v27.8.0)
# ============================================================================
async def fetch_binance_funding_rate(symbol: str) -> Optional[Dict]:
    """
    Fetch funding rate from Binance futures.
    
    Args:
        symbol: Trading pair
    
    Returns:
        Dict with funding data or None
    """
    if not USE_BINANCE_DATA:
        return None
    
    now = time.time()
    
    cached = cache_get(binance_funding_cache, symbol)
    if cached and now - cached['timestamp'] < 300:  # 5 min cache
        return cached['data']
    
    try:
        futures = await init_binance_futures()
        futures_symbol = symbol.replace('/', '')
        
        funding_data = await futures.fetch_funding_rate(futures_symbol)
        
        if funding_data:
            rate = float(funding_data.get('fundingRate', 0))
            result = {
                'rate': rate,
                'rate_pct': rate * 100,
                'exchange': 'binance'
            }
            
            binance_funding_cache[symbol] = {'data': result, 'timestamp': now}
            return result
        
        return None
        
    except Exception as e:
        logging.debug(f"Binance funding fetch error for {symbol}: {e}")
        return None

# ============================================================================
# COMBINED FUNDING ANALYSIS (v27.8.0)
# ============================================================================
async def get_combined_funding(symbol: str) -> Optional[Dict]:
    """
    Get combined funding analysis from Bybit and Binance.
    
    Strategy:
    - Both positive = extremely long (bearish signal - fade longs)
    - Both negative = extremely short (bullish signal - fade shorts)
    - Divergent = mixed positioning
    
    Args:
        symbol: Trading pair
    
    Returns:
        Dict with combined funding analysis or None
    """
    try:
        # Get Bybit funding
        bybit_rate = 0
        
        # Bybit funding from futures exchange
        try:
            futures_symbol = symbol.replace('/', '')
            bybit_funding_data = await futures_exchange.fetch_funding_rate(futures_symbol)
            if bybit_funding_data:
                bybit_rate = float(bybit_funding_data.get('fundingRate', 0)) * 100
        except Exception:
            pass
        
        # Get Binance funding
        binance_funding = await fetch_binance_funding_rate(symbol)
        binance_rate = binance_funding['rate_pct'] if binance_funding else 0
        
        # Calculate average
        rates = [r for r in [bybit_rate, binance_rate] if r != 0]
        if not rates:
            return None
        
        avg_rate = sum(rates) / len(rates)
        
        # Determine sentiment
        if avg_rate > FUNDING_EXTREME_PCT:
            sentiment = 'extremely_long'
        elif avg_rate > FUNDING_EXTREME_PCT / 2:
            sentiment = 'long'
        elif avg_rate < -FUNDING_EXTREME_PCT:
            sentiment = 'extremely_short'
        elif avg_rate < -FUNDING_EXTREME_PCT / 2:
            sentiment = 'short'
        else:
            sentiment = 'neutral'
        
        # Check for divergence between exchanges
        funding_divergent = False
        if bybit_rate != 0 and binance_rate != 0:
            if (bybit_rate > 0 and binance_rate < 0) or (bybit_rate < 0 and binance_rate > 0):
                funding_divergent = True
        
        return {
            'bybit_rate_pct': bybit_rate,
            'binance_rate_pct': binance_rate,
            'avg_rate_pct': avg_rate,
            'sentiment': sentiment,
            'divergent': funding_divergent,
            'exchanges_reporting': len(rates)
        }
        
    except Exception as e:
        logging.debug(f"Combined funding error for {symbol}: {e}")
        return None

# ============================================================================
# VOLUME COMPARISON (v27.8.0)
# ============================================================================
async def get_volume_comparison(symbol: str) -> Optional[Dict]:
    """
    Compare 24h volume between Bybit and Binance.
    
    Higher Bybit volume = derivatives speculation
    Higher Binance volume = spot-driven move
    
    Args:
        symbol: Trading pair
    
    Returns:
        Dict with volume comparison or None
    """
    if not USE_BINANCE_DATA:
        return None
    
    try:
        # Get Bybit volume
        bybit_ticker = await exchange.fetch_ticker(symbol)
        bybit_volume = bybit_ticker.get('quoteVolume', 0) if bybit_ticker else 0
        
        # Get Binance volume
        binance = await init_binance()
        binance_ticker = await binance.fetch_ticker(symbol)
        binance_volume = binance_ticker.get('quoteVolume', 0) if binance_ticker else 0
        
        if bybit_volume == 0 and binance_volume == 0:
            return None
        
        total_volume = bybit_volume + binance_volume
        bybit_share = bybit_volume / total_volume if total_volume > 0 else 0.5
        binance_share = binance_volume / total_volume if total_volume > 0 else 0.5
        
        # Determine dominant exchange
        if bybit_share > 0.6:
            dominant = 'bybit'
            interpretation = 'derivatives_speculation'
        elif binance_share > 0.6:
            dominant = 'binance'
            interpretation = 'spot_driven'
        else:
            dominant = 'balanced'
            interpretation = 'mixed'
        
        return {
            'bybit_volume_usd': bybit_volume,
            'binance_volume_usd': binance_volume,
            'total_volume_usd': total_volume,
            'bybit_share': bybit_share,
            'binance_share': binance_share,
            'dominant': dominant,
            'interpretation': interpretation
        }
        
    except Exception as e:
        logging.debug(f"Volume comparison error for {symbol}: {e}")
        return None

# ============================================================================
# ORDER FLOW / ORDERBOOK FETCHING
# ============================================================================
async def fetch_order_flow_batch() -> Dict[str, Dict]:
    """Fetch orderbook data for all symbols."""
    async with fetch_sem:
        now = time.time()
        
        cache_hits = sum(1 for s in SYMBOLS if s in order_flow_cache and now - order_flow_cache[s]['timestamp'] < ORDER_FLOW_CACHE_TTL)
        
        if cache_hits == len(SYMBOLS):
            return {s: order_flow_cache[s]['book'] for s in SYMBOLS}
        
        if await BanManager.check_and_sleep():
            return {s: cache_get(order_flow_cache, s).get('book') if cache_get(order_flow_cache, s) else {} for s in SYMBOLS}
        
        evict_if_full(order_flow_cache)
        order_books = {}
        
        for attempt in range(3):
            try:
                tasks = [exchange.fetch_order_book(s, limit=50) for s in SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        order_books[SYMBOLS[i]] = result
                        order_flow_cache[SYMBOLS[i]] = {'book': result, 'timestamp': now}
                
                break
                
            except Exception as e:
                logging.debug(f"Order flow poll attempt {attempt+1} failed: {e}")
        
        return order_books

# ============================================================================
# OPEN INTEREST FETCHING
# ============================================================================
async def fetch_open_interest(symbol: str) -> Optional[Dict[str, float]]:
    """Fetch open interest data for futures contracts."""
    async with fetch_sem:
        if await BanManager.check_and_sleep():
            return None
        
        try:
            futures_symbol = symbol.replace('/', '')
            oi_data = await futures_exchange.fetch_open_interest(futures_symbol)
            
            oi_value = None
            if isinstance(oi_data.get('openInterest'), dict):
                oi_value = oi_data['openInterest'].get('openInterestAmount')
            elif isinstance(oi_data.get('openInterest'), (int, float)):
                oi_value = oi_data['openInterest']
            else:
                info = oi_data.get('info', {})
                raw_oi = float(info.get('sumOpenInterest') or info.get('openInterest') or '0')
                oi_value = raw_oi if raw_oi > 0 else None
            
            if oi_value is None:
                return None
            
            prev_oi = last_oi.get(futures_symbol, oi_value)
            oi_change_pct = (oi_value - prev_oi) / prev_oi * 100 if prev_oi and prev_oi != 0 else 0
            last_oi[futures_symbol] = float(oi_value)
            
            await asyncio.sleep(2)
            
            return {
                'open_interest': float(oi_value),
                'oi_change_pct': oi_change_pct
            }
            
        except Exception as e:
            logging.debug(f"OI fetch error for {symbol}: {e}")
            await asyncio.sleep(5)
            return None

# ============================================================================
# BACKGROUND PRICE POLLING - v27.12.12 with logging
# ============================================================================
async def price_background_task():
    """Background task that continuously polls prices and order flow."""
    import logging
    logging.info("Background price polling task started")
    
    while True:
        try:
            await fetch_ticker_batch()
            await fetch_order_flow_batch()
            await asyncio.sleep(10)
            
        except Exception as e:
            logging.warning(f"Background poll error: {e}")
            await asyncio.sleep(5)

# ============================================================================
# CLEANUP
# ============================================================================
async def close_exchanges():
    """Close exchange connections gracefully"""
    await exchange.close()
    await futures_exchange.close()
    if binance_exchange:
        await binance_exchange.close()
    if binance_futures:
        await binance_futures.close()
    logging.info("All exchanges closed")
