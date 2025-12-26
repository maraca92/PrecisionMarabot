# funding.py - Grok Elite Signal Bot v27.3.0 - Funding Rate Analysis
"""
Perpetual futures funding rate analysis for sentiment detection.
Extreme funding = potential reversals (fade the crowd).

v27.3.0: Funding rate integration
"""
import logging
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

from bot.data_fetcher import futures_exchange

_funding_cache: Dict[str, Dict] = {}

async def get_funding_rate(symbol: str) -> Optional[Dict[str, float]]:
    """
    Fetch current funding rate for a symbol.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
    
    Returns:
        Dict with 'rate', 'rate_pct', 'next_funding_time', or None
    """
    try:
        futures_symbol = symbol.replace('/', '')
        
        funding_data = await futures_exchange.fetch_funding_rate(futures_symbol)
        
        if not funding_data:
            return None
        
        rate = funding_data.get('fundingRate', 0)
        next_time = funding_data.get('fundingTimestamp')
        
        return {
            'rate': float(rate),
            'rate_pct': float(rate) * 100,
            'next_funding_time': datetime.fromtimestamp(next_time / 1000, tz=timezone.utc) if next_time else None,
            'timestamp': datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logging.debug(f"Funding rate fetch error for {symbol}: {e}")
        return None

async def get_funding_sentiment(symbol: str) -> Tuple[str, float]:
    """
    Classify funding rate into sentiment categories.
    
    Args:
        symbol: Trading pair
    
    Returns:
        Tuple of (sentiment_label, rate_pct)
        
    Sentiment levels:
    - 'extremely_long': rate > 0.1% (longs overleveraged)
    - 'long': rate > 0.05%
    - 'neutral': -0.05% <= rate <= 0.05%
    - 'short': rate < -0.05%
    - 'extremely_short': rate < -0.1% (shorts overleveraged)
    """
    funding = await get_funding_rate(symbol)
    
    if not funding:
        return 'unknown', 0.0
    
    rate_pct = funding['rate_pct']
    
    if rate_pct > 0.1:
        return 'extremely_long', rate_pct
    elif rate_pct > 0.05:
        return 'long', rate_pct
    elif rate_pct < -0.1:
        return 'extremely_short', rate_pct
    elif rate_pct < -0.05:
        return 'short', rate_pct
    else:
        return 'neutral', rate_pct

async def calculate_funding_confluence(symbol: str, direction: str) -> Tuple[float, str]:
    """
    Calculate funding rate confluence for a trade direction.
    
    Strategy: Fade extreme funding (counter-trend)
    - Extremely long funding + Short trade = High confluence (fade longs)
    - Extremely short funding + Long trade = High confluence (fade shorts)
    
    Args:
        symbol: Trading pair
        direction: 'Long' or 'Short'
    
    Returns:
        Tuple of (confluence_score, confluence_string)
    """
    sentiment, rate_pct = await get_funding_sentiment(symbol)
    
    confluence_score = 0.0
    confluence_str = ""
    
    if sentiment == 'extremely_long' and direction == 'Short':
        confluence_score = 3.0
        confluence_str = f"+ExtremeLongFund({rate_pct:+.3f}%)"
    elif sentiment == 'extremely_short' and direction == 'Long':
        confluence_score = 3.0
        confluence_str = f"+ExtremeShortFund({rate_pct:+.3f}%)"
    
    elif sentiment == 'long' and direction == 'Short':
        confluence_score = 1.5
        confluence_str = f"+LongFund({rate_pct:+.3f}%)"
    elif sentiment == 'short' and direction == 'Long':
        confluence_score = 1.5
        confluence_str = f"+ShortFund({rate_pct:+.3f}%)"
    
    elif sentiment != 'unknown':
        confluence_str = f"Fund:{rate_pct:+.3f}%"
    
    logging.debug(f"Funding confluence for {symbol} {direction}: {confluence_score} ({sentiment})")
    
    return confluence_score, confluence_str

async def get_funding_batch(symbols: list = None) -> Dict[str, Dict]:
    """
    Fetch funding rates for multiple symbols.
    
    Args:
        symbols: List of symbols (default: all SYMBOLS)
    
    Returns:
        Dict mapping symbol -> funding data
    """
    from bot.config import SYMBOLS
    import asyncio
    
    if symbols is None:
        symbols = SYMBOLS
    
    results = {}
    
    tasks = [get_funding_rate(s) for s in symbols]
    funding_data = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, symbol in enumerate(symbols):
        if not isinstance(funding_data[i], Exception) and funding_data[i]:
            results[symbol] = funding_data[i]
    
    return results
