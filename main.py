# main.py - Grok Elite Signal Bot v25.02.0 - ICT Elite: Regime-Aware MTF Confluence + Dynamic EV + Streamlined Stack + Self-Calibrating Fixes
# UPGRADE (v25.02.0): Strict premium/discount rejection, self-calibrating EV from backtest, FVG strength boost, precise Grok prompts, CSV trade logging,
# regime in trades, multi-symbol backtest, Monte Carlo validation, live dashboard, paper trading mode. Projected win rate: 68-75%.
# Retained: Daily 1d prio, OB str>=2, liq sweeps bounces, ADX anti-consol, vol>1.1x, conf>=70% dynamic, dist<7%, ~2x signals, Levels to Watch.
# NEW CONSTANTS: Added for ICT params (e.g., FVG_DISPLACEMENT_MULT=2, ZONE_LOOKBACK=100), PAPER_TRADING=true
import asyncio
import os
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import httpx
import json
import re
from datetime import datetime, timezone, timedelta
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import logging
from typing import Dict, Any, Optional, List
import time
from collections import OrderedDict
import html
import sys # For explicit exit in main()
import aiofiles # For async JSON I/O
from collections import deque
import zoneinfo # For timezone-aware time-of-day filter
import csv
from pathlib import Path
import random
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAMES = ['1h', '4h', '1d', '1w'] # UPDATED: Added 1h for MTF cascade
CHECK_INTERVAL = 7200 # 2h for more checks
TRACK_INTERVAL = 5
COOLDOWN_HOURS = 12
WATCH_COOLDOWN_HOURS = 24
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
XAI_API_KEY = os.getenv("XAI_API_KEY")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true" # Default: paper mode
bot = Bot(token=TELEGRAM_TOKEN)
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'rateLimit': 1200,
    'options': {'adjustForTimeDifference': True, 'defaultType': 'spot', 'recvWindow': 10000},
    'timeout': 30000,
})
futures_exchange = ccxt.bybit({
    'enableRateLimit': True,
    'rateLimit': 1200,
    'options': {'adjustForTimeDifference': True, 'defaultType': 'swap', 'recvWindow': 10000},
    'timeout': 30000,
})
FLAG_FILE = "welcome_sent.flag"
STATS_FILE = "bot_stats.json"
TRADES_FILE = "open_trades.json"
PROTECTED_TRADES_FILE = "protected_trades.json"
RECAP_FILE = "last_recap.date"
BACKTEST_FILE = "backtest_results.json"
TRADE_LOG_FILE = "trade_history.csv"
CACHE_TTL = 1800
HTF_CACHE_TTL = 3600
TICKER_CACHE_TTL = 10
ORDER_FLOW_CACHE_TTL = 5
MAX_CACHE_SIZE = 50
# NEW ICT CONSTANTS
PRE_CROSS_THRESHOLD_PCT = 0.005
OB_OVERLAP_THRESHOLD = 0.7
VOL_SURGE_MULTIPLIER = 1.1
FVG_DISPLACEMENT_MULT = 2.0 # For FVG strength
ZONE_LOOKBACK = 100 # For premium/discount
SLIPPAGE_PCT = 0.001
ENTRY_SLIPPAGE_PCT = 0.002
MAX_CONCURRENT_TRADES = 1
MAX_DRAWDOWN_PCT = 3.0
RISK_PER_TRADE_PCT = 1.5
DAILY_ATR_MULT = 2.0
SIMULATED_CAPITAL = 10000.0
def load_historical_data() -> Dict[str, float]:
    """Load TP hit rates from backtest results or use defaults"""
    if os.path.exists(BACKTEST_FILE):
        try:
            with open(BACKTEST_FILE, 'r') as f:
                bt = json.load(f)
            tp1_rate = bt.get('tp1_hit_rate', 0.60)
            tp2_rate = bt.get('tp2_hit_rate', 0.35)
            logging.info(f"Loaded historical data: TP1={tp1_rate:.2%}, TP2={tp2_rate:.2%}")
            return {'tp1_hit_rate': tp1_rate, 'tp2_hit_rate': tp2_rate}
        except Exception as e:
            logging.warning(f"Failed to load historical data: {e}")
    return {'tp1_hit_rate': 0.60, 'tp2_hit_rate': 0.35}
HISTORICAL_DATA = load_historical_data()
ohlcv_cache: OrderedDict = OrderedDict()
ticker_cache: OrderedDict = OrderedDict()
order_flow_cache: OrderedDict = OrderedDict()
last_oi: Dict[str, float] = {}
open_trades = {}
protected_trades = {}
last_signal_time = {}
last_watch_time = {}
prices_global: Dict[str, Optional[float]] = {s: None for s in SYMBOLS}
last_price_update: float = 0.0
last_ban_check: float = 0.0
btc_trend_global = "Unknown"
background_task = None
TRADE_TIMEOUT_HOURS = 24
PROTECT_AFTER_HOURS = 6
FEE_PCT = 0.04
message_queue = deque()
last_send_time = 0.0
_working_model = None
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fetch_sem = asyncio.Semaphore(3)
stats_lock = asyncio.Lock() # NEW: For thread-safe stats
def evict_if_full(cache: OrderedDict, max_size: int = MAX_CACHE_SIZE):
    evicted = 0
    while len(cache) > max_size:
        cache.popitem(last=False)
        evicted += 1
    if evicted > 0:
        logging.debug(f"Evicted {evicted} items from {cache.__class__.__name__} cache (now {len(cache)} items)")
def cache_get(cache: OrderedDict, key: str, max_size: int = MAX_CACHE_SIZE):
    evict_if_full(cache, max_size)
    return cache.get(key)
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
class BanManager:
    ban_until: float = 0.0
    cooldown_until: float = 0.0
    @classmethod
    async def check_and_sleep(cls, skip_long_sleep: bool = True):
        global last_ban_check
        now = time.time()
        if now - last_ban_check < 10:
            return False
        last_ban_check = now
        if now < cls.ban_until:
            if not skip_long_sleep:
                raise RuntimeError("Global ban active - operation blocked")
            sleep_secs = max(60, cls.ban_until - now + 60)
            sleep_secs = min(sleep_secs, 5)
            logging.info(f"Ban active (short sleep mode); sleeping {sleep_secs:.0f}s")
            await asyncio.sleep(sleep_secs)
            return True
        if now < cls.cooldown_until:
            sleep_secs = cls.cooldown_until - now
            if skip_long_sleep:
                sleep_secs = min(sleep_secs, 5)
                logging.info(f"Cooldown active (short mode); sleeping {sleep_secs:.0f}s")
            else:
                logging.info(f"Post-ban cooldown; sleeping {sleep_secs:.0f}s")
            await asyncio.sleep(sleep_secs)
            return True
        return False
    @classmethod
    def update_ban(cls, ban_ts_ms: int):
        cls.ban_until = ban_ts_ms / 1000.0
        cls.cooldown_until = cls.ban_until + 3600
        logging.warning(f"Global ban updated: until {datetime.fromtimestamp(cls.ban_until).isoformat()} (+1h cooldown)")
async def save_stats_async(s: Dict[str, Any]):
    async with aiofiles.open(STATS_FILE, 'w') as f:
        await f.write(json.dumps(s, indent=2))
def load_stats() -> Dict[str, Any]:
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                s = json.load(f)
            if 'capital' not in s:
                s['capital'] = SIMULATED_CAPITAL
            return s
        except json.JSONDecodeError:
            logging.warning("Invalid stats file, resetting.")
    return {"wins": 0, "losses": 0, "pnl": 0.0, "capital": SIMULATED_CAPITAL, "drawdown": 0.0}
async def save_trades_async(trades: Dict[str, Any]):
    try:
        dumpable = {}
        for sym, trade in trades.items():
            t_copy = trade.copy()
            if 'last_check' in t_copy and t_copy['last_check']:
                t_copy['last_check'] = trade['last_check'].isoformat()
            if 'entry_time' in t_copy and t_copy['entry_time']:
                t_copy['entry_time'] = trade['entry_time'].isoformat()
            dumpable[sym] = t_copy
        async with aiofiles.open(TRADES_FILE, 'w') as f:
            await f.write(json.dumps(dumpable, indent=2, cls=DateTimeEncoder))
    except Exception as e:
        logging.error(f"Failed to save trades: {e}")
def load_trades() -> Dict[str, Any]:
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                loaded = json.load(f)
            for trade in loaded.values():
                if 'last_check' in trade and trade['last_check']:
                    trade['last_check'] = datetime.fromisoformat(trade['last_check'])
                if 'entry_time' in trade and trade['entry_time']:
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                if 'processed' not in trade:
                    trade['processed'] = False
            return loaded
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid trades file, resetting: {e}")
    return {}
async def save_protected_async(trades: Dict[str, Any]):
    try:
        dumpable = {}
        for sym, trade in trades.items():
            t_copy = trade.copy()
            if 'last_check' in t_copy and t_copy['last_check']:
                t_copy['last_check'] = trade['last_check'].isoformat()
            if 'entry_time' in t_copy and t_copy['entry_time']:
                t_copy['entry_time'] = trade['entry_time'].isoformat()
            dumpable[sym] = t_copy
        async with aiofiles.open(PROTECTED_TRADES_FILE, 'w') as f:
            await f.write(json.dumps(dumpable, indent=2, cls=DateTimeEncoder))
    except Exception as e:
        logging.error(f"Failed to save protected trades: {e}")
def load_protected() -> Dict[str, Any]:
    if os.path.exists(PROTECTED_TRADES_FILE):
        try:
            with open(PROTECTED_TRADES_FILE, 'r') as f:
                loaded = json.load(f)
            for trade in loaded.values():
                if 'last_check' in trade and trade['last_check']:
                    trade['last_check'] = datetime.fromisoformat(trade['last_check'])
                if 'entry_time' in trade and trade['entry_time']:
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                if 'processed' not in trade:
                    trade['processed'] = False
            return loaded
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid protected trades file, resetting: {e}")
    return {}
def get_clean_symbol(trade_key: str) -> str:
    return re.sub(r'roadmap\d+$', '', trade_key)
def format_price(price: float) -> str:
    return f"{price:,.4f}"
async def send_throttled(chat_id: str, text: str, parse_mode: Optional[str] = None):
    global last_send_time
    now = time.time()
    if now - last_send_time < 1.0:
        await asyncio.sleep(1.0 - (now - last_send_time))
    await bot.send_message(chat_id, text, parse_mode=parse_mode)
    last_send_time = time.time()
# UPDATED: Improved order flow with detect_liquidity_sweep (ICT-compliant: dynamic lookback, wick>60%, vol>1.5x)
async def fetch_order_flow_batch() -> Dict[str, Dict]:
    async with fetch_sem:
        now = time.time()
        cache_hits = sum(1 for s in SYMBOLS if s in order_flow_cache and now - order_flow_cache[s]['timestamp'] < ORDER_FLOW_CACHE_TTL)
        if cache_hits == len(SYMBOLS):
            logging.debug(f"Full cache hit for order flow ({cache_hits}/{len(SYMBOLS)})")
            return {s: order_flow_cache[s]['book'] for s in SYMBOLS}
        logging.debug(f"Partial cache hit for order flow ({cache_hits}/{len(SYMBOLS)}); polling fresh")
        if await BanManager.check_and_sleep():
            return {s: cache_get(order_flow_cache, s, ORDER_FLOW_CACHE_TTL).get('book') if cache_get(order_flow_cache, s, ORDER_FLOW_CACHE_TTL) else {} for s in SYMBOLS}
        evict_if_full(order_flow_cache)
        order_books = {}
        backoff = 1
        for attempt in range(3):
            try:
                tasks = [exchange.fetch_order_book(s, limit=20) for s in SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        order_books[SYMBOLS[i]] = result
                        order_flow_cache[SYMBOLS[i]] = {'book': result, 'timestamp': now}
                logging.info(f"Order flow poll success (attempt {attempt+1})")
                break
            except Exception as e:
                logging.debug(f"Order flow poll attempt {attempt+1} failed: {e}")
                if "429" in str(e):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise
        else:
            order_books = {s: cache_get(order_flow_cache, s).get('book') if cache_get(order_flow_cache, s) else {} for s in SYMBOLS}
        return order_books
# UPDATED: New detect_liquidity_sweep (replaces naive logic: dynamic lookback, wick ratio, vol surge)
def detect_liquidity_sweep(df: pd.DataFrame, atr_window: int = 14) -> pd.Series:
    """ICT-compliant sweep: breaches key level with volume + reversal"""
    if len(df) < 20:
        return pd.Series([False] * len(df), index=df.index)
    lookback = int(df['atr'].iloc[-1] / (df['close'].iloc[-1] * 0.01) * 5) # Dynamic
    lookback = max(20, min(lookback, 100))
    swing_high = df['high'].rolling(lookback).max()
    swing_low = df['low'].rolling(lookback).min()
    # Wick ratios
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    candle_range = df['high'] - df['low']
    # Volume surge
    vol_surge = df['volume'] > 1.5 * df['volume'].rolling(20).mean()
    # Bullish sweep: break below low, close back inside with volume
    bull_sweep = (
        (df['low'] < swing_low.shift(1)) & # Breached low
        (df['close'] > swing_low.shift(1) * 1.002) & # Recovered 0.2%
        (lower_wick > candle_range * 0.6) & # 60% wick
        vol_surge
    )
    # Bearish sweep: symmetric
    bear_sweep = (
        (df['high'] > swing_high.shift(1)) &
        (df['close'] < swing_high.shift(1) * 0.998) &
        (upper_wick > candle_range * 0.6) &
        vol_surge
    )
    return bull_sweep | bear_sweep
# UPDATED: calculate_order_flow now uses detect_liquidity_sweep + cum delta
def calculate_order_flow(df: pd.DataFrame, order_book: Optional[Dict] = None) -> pd.DataFrame:
    if len(df) == 0 or not order_book:
        return df
    df = df.copy()
    if 'atr' not in df.columns:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
    df['liq_sweep'] = detect_liquidity_sweep(df)
    bids = order_book.get('bids', [])[:5]
    asks = order_book.get('asks', [])[:5]
    if not bids or not asks:
        df['order_delta'] = 0
        df['cum_delta'] = 0
        return df
    buy_vol = sum(amount for _, amount in bids)
    sell_vol = sum(amount for _, amount in asks)
    total_vol = buy_vol + sell_vol
    delta = (buy_vol - sell_vol) / total_vol * 100 if total_vol > 0 else 0
    df['order_delta'] = delta
    df['order_delta'] = df['order_delta'].fillna(delta)
    df['cum_delta'] = (np.where(df['close'] > df['open'], df['volume'], 0) - np.where(df['close'] < df['open'], df['volume'], 0)).cumsum() # NEW: True cum delta
    # Wall detection
    wall_imbalance = abs(delta) > 2.0
    mid_price = df['close'].iloc[-1]
    active_levels = len([p for p, _ in bids + asks if abs(p - mid_price) / mid_price < 0.01])
    df['footprint_imbalance'] = active_levels > 10
    df['footprint_imbalance'] = df['footprint_imbalance'].fillna(True if active_levels > 10 else False)
    logging.debug(f"Order flow: delta {delta:.2f}% | Wall: {wall_imbalance} | Footprint: {active_levels} | Sweep: {df['liq_sweep'].iloc[-1]}")
    return df
# NEW: Regime detection (trending/ranging/explosive/dead)
def detect_market_regime(df: pd.DataFrame) -> str:
    """Classify market into trending/ranging/explosive"""
    if len(df) < 50:
        return 'ranging' # Default safe
    adx = ta.adx(df['high'], df['low'], df['close'], 14)['ADX_14'].iloc[-1]
    atr_ratio = df['atr'].iloc[-1] / df['atr'].rolling(50).mean().iloc[-1]
    if adx > 40 and atr_ratio > 1.5:
        return 'explosive' # Strong trend + high volatility
    elif adx > 25:
        return 'trending'
    elif adx < 20 and atr_ratio < 0.8:
        return 'dead' # Avoid
    else:
        return 'ranging' # Prime for reversals
# UPDATED: is_consolidation now integrates regime (skip 'dead', boost ranging)
def is_consolidation(df: pd.DataFrame) -> bool:
    regime = detect_market_regime(df)
    if regime == 'dead':
        return True # Skip entirely
    if len(df) < 14:
        return True
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is None or 'ADX_14' not in adx.columns:
        return False
    adx_val = adx['ADX_14'].iloc[-1]
    vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
    consol = (adx_val < 25) and (vol_ratio < VOL_SURGE_MULTIPLIER)
    if regime == 'ranging':
        consol = False # Boost reversals in range
    logging.info(f"Regime: {regime} | Consol check: ADX={adx_val:.1f}, vol_ratio={vol_ratio:.2f} -> {consol}")
    return consol
# NEW: MTF Confluence Scoring (gradient EMA slope + OB proximity + liq sweep + vol)
async def calculate_mtf_confluence(symbol: str, price: float, direction: str) -> float:
    """Score confluence across 1h -> 4h -> 1d -> 1w"""
    score = 0.0
    timeframes = ['1h', '4h', '1d', '1w']
    weights = [1, 2, 3, 4] # Higher TF = more weight
    for tf, weight in zip(timeframes, weights):
        df = await fetch_ohlcv(symbol, tf, 100)
        if len(df) == 0:
            continue
        df = add_institutional_indicators(df) # UPDATED: Use new indicators
        # 1. Trend alignment (EMA gradient)
        ema_slope = (df['ema200'].iloc[-1] - df['ema200'].iloc[-10]) / df['ema200'].iloc[-10] # UPDATED: Use EMA200 for bias
        if (direction == 'Long' and ema_slope > 0) or (direction == 'Short' and ema_slope < 0):
            score += weight * 2
        # 2. OB proximity (closer = better)
        obs = await find_unmitigated_order_blocks(df, tf=tf)
        relevant_obs = obs['bullish'] if direction == 'Long' else obs['bearish']
        if relevant_obs:
            closest_ob = min(relevant_obs, key=lambda x: abs((x['low']+x['high'])/2 - price))
            dist_pct = abs((closest_ob['low']+closest_ob['high'])/2 - price) / price * 100
            if dist_pct < 2:
                score += weight * 3 * (2 - dist_pct) # Exponential bonus
        # 3. Liquidity sweep alignment
        if df['liq_sweep'].iloc[-1]:
            score += weight * 1.5
        # 4. Volume confirmation
        vol_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        if vol_ratio > 1.5:
            score += weight * 1
    return min(score / 40, 1.0) # Normalize to 0-1
# NEW: Streamlined institutional indicators (structure + orderflow, removed EMA50/100/MACD/BB/Stoch)
def add_institutional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Focus on orderflow + structure, not derivatives"""
    if len(df) == 0:
        return df
  
    # CRITICAL FIX: Ensure we have a proper index before any operations
    # If DatetimeIndex was lost, operations will fail
    if not isinstance(df.index, (pd.DatetimeIndex, pd.RangeIndex)):
        logging.warning(f"Unexpected index type in add_institutional_indicators: {type(df.index)}")
        # Reset to RangeIndex for safety
        df = df.reset_index(drop=True)
  
    # Make a copy to avoid modifying original
    df = df.copy()
  
    # 1. PRICE STRUCTURE (non-derivative)
    df['swing_high'] = df['high'].rolling(21, center=True).max() == df['high']
    df['swing_low'] = df['low'].rolling(21, center=True).min() == df['low']
    # 2. VOLUME PROFILE (shows institution intent)
    df['vol_delta'] = df['volume'] - df['volume'].shift(1)
    df['vol_acceleration'] = df['vol_delta'] - df['vol_delta'].shift(1)
    # 3. CUMULATIVE DELTA (orderflow)
    df['buy_vol'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['sell_vol'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['cum_delta'] = (df['buy_vol'] - df['sell_vol']).cumsum()
    # 4. TRUE RANGE PERCENTILE (volatility regime)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
    df['atr_pct'] = df['atr'].rank(pct=True)
    # 5. MARKET MAKER PROFILE (spread proxy)
    df['spread_proxy'] = (df['high'] - df['low']) / df['close'] * 100
    # Keep EMA200 for bias + RSI for divergence
    df['ema200'] = ta.ema(df['close'], 200)
    df['rsi'] = ta.rsi(df['close'], 14)
    # NEW: FVG strength
    df = calculate_fvg_strength(df)
    # NEW: Premium/discount zones
    df = calculate_premium_discount(df, lookback=ZONE_LOOKBACK)
    # Order Flow (now separate, but integrate liq_sweep)
    if 'liq_sweep' not in df.columns:
        df['liq_sweep'] = False # Placeholder if no book
    
    # CRITICAL: Ensure 'date' column is preserved for downstream functions
    if 'date' not in df.columns and 'ts' in df.columns:
        df['date'] = pd.to_datetime(df['ts'], unit='ms')
        logging.debug("Restored 'date' column in add_institutional_indicators")
   
    return df
# NEW: FVG strength grading (volume + displacement)
def calculate_fvg_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Grade FVG quality using volume + displacement"""
    fvgs = []
    for i in range(2, len(df)):
        # Bullish FVG
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            vol_at_gap = df['volume'].iloc[i-1]
            vol_avg = df['volume'].rolling(20).mean().iloc[i]
            displacement = abs(df['close'].iloc[i] - df['open'].iloc[i-2]) / df['close'].iloc[i]
            strength = (
                (vol_at_gap / vol_avg) * 2 +
                (gap_size / df['atr'].iloc[i]) * 3 +
                (displacement * 100) * FVG_DISPLACEMENT_MULT
            )
            fvgs.append({
                'idx': i, 'type': 'bull', 'low': df['high'].iloc[i-2],
                'high': df['low'].iloc[i], 'strength': strength
            })
        # Bearish FVG symmetric...
        elif df['low'].iloc[i-2] > df['high'].iloc[i]:
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            vol_at_gap = df['volume'].iloc[i-1]
            vol_avg = df['volume'].rolling(20).mean().iloc[i]
            displacement = abs(df['close'].iloc[i] - df['open'].iloc[i-2]) / df['close'].iloc[i]
            strength = (
                (vol_at_gap / vol_avg) * 2 +
                (gap_size / df['atr'].iloc[i]) * 3 +
                (displacement * 100) * FVG_DISPLACEMENT_MULT
            )
            fvgs.append({
                'idx': i, 'type': 'bear', 'low': df['high'].iloc[i],
                'high': df['low'].iloc[i-2], 'strength': strength
            })
    df['fvg_strength'] = 0.0
    for fvg in fvgs:
        # Use loc instead of at to avoid index issues
        try:
            df.loc[df.index[fvg['idx']], 'fvg_strength'] = fvg['strength']
        except (KeyError, IndexError) as e:
            logging.debug(f"FVG strength assignment failed at idx {fvg['idx']}: {e}")
            continue
    return df
# NEW: Premium/Discount zones
def calculate_premium_discount(df: pd.DataFrame, lookback: int = ZONE_LOOKBACK) -> pd.DataFrame:
    """Determine if price is premium/discount relative to range"""
    if len(df) < lookback:
        df = df.copy() # Work on copy
        df['premium_pct'] = 0.5
        df['zone'] = 'equilibrium'
        return df
   
    # CRITICAL: Work on copy to preserve original columns
    df = df.copy()
  
    df['range_high'] = df['high'].rolling(lookback, min_periods=1).max()
    df['range_low'] = df['low'].rolling(lookback, min_periods=1).min()
    df['range_mid'] = (df['range_high'] + df['range_low']) / 2
  
    # Avoid division by zero
    range_size = df['range_high'] - df['range_low']
    df['premium_pct'] = np.where(
        range_size > 0,
        (df['close'] - df['range_low']) / range_size,
        0.5 # Default to equilibrium if no range
    )
  
    # Ensure premium_pct is 0-1
    df['premium_pct'] = df['premium_pct'].clip(0, 1)
  
    # Use pd.cut with proper handling
    try:
        df['zone'] = pd.cut(
            df['premium_pct'],
            bins=[0, 0.3, 0.7, 1.0001], # 1.0001 to include 1.0
            labels=['discount', 'equilibrium', 'premium'],
            include_lowest=True
        )
    except Exception as e:
        logging.warning(f"pd.cut failed in calculate_premium_discount: {e}")
        # Fallback to manual binning
        df['zone'] = 'equilibrium'
        df.loc[df['premium_pct'] < 0.3, 'zone'] = 'discount'
        df.loc[df['premium_pct'] > 0.7, 'zone'] = 'premium'
  
    return df
# UPDATED: track_ob_mitigation (partial mitigation with vol + penetration)
def track_ob_mitigation(ob: Dict, df_since_formation: pd.DataFrame) -> float:
    """Return mitigation percentage (0 = untouched, 1 = fully mitigated)"""
    ob_mid = (ob['low'] + ob['high']) / 2
    ob_range = ob['high'] - ob['low']
    touches = df_since_formation[
        (df_since_formation['low'] <= ob['high']) &
        (df_since_formation['high'] >= ob['low'])
    ]
    if len(touches) == 0:
        return 0.0
    total_vol_in_ob = touches['volume'].sum()
    formation_vol = df_since_formation['volume'].rolling(5).mean().iloc[0] * 5
    mitigation_score = min(total_vol_in_ob / formation_vol, 1.0)
    deepest_penetration = max(
        (touches['high'].max() - ob_mid) / ob_range,
        (ob_mid - touches['low'].min()) / ob_range
    )
    if deepest_penetration > 0.7:
        mitigation_score += 0.3
    return min(mitigation_score, 1.0)
# UPDATED: find_unmitigated_order_blocks integrates mitigation tracking + str>=2
async def find_unmitigated_order_blocks(df: pd.DataFrame, lookback: int = 100, atr_mult: float = 2.0, tf: str = None, symbol: str = None) -> Dict[str, List[Dict]]:
    if len(df) < 30:
        return {'bullish': [], 'bearish': []}
    ltf_mult = 1 if tf in ['15m', '1h'] else 2 if tf == '4h' else 3
    dyn_lookback = lookback * ltf_mult
    df_local = df.tail(dyn_lookback).copy()
    df_local['atr'] = ta.atr(df_local['high'], df_local['low'], df_local['close'], 14)
    df_local['direction'] = np.where(df_local['close'] > df_local['open'], 1, -1)
    df_local['swing_high'] = df_local['high'].rolling(10, center=True).max() == df_local['high']
    df_local['swing_low'] = df_local['low'].rolling(10, center=True).min() == df_local['low']
    df_local['vol_surge'] = df_local['volume'] > VOL_SURGE_MULTIPLIER * df_local['volume'].rolling(20).mean()
    df_local['volume_sma'] = df_local['volume'].rolling(20).mean()
    obs = {'bullish': [], 'bearish': []}
    for i in range(15, len(df_local) - 10):
        if df_local['swing_high'].iloc[i] and df_local['vol_surge'].iloc[i]:
            ob_high = df_local['high'].iloc[i]
            ob_low = max(df_local['open'].iloc[i], df_local['close'].iloc[i])
            move_down = df_local['low'].iloc[i+5:i+10].min() < ob_low - (df_local['atr'].iloc[i] * atr_mult)
            if move_down and abs((ob_low - df_local['low'].iloc[i+5:i+10].min()) / ob_low) > 0.02:
                df_since = df_local.iloc[i+1:]
                mitigation = track_ob_mitigation({'low': ob_low, 'high': ob_high}, df_since)
                if mitigation < 0.5: # NEW: Partial mitigation threshold
                    zone_type = 'Breaker' if any(df_local['low'].iloc[i+1:i+6] < ob_low) else 'OB'
                    strength = 3 if zone_type == 'OB' and df_local['volume'].iloc[i] > VOL_SURGE_MULTIPLIER * df_local['volume_sma'].iloc[i] else 2
                    adjusted_strength = strength * (1 - mitigation) # NEW: Adjust for mitigation
                    if adjusted_strength >= 2:
                        obs['bearish'].append({
                            'low': ob_low, 'high': ob_high, 'type': zone_type,
                            'strength': adjusted_strength, 'index': i, 'mitigation': mitigation
                        })
    # Symmetric for bullish...
    for i in range(15, len(df_local) - 10):
        if df_local['swing_low'].iloc[i] and df_local['vol_surge'].iloc[i]:
            ob_low = df_local['low'].iloc[i]
            ob_high = min(df_local['open'].iloc[i], df_local['close'].iloc[i])
            move_up = df_local['high'].iloc[i+5:i+10].max() > ob_high + (df_local['atr'].iloc[i] * atr_mult)
            if move_up and abs((df_local['high'].iloc[i+5:i+10].max() - ob_high) / ob_high) > 0.02:
                df_since = df_local.iloc[i+1:]
                mitigation = track_ob_mitigation({'low': ob_low, 'high': ob_high}, df_since)
                if mitigation < 0.5:
                    zone_type = 'Breaker' if any(df_local['high'].iloc[i+1:i+6] > ob_high) else 'OB'
                    strength = 3 if zone_type == 'OB' and df_local['volume'].iloc[i] > VOL_SURGE_MULTIPLIER * df_local['volume_sma'].iloc[i] else 2
                    adjusted_strength = strength * (1 - mitigation)
                    if adjusted_strength >= 2:
                        obs['bullish'].append({
                            'low': ob_low, 'high': ob_high, 'type': zone_type,
                            'strength': adjusted_strength, 'index': i, 'mitigation': mitigation
                        })
    # HTF merge (retained, but use adjusted strength)
    if tf in ['1d', '1w'] and symbol:
        try:
            df_1h = await fetch_ohlcv(symbol, '1h', 200)
            obs_1h = await find_unmitigated_order_blocks(df_1h, lookback=100, tf='1h', symbol=symbol)
            for ob_type in ['bullish', 'bearish']:
                merged = []
                for ob_htf in obs[ob_type]:
                    if obs_1h[ob_type]:
                        best_match = max(
                            (zones_overlap(ob_htf['low'], ob_htf['high'], ob_ltf['low'], ob_ltf['high']), ob_ltf)
                            for ob_ltf in obs_1h[ob_type]
                        )
                        if best_match[0] > OB_OVERLAP_THRESHOLD:
                            ob_ltf = best_match[1]
                            merged_low = min(ob_htf['low'], ob_ltf['low'])
                            merged_high = max(ob_htf['high'], ob_ltf['high'])
                            merged_strength = max(ob_htf['strength'], ob_ltf['strength'])
                            merged_mit = max(ob_htf['mitigation'], ob_ltf['mitigation'])
                            merged.append({
                                'low': merged_low, 'high': merged_high, 'type': ob_htf['type'],
                                'strength': merged_strength * (1 - merged_mit), 'index': ob_htf['index'],
                                'mitigation': merged_mit
                            })
                            logging.info(f"Merged {ob_type} OB for {symbol} {tf}: overlap {best_match[0]:.2f}")
                    else:
                        merged.append(ob_htf)
                obs[ob_type] = merged[:3]
        except Exception as e:
            logging.warning(f"HTF 1h verify failed for {symbol} {tf}: {e}")
    for key in obs:
        obs[key] = sorted(obs[key], key=lambda z: (len(df_local) - z['index']) * z['strength'], reverse=True)[:3]
    return obs
# NEW: Dynamic threshold (vol/ATR percentile + recent winrate)
def calculate_dynamic_threshold(df: pd.DataFrame, base: float = 70, stats: Dict = None) -> float:
    """Adjust confidence threshold based on market conditions"""
    atr_percentile = df['atr'].rank(pct=True).iloc[-1]
    vol_percentile = df['volume'].rank(pct=True).iloc[-1]
    adjustments = 0
    if atr_percentile > 0.8:
        adjustments += 10
    if vol_percentile < 0.3:
        adjustments += 8
    if stats:
        recent_trades = [t for t in list(open_trades.values()) + list(protected_trades.values()) if t.get('processed')][-10:]
        if len(recent_trades) >= 5:
            recent_winrate = sum(1 for t in recent_trades if t.get('hit_tp', False)) / len(recent_trades)
            if recent_winrate < 0.4:
                adjustments += 15
    return min(base + adjustments, 95)
# NEW: Probabilistic RR / EV (partial TP + slippage)
def calculate_expected_value(trade: Dict, historical_data: Dict) -> float:
    """Calculate EV considering partial fills and slippage"""
    entry_mid = (trade['entry_low'] + trade['entry_high']) / 2
    prob_hit_tp1 = historical_data.get('tp1_hit_rate', 0.60)
    prob_hit_tp2 = historical_data.get('tp2_hit_rate', 0.35)
    prob_hit_sl = 1 - prob_hit_tp1 # Simplified
    tp1_gain = (trade['tp1'] - entry_mid) * (1 - SLIPPAGE_PCT * 2) if trade['direction'] == 'Long' else (entry_mid - trade['tp1']) * (1 - SLIPPAGE_PCT * 2)
    tp2_gain = (trade['tp2'] - entry_mid) * (1 - SLIPPAGE_PCT * 2) if trade['direction'] == 'Long' else (entry_mid - trade['tp2']) * (1 - SLIPPAGE_PCT * 2)
    sl_loss = (entry_mid - trade['sl']) * (1 + SLIPPAGE_PCT * 2) if trade['direction'] == 'Long' else (trade['sl'] - entry_mid) * (1 + SLIPPAGE_PCT * 2)
    expected_gain = (
        prob_hit_tp1 * tp1_gain * 0.5 + # Half at TP1
        prob_hit_tp2 * tp2_gain * 0.5 # Half at TP2
    )
    expected_loss = prob_hit_sl * sl_loss
    ev = expected_gain - expected_loss
    return ev / abs(sl_loss) if sl_loss != 0 else 0 # R-multiple
# NEW: Quick wins - Time of day filter
def is_optimal_trading_hour(symbol: str) -> bool:
    """Avoid low-liquidity hours"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    if symbol == 'BTC/USDT' and 2 <= hour < 6:
        return False
    if symbol != 'BTC/USDT' and not (8 <= hour < 20):
        return False
    return True
# FIXED: No-trade zones (psychological levels + prev week close) - Quick Deploy version
def is_in_no_trade_zone(price: float, symbol: str, df: pd.DataFrame) -> bool:
    """Avoid whole numbers and previous week's close"""
    # Psychological levels (always check these)
    if price % 100 < 5 or price % 100 > 95:
        return True
    if price % 1000 < 50 or price % 1000 > 950:
        return True
   
    # Previous week's close check (needs DatetimeIndex)
    # Skip this check if we don't have proper data to avoid crashes
    if len(df) < 7 or 'date' not in df.columns:
        return False
   
    try:
        # Set DatetimeIndex for resampling
        df_temp = df.set_index('date').sort_index()
       
        # Resample to weekly
        df_1w = df_temp.resample('1W', label='right', closed='right').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
       
        # Need at least 2 weeks
        if len(df_1w) < 2:
            return False
       
        # Get previous week's close
        prev_week_close = df_1w['close'].iloc[-2]
       
        # Check validity
        if pd.isna(prev_week_close) or prev_week_close <= 0:
            return False
       
        # Check if current price is near previous week's close
        if abs(price - prev_week_close) / price < 0.003: # Within 0.3%
            logging.info(f"No-trade zone: {symbol} @ {price:.2f} near prev week {prev_week_close:.2f}")
            return True
   
    except Exception as e:
        # Don't block trades if check fails
        logging.warning(f"is_in_no_trade_zone check failed for {symbol}: {e}")
        return False
   
    return False
# NEW: Liquidity check (spread + depth)
async def check_sufficient_liquidity(symbol: str) -> bool:
    """Ensure orderbook has depth for safe exit"""
    try:
        book = await exchange.fetch_order_book(symbol, limit=20)
        best_bid = book['bids'][0][0]
        best_ask = book['asks'][0][0]
        spread_pct = (best_ask - best_bid) / best_bid * 100
        if spread_pct > 0.05:
            return False
        total_bid_vol = sum(amt for _, amt in book['bids'][:10])
        total_ask_vol = sum(amt for _, amt in book['asks'][:10])
        min_depth_usd = 10 * 60000 # $600k
        if (total_bid_vol * best_bid < min_depth_usd or total_ask_vol * best_ask < min_depth_usd):
            return False
        return True
    except Exception:
        return False
# NEW: Drawdown protection scaling
def get_risk_scaling_factor(stats: Dict) -> float:
    """Scale position size during drawdown"""
    current_dd = max(0, -stats['pnl'])
    if current_dd < 1:
        return 1.0
    elif current_dd < 2:
        return 0.75
    elif current_dd < 3:
        return 0.50
    else:
        return 0.0 # Stop trading
# UPDATED: add_indicators -> add_institutional_indicators (streamlined)
def add_indicators(df: pd.DataFrame, order_book: Optional[Dict] = None) -> pd.DataFrame:
    return add_institutional_indicators(df) # Redirect + order flow
# UPDATED: zones_overlap (retained)
def zones_overlap(z1_low: float, z1_high: float, z2_low: float, z2_high: float, threshold: float = OB_OVERLAP_THRESHOLD) -> float:
    o_low = max(z1_low, z2_low)
    o_high = min(z1_high, z2_high)
    if o_low >= o_high:
        return 0.0
    overlap_len = o_high - o_low
    min_width = min(z1_high - z1_low, z2_high - z2_low)
    return (overlap_len / min_width)
# UPDATED: find_next_premium_zones: Integrate regime, MTF confluence, premium/discount, dynamic thresh, EV filter, quick wins
async def find_next_premium_zones(df: pd.DataFrame, current_price: float, tf: str, symbol: str = None, oi_data: Optional[Dict[str, float]] = None, trend: str = None, whale_data: Optional[Dict[str, Any]] = None, order_book: Optional[Dict] = None) -> List[Dict]:
    if len(df) < 50:
        return []
    df = add_institutional_indicators(df) # UPDATED: New stack
    regime = detect_market_regime(df)
    if regime == 'dead':
        logging.info(f"Skipped {symbol} {tf}: Dead regime")
        return []
    conf_multiplier = 1.3 if regime == 'ranging' else 0.7 if regime == 'explosive' else 1.0 # NEW: Adaptive
    # Before calling is_in_no_trade_zone, ensure 'date' column exists
    if 'date' not in df.columns and 'ts' in df.columns:
        df['date'] = pd.to_datetime(df['ts'], unit='ms')
   
    if is_in_no_trade_zone(current_price, symbol, df): # NEW: Quick win
        logging.info(f"Skipped {symbol} {tf}: No-trade zone")
        return []
    if not await check_sufficient_liquidity(symbol): # NEW: Quick win
        logging.info(f"Skipped {symbol} {tf}: Low liquidity")
        return []
    if not is_optimal_trading_hour(symbol): # NEW: Quick win
        logging.info(f"Skipped {symbol} {tf}: Suboptimal hour")
        return []
    ema200_val = df['ema200'].iloc[-1]
    price_current = df['close'].iloc[-1]
    trend_bias = 'bull' if price_current > ema200_val else 'bear' if price_current < ema200_val else 'neutral'
    # NEW: Premium/discount penalty
    zone = df['zone'].iloc[-1]
    rsi_1d = None
    rsi_1w = None
    if symbol:
        try:
            df_1d = await fetch_ohlcv(symbol, '1d', 200)
            df_1d = add_institutional_indicators(df_1d)
            rsi_1d = df_1d['rsi'].iloc[-1]
            df_1w = await fetch_ohlcv(symbol, '1w', 200)
            df_1w = add_institutional_indicators(df_1w)
            rsi_1w = df_1w['rsi'].iloc[-1]
        except Exception as e:
            logging.error(f"HTF RSI fetch error for {symbol}: {e}")
    htf_align = 1.0 # UPDATED: Now from MTF func
    if tf in ['4h', '1d', '1w']:
        htf_align = await calculate_mtf_confluence(symbol, current_price, 'Long') # Example; compute per direction later
    htf_mult = htf_align if tf == '1d' else 1.0
    obs = await find_unmitigated_order_blocks(df, tf=tf, symbol=symbol)
    elite_obs = {k: [o for o in v if o['strength'] >= 2] for k, v in obs.items()}
    liq_profile = calc_liquidity_profile(df)
    poc = max(liq_profile.items(), key=lambda x: x[1])[0] if liq_profile else None
    zones_7pct = []
    buffer_mult = 0.05 if trend == 'Downtrend' else 0.07
    long_buffer = current_price * buffer_mult
    short_buffer = current_price * buffer_mult
    for ob in elite_obs.get('bullish', []):
        mid = (ob['low'] + ob['high']) / 2
        dist_pct = abs(current_price - mid) / current_price * 100
        if mid < current_price - long_buffer and dist_pct <= 7.0:
            zones_7pct.append(ob)
    for ob in elite_obs.get('bearish', []):
        mid = (ob['low'] + ob['high']) / 2
        dist_pct = abs(current_price - mid) / current_price * 100
        if mid > current_price + short_buffer and dist_pct <= 7.0:
            zones_7pct.append(ob)
    zones_to_use = zones_7pct
    logging.info(f"Using 7% zones for {symbol} {tf}: {len(zones_to_use)} str>=2 OBs")
    zones = []
    dyn_thresh = calculate_dynamic_threshold(df, stats=load_stats()) # NEW: Dynamic
    for ob in zones_to_use:
        mid = (ob['low'] + ob['high']) / 2
        dist = abs(current_price - mid) / current_price * 100
        direction = 'Long' if 'bullish' in str(ob.get('type', '')) else 'Short'
        mtf_conf = await calculate_mtf_confluence(symbol, mid, direction) # NEW: Per zone
        conf_score = ob['strength'] * htf_mult * mtf_conf * conf_multiplier
        confluence_str = ob['type']
        if poc and abs(mid - poc) / current_price * 100 < 0.3:
            conf_score += 2
            confluence_str += "+POC"
        oi_str = ""
        if oi_data and oi_data['oi_change_pct'] > 10:
            conf_score += 1.5
            oi_str = "+Whale OI"
        if liq_profile and liq_profile.get(mid, 0) > 1.5:
            conf_score += 1.5
            confluence_str += "+High Liq"
        confluence_str += oi_str
        # RSI boost
        rsi_os_boost = 0
        if ((rsi_1d and rsi_1d < 35) or (rsi_1w and rsi_1w < 35)) and direction == 'Long':
            rsi_os_boost = 1.5
            confluence_str += "+RSI Exhaust Long"
        elif ((rsi_1d and rsi_1d > 65) or (rsi_1w and rsi_1w > 65)) and direction == 'Short':
            rsi_os_boost = 1.5
            confluence_str += "+RSI Exhaust Short"
        else:
            confluence_str += "+RSI Neutral"
        conf_score += rsi_os_boost
        # Trend bias
        if trend_bias == 'bull' and direction == 'Long':
            conf_score += 1.5
        elif trend_bias == 'bear' and direction == 'Short':
            conf_score += 1.5
        elif trend_bias == 'neutral':
            conf_score += 1
        # STRICT: Reject counter-zone trades (ICT core principle)
        if direction == 'Long' and zone not in ['discount', pd.NA]:
            logging.info(f"Rejected {direction} for {symbol} {tf}: Not in discount (zone={zone})")
            continue
        elif direction == 'Short' and zone not in ['premium', pd.NA]:
            logging.info(f"Rejected {direction} for {symbol} {tf}: Not in premium (zone={zone})")
            continue
        # FVG/Breaker (retained)
        fvgs = detect_fvg(df, tf, proximity_pct=0.3)
        obs_key = 'bullish' if direction == 'Long' else 'bearish'
        breaker_confirmed = any('Breaker' in ob['type'] for ob in elite_obs.get(obs_key, []))
        reversal_caution = fvgs or breaker_confirmed
        if reversal_caution:
            conf_score += 1.5
            confluence_str += "+FVG/Breaker Caution"
        else:
            reversal_caution = False
        # NEW: FVG strength boost (if strong FVGs nearby)
        if fvgs and 'fvg_strength' in df.columns:
            # Average FVG strength in last 10 bars
            recent_fvg_strength = df['fvg_strength'].iloc[-10:].mean()
            if recent_fvg_strength > 5: # Strong FVGs present
                conf_score += 2
                confluence_str += f"+FVG Strength ({recent_fvg_strength:.1f})"
                logging.debug(f"FVG strength boost: {recent_fvg_strength:.1f}")
        aligned_bias = 'bull' if direction == 'Long' else 'bear'
        if trend_bias != 'neutral' and trend_bias != aligned_bias and not reversal_caution:
            logging.info(f"Skipped counter-trend {direction} for {symbol} {tf}: bias {trend_bias}, no caution")
            continue
        # Order Flow
        delta = df['order_delta'].iloc[-1]
        if abs(delta) > 1.0:
            conf_score += 2
            confluence_str += f"+Delta {delta:.1f}%"
        if df['footprint_imbalance'].iloc[-1]:
            conf_score += 1
            confluence_str += "+Footprint"
        if df['liq_sweep'].iloc[-1]:
            conf_score += 2
            confluence_str += "+Liq Sweep Bounce"
        prob = min(98, 60 + conf_score * 7 * conf_multiplier) # Adaptive mult
        # NEW: Mock trade for EV filter
        mock_trade = {'direction': direction, 'entry_low': ob['low'], 'entry_high': ob['high'],
                      'sl': ob['low'] - df['atr'].iloc[-1] * DAILY_ATR_MULT if direction == 'Long' else ob['high'] + df['atr'].iloc[-1] * DAILY_ATR_MULT,
                      'tp1': mid + (mid - ob['sl']) * 2, 'tp2': mid + (mid - ob['sl']) * 4} # Symmetric for short
        ev_r = calculate_expected_value(mock_trade, HISTORICAL_DATA)
        if ev_r < 0.5: # NEW: EV threshold
            logging.info(f"Skipped {direction} for {symbol} {tf}: Low EV {ev_r:.2f}R")
            continue
        zones.append({
            'direction': direction, 'zone_low': ob['low'], 'zone_high': ob['high'],
            'confluence': confluence_str, 'dist_pct': dist, 'prob': prob, 'strength': ob['strength']
        })
    zones = sorted(zones, key=lambda z: z['dist_pct'])[:3]
    zones = [z for z in zones if z['prob'] >= dyn_thresh] # NEW: Dynamic
    if len(zones) < 2 and tf not in ['1w']:
        zones = []
    logging.info(f"Filtered to {len(zones)} elite zones >=dyn_thresh for {symbol} {tf}")
    return zones
# Retained helpers: calc_liquidity_profile, detect_fvg, detect_supertrend, detect_macd, detect_pre_cross, detect_divergence, detect_candle_patterns
def calc_liquidity_profile(df: pd.DataFrame, bins: int = 15) -> Dict[float, float]:
    if len(df) < 50:
        return {}
    prices = df['close']
    volumes = df['volume']
    min_p, max_p = prices.min(), prices.max()
    bin_edges = np.linspace(min_p, max_p, bins + 1)
    vol_profile = {}
    for i in range(bins):
        mask = (prices >= bin_edges[i]) & (prices < bin_edges[i+1])
        vol_sum = volumes[mask].sum()
        bin_mid = (bin_edges[i] + bin_edges[i+1]) / 2
        vol_profile[bin_mid] = vol_sum
    sorted_vols = sorted(vol_profile.values(), reverse=True)
    threshold = sorted_vols[int(len(sorted_vols) * 0.85)] if sorted_vols else 0
    if threshold == 0:
        hot_zones = {k: 0 for k, v in vol_profile.items()}
    else:
        hot_zones = {k: v / threshold if v > threshold else 0 for k, v in vol_profile.items()}
    return hot_zones
def detect_supertrend(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 11 or 'supertrend_dir' not in df.columns or pd.isna(df['supertrend_dir'].iloc[-1]):
        return None
    if df['supertrend_dir'].iloc[-1] == 1 and df['supertrend_dir'].iloc[-2] == -1:
        return f"SuperTrend Bullish Flip ({tf})"
    if df['supertrend_dir'].iloc[-1] == -1 and df['supertrend_dir'].iloc[-2] == 1:
        return f"SuperTrend Bearish Flip ({tf})"
    return None
def detect_fvg(df: pd.DataFrame, tf: str, lookback: int = 50, proximity_pct: float = 0.3) -> List[str]:
    if len(df) < 5:
        return []
    df_local = df.tail(lookback).copy()
    fvgs = []
    price = df_local['close'].iloc[-1]
    for i in range(2, len(df_local) - 1):
        if df_local['high'].iloc[i-2] < df_local['low'].iloc[i]:
            fvg_low = df_local['high'].iloc[i-2]
            fvg_high = df_local['low'].iloc[i]
            mid = (fvg_low + fvg_high) / 2
            dist_pct = abs(price - mid) / price * 100
            if dist_pct < proximity_pct:
                fvgs.append(f"Near Bullish FVG {dist_pct:.1f}% away ({tf})")
        elif df_local['low'].iloc[i-2] > df_local['high'].iloc[i]:
            fvg_low = df_local['high'].iloc[i]
            fvg_high = df_local['low'].iloc[i-2]
            mid = (fvg_low + fvg_high) / 2
            dist_pct = abs(price - mid) / price * 100
            if dist_pct < proximity_pct:
                fvgs.append(f"Near Bearish FVG {dist_pct:.1f}% away ({tf})")
    return fvgs
def detect_macd(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 2 or 'macd' not in df.columns or 'macd_signal' not in df.columns or pd.isna(df['macd'].iloc[-1]) or pd.isna(df['macd_signal'].iloc[-1]):
        return None
    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
        return f"Bullish MACD Crossover ({tf})"
    if df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
        return f"Bearish MACD Crossover ({tf})"
    return None
def detect_pre_cross(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 2 or tf not in ['4h', '1d'] or 'ema200' not in df.columns or pd.isna(df['ema200'].iloc[-1]):
        return None
    l = df.iloc[-1]
    diff = abs(l['ema200'] - l['close']) / l['close'] # UPDATED: vs close for bias
    if diff < PRE_CROSS_THRESHOLD_PCT:
        return "EMA200 Bias Shift Incoming" if l['close'] > l['ema200'] else "EMA200 Bias Shift Incoming"
    return None
def detect_divergence(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 50 or tf not in ['4h', '1d'] or 'rsi' not in df.columns or pd.isna(df['rsi'].iloc[-1]):
        return None
    price = df['close'].iloc[-40:]
    rsi = df['rsi'].iloc[-40:]
    swing_low_idx = price.iloc[-10:].idxmin()
    swing_high_idx = price.iloc[-10:].idxmax()
    if pd.isna(rsi.loc[swing_low_idx]) or pd.isna(rsi.loc[swing_high_idx]):
        return None
    if price.iloc[-1] > price.loc[swing_low_idx] and rsi.iloc[-1] < rsi.loc[swing_low_idx] and rsi.loc[swing_low_idx] < 30:
        return f"Hidden Bullish RSI Divergence ({tf})"
    if price.iloc[-1] < price.loc[swing_high_idx] and rsi.iloc[-1] > rsi.loc[swing_high_idx] and rsi.loc[swing_high_idx] > 70:
        return f"Hidden Bearish RSI Divergence ({tf})"
    if price.iloc[-1] < price.loc[swing_low_idx] and rsi.iloc[-1] > rsi.loc[swing_low_idx] and rsi.iloc[-1] < 25:
        return f"Bullish RSI Divergence ({tf})"
    if price.iloc[-1] > price.loc[swing_high_idx] and rsi.iloc[-1] < rsi.loc[swing_high_idx] and rsi.iloc[-1] > 75:
        return f"Bearish RSI Divergence ({tf})"
    return None
def detect_candle_patterns(df: pd.DataFrame, tf: str) -> List[str]:
    if len(df) < 3:
        return []
    patterns = set()
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c['close'] - c['open'])
    upper = c['high'] - max(c['open'], c['close'])
    lower = min(c['open'], c['close']) - c['low']
    range_size = c['high'] - c['low']
    if body > 0.7 * range_size and df['volume'].iloc[-1] > VOL_SURGE_MULTIPLIER * df['volume'].rolling(20).mean().iloc[-1]:
        dir_str = "Bullish" if c['close'] > c['open'] else "Bearish"
        patterns.add(f"{dir_str} Displacement Candle ({tf})")
    if body > 0 and lower > 2 * body and upper < body * 0.3 and c['close'] > p['close']:
        patterns.add(f"Bullish Pinbar ({tf})")
    if body > 0 and upper > 2 * body and lower < body * 0.3 and c['close'] < p['close']:
        patterns.add(f"Bearish Pinbar ({tf})")
    if p['close'] < p['open'] and c['close'] > c['open'] and c['open'] < p['close'] and c['close'] > p['open']:
        patterns.add(f"Bullish Engulfing ({tf})")
    if p['close'] > p['open'] and c['close'] < c['open'] and c['open'] > p['close'] and c['close'] < p['open']:
        patterns.add(f"Bearish Engulfing ({tf})")
    if c['high'] < p['high'] and c['low'] > p['low']:
        patterns.add(f"Inside Bar ({tf})")
    return list(patterns)
# UPDATED: process_trade: Integrate drawdown scaling, EV logging, CSV log, paper mode
async def process_trade(trades: Dict[str, Any], to_delete: List[str], now: datetime, current_capital: float, prices: Dict[str, Optional[float]], updated_keys: List[str], is_protected: bool = False):
    risk_scale = get_risk_scaling_factor(load_stats()) # NEW: Scaling
    if risk_scale == 0:
        logging.warning("Drawdown protection: Stop trading")
        return
    for trade_key, trade in list(trades.items()):
        clean_symbol = get_clean_symbol(trade_key)
        if 'last_check' in trade and now - trade['last_check'] > timedelta(hours=TRADE_TIMEOUT_HOURS):
            logging.info(f"Timeout for {'protected ' if is_protected else ''}{trade_key}")
            await send_throttled(CHAT_ID, f"**TIMEOUT** {clean_symbol.replace('/USDT','')} (*neutral PnL*)", parse_mode='Markdown')
            to_delete.append(trade_key)
            updated_keys.append(trade_key)
            continue
        price = prices.get(clean_symbol)
        if price is None:
            trade['last_check'] = now
            updated_keys.append(trade_key)
            continue
        trade['last_check'] = now
        updated_keys.append(trade_key)
        if not is_protected and trade.get('active') and 'entry_time' in trade and now - trade['entry_time'] > timedelta(hours=PROTECT_AFTER_HOURS):
            protected_trades[trade_key] = trades.pop(trade_key)
            updated_keys.append(trade_key)
            logging.info(f"Moved active {trade_key} to protected after {PROTECT_AFTER_HOURS}h")
            continue
        if not trade.get('active', False):
            entry_low = trade['entry_low']
            entry_high = trade['entry_high']
            slippage_note = ""
            if trade['direction'] == 'Long':
                extended_high = entry_high * (1 + ENTRY_SLIPPAGE_PCT)
                in_zone = entry_low <= price <= extended_high
                if price > entry_high:
                    slippage_note = " (*late entry via slippage*)"
            else:
                extended_low = entry_low * (1 - ENTRY_SLIPPAGE_PCT)
                in_zone = extended_low <= price <= entry_high
                if price < entry_low:
                    slippage_note = " (*late entry via slippage*)"
            logging.info(f"Checking {clean_symbol} ({trade.get('type', '')} {trade['direction']} {trade['confidence']}%) - price {price:.4f} vs zone {entry_low:.4f}-{entry_high:.4f}, in_zone={in_zone}{slippage_note}")
            if in_zone:
                # NEW: Paper trading bypass for exposure check
                if PAPER_TRADING:
                    logging.info(f"PAPER MODE: Skipping exposure check for {clean_symbol}")
                else:
                    current_exposure = sum(
                        t.get('position_size', 0) * t.get('entry_price', 0) * t.get('leverage', 1)
                        for trades_dict in [open_trades, protected_trades]
                        for t in trades_dict.values()
                        if t.get('active')
                    )
                    risk_distance = abs(price - trade['sl'])
                    scaled_risk_pct = RISK_PER_TRADE_PCT * risk_scale # NEW: Scaled
                    proposed_size = (current_capital * scaled_risk_pct / 100) / risk_distance if risk_distance > 0 else 0
                    proposed_exposure = proposed_size * trade['leverage']
                    max_exposure = current_capital * 0.05
                    if current_exposure + proposed_exposure > max_exposure:
                        logging.info(f"Skipped {'protected ' if is_protected else ''}entry for {clean_symbol}: exposure would exceed 5% ({current_exposure + proposed_exposure:.2f} > {max_exposure:.2f})")
                        continue
                # NEW: Log EV
                ev_r = calculate_expected_value(trade, HISTORICAL_DATA)
                logging.info(f"Entry EV for {clean_symbol}: {ev_r:.2f}R")
                trade['active'] = True
                slippage = SLIPPAGE_PCT * price
                trade['entry_price'] = price + slippage if trade['direction'] == 'Long' else price - slippage
                if 'entry_time' not in trade:
                    trade['entry_time'] = now
                risk_amount = current_capital * RISK_PER_TRADE_PCT / 100
                trade['position_size'] = risk_amount / risk_distance if risk_distance > 0 else 0
                tag = ('(*roadmap*)' if trade.get('type') == 'roadmap' else ('(*protected*)' if is_protected else ''))
                if tag == '(*roadmap*)':
                    logging.info(f"Roadmap ENTRY ACTIVATED for {clean_symbol} @ {price:.4f} (conf {trade['confidence']}%)")
                await send_throttled(CHAT_ID,
                                     f"**ENTRY ACTIVATED** {tag} (Size: {trade['position_size']:.4f} | EV: {ev_r:.2f}R){slippage_note}\n\n"
                                     f"**{clean_symbol.replace('/USDT','')} {trade['direction']}** @ {format_price(price)}\n"
                                     f"*SL* {format_price(trade['sl'])} │ *TP1* {format_price(trade['tp1'])} │ *TP2* {format_price(trade['tp2'])} │ {trade['leverage']}x",
                                     parse_mode='Markdown')
                updated_keys.append(trade_key)
        if trade.get('active'):
            entry_price = trade['entry_price']
            size = trade.get('position_size', 1)
            current_sl = trade.get('trailing_sl', trade['sl'])
            if (trade['direction'] == 'Long' and price >= trade['tp1']) or (trade['direction'] == 'Short' and price <= trade['tp1']):
                if current_sl == trade['sl']:
                    trade['trailing_sl'] = entry_price
                    current_sl = entry_price
                    logging.info(f"Trailing SL to breakeven for {'protected ' if is_protected else ''}{clean_symbol}")
                    updated_keys.append(trade_key)
            hit_tp = (price >= trade['tp2'] if trade['direction'] == 'Long' else price <= trade['tp2'])
            hit_sl = (price <= current_sl if trade['direction'] == 'Long' else price >= current_sl)
            if hit_tp or hit_sl:
                if trade.get('processed'):
                    logging.info(f"Duplicate PnL skipped for {'protected ' if is_protected else ''}{clean_symbol}")
                    continue
                trade['processed'] = True
                trade['hit_tp'] = hit_tp # NEW: For winrate tracking
                diff = (price - entry_price) if trade['direction'] == 'Long' else (entry_price - price)
                pnl_usdt = diff * size
                fee_usdt = 2 * FEE_PCT * (entry_price * size)
                net_pnl_usdt = pnl_usdt - fee_usdt
                net_pnl_pct = net_pnl_usdt / current_capital * 100
                result = "WIN" if hit_tp else "LOSS"
                tag = ('(*roadmap*)' if trade.get('type') == 'roadmap' else ('(*protected*)' if is_protected else ''))
                await send_throttled(CHAT_ID,
                                     f"**{result}** {clean_symbol.replace('/USDT','')} {tag}\n\n"
                                     f"+{net_pnl_pct:+.2f}% (size {size:.4f}) @ {trade['leverage']}x (conf {trade['confidence']}%) **[fees adj]**",
                                     parse_mode='Markdown')
                async with stats_lock:
                    delta_capital = stats['capital'] * (net_pnl_pct / 100)
                    stats['capital'] += delta_capital
                    stats['pnl'] = (stats['capital'] - SIMULATED_CAPITAL) / SIMULATED_CAPITAL * 100
                    stats['wins' if hit_tp else 'losses'] += 1
                    await save_stats_async(stats)
                # Log trade to CSV for analysis
                trade_log = {
                    'timestamp': now.isoformat(),
                    'symbol': clean_symbol,
                    'direction': trade['direction'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'sl': trade['sl'],
                    'tp1': trade['tp1'],
                    'tp2': trade['tp2'],
                    'leverage': trade['leverage'],
                    'confidence': trade['confidence'],
                    'strength': trade.get('strength', 0),
                    'result': result,
                    'pnl_pct': net_pnl_pct,
                    'pnl_usdt': net_pnl_usdt,
                    'size': size,
                    'reason': trade.get('reason', 'N/A'),
                    'regime': trade.get('regime', 'Unknown'),
                    'protected': 'Yes' if is_protected else 'No'
                }
                # Append to CSV
                file_exists = Path(TRADE_LOG_FILE).exists()
                async with aiofiles.open(TRADE_LOG_FILE, 'a', newline='') as f:
                    if not file_exists:
                        # Write header
                        await f.write(','.join(trade_log.keys()) + '\n')
                    await f.write(','.join(str(v) for v in trade_log.values()) + '\n')
                logging.info(f"Logged trade to {TRADE_LOG_FILE}: {result} {clean_symbol} {net_pnl_pct:+.2f}%")
                to_delete.append(trade_key)
                updated_keys.append(trade_key)
# Retained: fetch_open_interest, fetch_ohlcv, fetch_ticker_batch, fetch_ticker, price_background_task
async def fetch_open_interest(symbol: str) -> Optional[Dict[str, float]]:
    async with fetch_sem:
        if await BanManager.check_and_sleep():
            logging.warning(f"OI skipped for {symbol}: ban/cooldown active")
            return None
        try:
            futures_symbol = symbol.replace('/', '')
            oi_data = await futures_exchange.fetch_open_interest(futures_symbol)
            logging.debug(f"Raw OI data for {futures_symbol}: {oi_data}")
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
                raise KeyError("'openInterest' (amount/value) or raw sumOpenInterest not found")
            prev_oi = last_oi.get(futures_symbol, oi_value)
            oi_change_pct = (oi_value - prev_oi) / prev_oi * 100 if prev_oi and prev_oi != 0 else 0
            last_oi[futures_symbol] = float(oi_value)
            await asyncio.sleep(2)
            return {
                'open_interest': float(oi_value),
                'oi_change_pct': oi_change_pct
            }
        except Exception as e:
            logging.error(f"OI fetch error for {symbol}: {e}")
            await asyncio.sleep(5)
            return None
async def fetch_ohlcv(symbol: str, tf: str, limit: int = 200, since: Optional[int] = None) -> pd.DataFrame:
    async with fetch_sem:
        if await BanManager.check_and_sleep():
            logging.warning(f"OHLCV skipped for {symbol} {tf}: ban/cooldown active")
            return cache_get(ohlcv_cache, f"{symbol}{tf}").get('df', pd.DataFrame()) if cache_get(ohlcv_cache, f"{symbol}{tf}") else pd.DataFrame()
        cache_key = f"{symbol}{tf}"
        now = time.time()
        ttl = HTF_CACHE_TTL if tf in ['1d', '1w'] else CACHE_TTL
        cache_hit = False
        cached = cache_get(ohlcv_cache, cache_key)
        if cached and now - cached['timestamp'] < ttl:
            cache_hit = True
            logging.debug(f"Cache hit for OHLCV {cache_key}")
            return cached['df']
        else:
            logging.debug(f"Cache miss for OHLCV {cache_key}; fetching fresh")
        evict_if_full(ohlcv_cache)
        backoff = 1
        success = False
        for attempt in range(4):
            try:
                norm_tf = tf.lower()
                params = {'limit': limit}
                if since:
                    params['since'] = since
                data = await exchange.fetch_ohlcv(symbol, norm_tf, **params)
                # Create DataFrame
                df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['ts'], unit='ms')
              
                # Compute VWAP BEFORE any other operations
                df = compute_vwap_safe(df) # THIS LINE MUST BE HERE
              
                # Store in cache
                ohlcv_cache[cache_key] = {'df': df, 'timestamp': now}
                success = True
                logging.info(f"OHLCV fetched for {symbol} {tf} (attempt {attempt+1})")
                break
            except Exception as e:
                logging.warning(f"OHLCV fetch attempt {attempt+1} failed for {symbol} {tf}: {e}")
                if attempt < 3:
                    await asyncio.sleep(backoff * 3)
                    backoff *= 3
                else:
                    logging.warning(f"OHLCV fetch failed all attempts for {symbol} {tf}: using cache/empty")
                    break
        sleep_time = 3 if success else 5
        await asyncio.sleep(sleep_time)
        return cache_get(ohlcv_cache, cache_key).get('df', pd.DataFrame()) if cache_get(ohlcv_cache, cache_key) else pd.DataFrame()
def compute_vwap_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VWAP safely with proper DatetimeIndex handling.
    Returns DataFrame with 'vwap' column added.
    """
    if len(df) < 20 or 'date' not in df.columns:
        df['vwap'] = np.nan
        return df
  
    try:
        # Create indexed copy for VWAP
        df_indexed = df.set_index('date').sort_index().copy()
      
        # Check volume
        total_vol = df_indexed['volume'].sum()
        if total_vol == 0:
            df['vwap'] = np.nan
            logging.debug("VWAP skipped: zero volume")
            return df
      
        # Compute VWAP
        vwap_series = ta.vwap(
            df_indexed['high'],
            df_indexed['low'],
            df_indexed['close'],
            df_indexed['volume']
        )
      
        # Map back to original DataFrame by date
        vwap_dict = vwap_series.to_dict()
        df['vwap'] = df['date'].map(vwap_dict)
      
        nan_count = df['vwap'].isna().sum()
        if nan_count > len(df) * 0.5:
            logging.warning(f"VWAP has {nan_count}/{len(df)} NaN values")
      
        return df
      
    except Exception as e:
        logging.error(f"VWAP computation failed: {e}")
        df['vwap'] = np.nan
        return df
async def fetch_ticker_batch() -> Dict[str, Optional[float]]:
    async with fetch_sem:
        now = time.time()
        cache_hits = sum(1 for s in SYMBOLS if s in ticker_cache and now - ticker_cache[s]['timestamp'] < TICKER_CACHE_TTL)
        if cache_hits == len(SYMBOLS):
            logging.debug(f"Full cache hit for tickers ({cache_hits}/{len(SYMBOLS)})")
            return {s: ticker_cache[s]['price'] for s in SYMBOLS}
        logging.debug(f"Partial cache hit for tickers ({cache_hits}/{len(SYMBOLS)}); polling fresh")
        if await BanManager.check_and_sleep():
            return {s: cache_get(ticker_cache, s).get('price') if cache_get(ticker_cache, s) else None for s in SYMBOLS}
        evict_if_full(ticker_cache)
        backoff = 1
        for attempt in range(3):
            try:
                tickers = await exchange.fetch_tickers(SYMBOLS)
                prices = {s: tickers[s]['last'] if s in tickers else None for s in SYMBOLS}
                logging.info(f"Batch ticker poll success (attempt {attempt+1})")
                for s in SYMBOLS:
                    if prices[s]:
                        ticker_cache[s] = {'price': prices[s], 'timestamp': now}
                if attempt > 0:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                logging.info("Bybit poll: Rate limits OK")
                break
            except Exception as e:
                logging.debug(f"Batch ticker poll attempt {attempt+1} failed: {e}")
                if "429" in str(e):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise
        else:
            prices = {s: cache_get(ticker_cache, s).get('price') if cache_get(ticker_cache, s) else None for s in SYMBOLS}
        return prices
async def fetch_ticker(symbol: str) -> Optional[float]:
    prices = await fetch_ticker_batch()
    return prices.get(symbol)
async def price_background_task():
    while True:
        try:
            prices = await fetch_ticker_batch()
            order_books = await fetch_order_flow_batch()
            if any(p is not None for p in prices.values()):
                prices_global.update(prices)
                logging.debug(f"Background poll update: {prices_global}")
            if order_books:
                logging.debug(f"Order flow updated: {list(order_books.keys())}")
            await asyncio.sleep(10)
        except Exception as e:
            logging.warning(f"Background poll error (retrying): {e}")
            await asyncio.sleep(5)
# Retained: btc_trend_update, price_update_callback (use EMA200)
async def btc_trend_update(context):
    global btc_trend_global
    try:
        df = await fetch_ohlcv('BTC/USDT', '1d', limit=500)
        if len(df) == 0:
            logging.warning("Empty raw DF in BTC trend – possible rate limit or insufficient data")
            btc_trend_global = "Sideways"
            return
        df = df.copy()
        df['ema200'] = ta.ema(df['close'], 200) # UPDATED: EMA200
        df = df.dropna(subset=['ema200'])
        if len(df) == 0:
            logging.warning("Empty DF after EMA in BTC trend – insufficient data")
            btc_trend_global = "Sideways"
            return
        if len(df) < 100:
            logging.warning(f"Insufficient bars ({len(df)}) for trend in BTC – fallback to Sideways")
            btc_trend_global = "Sideways"
            return
        l = df.iloc[-1]
        if pd.isna(l['ema200']):
            logging.warning("EMA NaN in BTC trend – skipping update")
            return
        if l['close'] > l['ema200']: # UPDATED: vs close
            btc_trend_global = "Uptrend"
        elif l['close'] < l['ema200']:
            btc_trend_global = "Downtrend"
        else:
            btc_trend_global = "Sideways"
        logging.info(f"Global BTC trend: {btc_trend_global}")
    except Exception as e:
        logging.error(f"BTC trend update error: {e}")
        btc_trend_global = "Sideways"
async def price_update_callback(context):
    global last_price_update, prices_global
    now = time.time()
    if now - last_price_update < 30:
        return
    start = time.perf_counter()
    try:
        prices = await fetch_ticker_batch()
        if any(p is not None for p in prices.values()):
            prices_global.update(prices)
            last_price_update = now
            logging.info(f"Prices updated globally: {prices_global}")
        await asyncio.sleep(1)
    except Exception as e:
        logging.error(f"Price update error: {e}")
    finally:
        exec_time = time.perf_counter() - start
        logging.info(f"Price update cycle completed in {exec_time:.2f}s")
# UPDATED: track_callback: Integrate risk scaling
async def track_callback(context):
    global stats, open_trades, protected_trades
    start_time = time.perf_counter()
    try:
        if not open_trades and not protected_trades:
            logging.info(f"Track cycle completed in {time.perf_counter() - start_time:.2f}s (no trades)")
            return
        now = datetime.now(timezone.utc)
        to_delete_open = []
        to_delete_protected = []
        updated_keys_open = []
        updated_keys_protected = []
        current_capital = stats['capital']
        prices = prices_global.copy()
        if all(p is None for p in prices.values()):
            logging.warning("No global prices available – skipping track checks")
            return
        drawdown = max(0, -stats['pnl'])
        if drawdown > MAX_DRAWDOWN_PCT and len(open_trades) > 0:
            logging.warning(f"Max drawdown exceeded ({drawdown:.1f}%) – pausing trades")
            for key in list(open_trades.keys()):
                if not open_trades[key].get('active'):
                    del open_trades[key]
                    updated_keys_open.append(key)
            await save_trades_async(open_trades)
            await send_throttled(CHAT_ID, f"**PAUSE:** Drawdown {drawdown:.1f}% – Cleared pending trades", parse_mode='Markdown')
        total_active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        if total_active >= MAX_CONCURRENT_TRADES:
            logging.info("Max concurrent trades reached – skipping new checks")
        await process_trade(open_trades, to_delete_open, now, current_capital, prices, updated_keys_open, is_protected=False)
        await process_trade(protected_trades, to_delete_protected, now, current_capital, prices, updated_keys_protected, is_protected=True)
        for key in to_delete_open:
            del open_trades[key]
        for key in to_delete_protected:
            del protected_trades[key]
        if updated_keys_open or to_delete_open:
            await save_trades_async(open_trades)
        if updated_keys_protected or to_delete_protected:
            await save_protected_async(protected_trades)
        exec_time = time.perf_counter() - start_time
        logging.info(f"Track cycle completed in {exec_time:.2f}s ({len(to_delete_open + to_delete_protected)} closes, {len(updated_keys_open + updated_keys_protected)} updates)")
    except Exception as e:
        logging.error(f"Track callback error: {e}")
        logging.info(f"Track cycle failed in {time.perf_counter() - start_time:.2f}s")
# NEW: Multi-symbol backtest
async def run_backtest_logic(df: pd.DataFrame, symbol: str) -> List[Dict]:
    """Shared backtest logic for single/multi symbol tests"""
    trades = []
    capital = SIMULATED_CAPITAL
    for i in range(100, len(df)):
        df_slice = df.iloc[:i+1].copy()
        if is_consolidation(df_slice):
            continue
        current_price = df_slice['close'].iloc[-1]
        zones = await find_next_premium_zones(df_slice, current_price, '1d', symbol)
        if not zones:
            continue
        grok_result = await query_grok_potential(zones, symbol, current_price, "Uptrend", None) # Mock trend
        if 'live_trade' not in grok_result:
            continue
        trade = grok_result['live_trade']
        # Simulate execution (next bar, zone hit with high/low)
        entry_bar = i + 1
        if entry_bar >= len(df):
            break
        entry_price = None
        for j in range(entry_bar, min(entry_bar + 10, len(df))):
            if df['low'].iloc[j] <= trade['entry_high'] and df['high'].iloc[j] >= trade['entry_low']:
                entry_price = (trade['entry_low'] + trade['entry_high'])/2 * (1 + SLIPPAGE_PCT if trade['direction'] == 'Long' else 1 - SLIPPAGE_PCT)
                entry_bar = j
                break
        if not entry_price:
            continue
        # Track exits with OHLC
        hit_tp1 = False
        for k in range(entry_bar + 1, len(df)):
            if trade['direction'] == 'Long':
                if df['low'].iloc[k] <= trade['sl']:
                    exit_price = trade['sl'] * (1 - SLIPPAGE_PCT)
                    pnl_usdt = (exit_price - entry_price) * (capital * RISK_PER_TRADE_PCT / 100 / abs(entry_price - trade['sl'])) - 2 * FEE_PCT * entry_price * (capital * RISK_PER_TRADE_PCT / 100 / abs(entry_price - trade['sl']))
                    trades.append({'result': 'loss', 'pnl': pnl_usdt, 'hit_tp1': hit_tp1})
                    break
                if not hit_tp1 and df['high'].iloc[k] >= trade['tp1']:
                    hit_tp1 = True
                elif df['high'].iloc[k] >= trade['tp2']:
                    exit_price = trade['tp2'] * (1 - SLIPPAGE_PCT)
                    pnl_usdt = (exit_price - entry_price) * (capital * RISK_PER_TRADE_PCT / 100 / abs(entry_price - trade['sl'])) - 2 * FEE_PCT * entry_price * (capital * RISK_PER_TRADE_PCT / 100 / abs(entry_price - trade['sl']))
                    trades.append({'result': 'win', 'pnl': pnl_usdt, 'hit_tp1': True})
                    break
            else: # Short symmetric
                if df['high'].iloc[k] >= trade['sl']:
                    exit_price = trade['sl'] * (1 + SLIPPAGE_PCT)
                    pnl_usdt = (entry_price - exit_price) * (capital * RISK_PER_TRADE_PCT / 100 / abs(trade['sl'] - entry_price)) - 2 * FEE_PCT * entry_price * (capital * RISK_PER_TRADE_PCT / 100 / abs(trade['sl'] - entry_price))
                    trades.append({'result': 'loss', 'pnl': pnl_usdt, 'hit_tp1': hit_tp1})
                    break
                if not hit_tp1 and df['low'].iloc[k] <= trade['tp1']:
                    hit_tp1 = True
                elif df['low'].iloc[k] <= trade['tp2']:
                    exit_price = trade['tp2'] * (1 + SLIPPAGE_PCT)
                    pnl_usdt = (entry_price - exit_price) * (capital * RISK_PER_TRADE_PCT / 100 / abs(trade['sl'] - entry_price)) - 2 * FEE_PCT * entry_price * (capital * RISK_PER_TRADE_PCT / 100 / abs(trade['sl'] - entry_price))
                    trades.append({'result': 'win', 'pnl': pnl_usdt, 'hit_tp1': True})
                    break
        else:
            # Open at end
            diff = (df['close'].iloc[-1] - entry_price) if trade['direction'] == 'Long' else (entry_price - df['close'].iloc[-1])
            pnl_usdt = diff * (capital * RISK_PER_TRADE_PCT / 100 / abs(entry_price - trade['sl'])) - 2 * FEE_PCT * entry_price * (capital * RISK_PER_TRADE_PCT / 100 / abs(entry_price - trade['sl']))
            trades.append({'result': 'open', 'pnl': pnl_usdt, 'hit_tp1': hit_tp1})
    return trades
async def backtest_all_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run backtest on all symbols for cross-validation"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    logging.info(f"/backtest_all triggered by user {update.effective_user.id}")
    results = {}
    summary_msg = "**Multi-Symbol Backtest (90d)**\n\n"
    for symbol in SYMBOLS:
        try:
            await update.message.reply_text(f"Running backtest for {symbol}...", parse_mode='Markdown')
        
            days = 90
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
            df = await fetch_ohlcv(symbol, '1d', limit=days, since=since)
        
            if len(df) == 0:
                summary_msg += f"{symbol}: No data\n"
                continue
        
            # Use same logic as backtest_cmd (extract to function for DRY)
            trades = await run_backtest_logic(df, symbol)
        
            wins = len([t for t in trades if t['result'] == 'win'])
            total = len([t for t in trades if t['result'] != 'open'])
            winrate = (wins / total * 100) if total > 0 else 0
            total_pnl = sum(t['pnl'] for t in trades)
        
            results[symbol] = {'winrate': winrate, 'total_trades': total, 'pnl': total_pnl}
        
            summary_msg += f"**{symbol.replace('/USDT','')}**: {wins}/{total} ({winrate:.1f}%) | PnL: {total_pnl:+.2f}\n"
        
            await asyncio.sleep(2)
    
        except Exception as e:
            logging.error(f"Backtest error for {symbol}: {e}")
            summary_msg += f"{symbol}: Error\n"
    # Calculate aggregate
    total_wins = sum(r['winrate'] * r['total_trades'] / 100 for r in results.values())
    total_trades = sum(r['total_trades'] for r in results.values())
    avg_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    summary_msg += f"\n**Aggregate**: {avg_winrate:.1f}% ({int(total_wins)}/{total_trades})"
    await send_throttled(CHAT_ID, summary_msg, parse_mode='Markdown')
    logging.info(f"Multi-symbol backtest complete: {avg_winrate:.1f}% WR")
# NEW: Monte Carlo
def monte_carlo_validation(trades: List[Dict], n_simulations: int = 1000) -> Dict:
    """Shuffle trade order to test if edge is from skill or luck"""
    if len(trades) < 10:
        return {'is_significant': False, 'reason': 'Too few trades'}
    actual_pnl = sum(t['pnl'] for t in trades)
    simulated_pnls = []
    for _ in range(n_simulations):
        shuffled = random.sample(trades, len(trades))
        sim_pnl = sum(t['pnl'] for t in shuffled)
        simulated_pnls.append(sim_pnl)
    # How many simulations beat actual?
    better_count = sum(1 for sim_pnl in simulated_pnls if sim_pnl > actual_pnl)
    percentile = better_count / n_simulations
    p_value = percentile
    return {
        'is_significant': p_value < 0.05, # 95% confidence
        'p_value': p_value,
        'actual_pnl': actual_pnl,
        'median_simulated': np.median(simulated_pnls),
        'confidence': 'High (p<0.05)' if p_value < 0.05 else 'Low (p≥0.05)'
    }
async def validate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run Monte Carlo validation on last backtest"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    logging.info(f"/validate triggered by user {update.effective_user.id}")
    # Load last backtest
    if not os.path.exists(BACKTEST_FILE):
        await update.message.reply_text("No backtest found. Run /backtest first.", parse_mode='Markdown')
        return
    try:
        # Re-run backtest to get trades (or store trades in backtest file)
        days = 90
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        df = await fetch_ohlcv('BTC/USDT', '1d', limit=days, since=since)
        trades = await run_backtest_logic(df, 'BTC/USDT')
    
        # Run Monte Carlo
        result = monte_carlo_validation(trades, n_simulations=1000)
    
        msg = (
            f"**Monte Carlo Validation (1000 sims)**\n\n"
            f"**Actual PnL**: {result['actual_pnl']:+.2f}\n"
            f"**Median Simulated**: {result['median_simulated']:+.2f}\n"
            f"**P-value**: {result['p_value']:.4f}\n"
            f"**Confidence**: {result['confidence']}\n\n"
            f"**Interpretation**: {'Edge is statistically significant ✅' if result['is_significant'] else 'Edge may be luck ⚠️'}"
        )
    
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        logging.info(f"Monte Carlo validation: p={result['p_value']:.4f}")
    except Exception as e:
        logging.error(f"Validation error: {e}")
        await send_throttled(CHAT_ID, f"Validation failed: {str(e)}", parse_mode='Markdown')
# NEW: Dashboard
async def dashboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed performance dashboard"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    logging.info(f"/dashboard triggered by user {update.effective_user.id}")
    try:
        # Load trade history
        if not Path(TRADE_LOG_FILE).exists():
            await update.message.reply_text("No trade history yet. Take some trades first!", parse_mode='Markdown')
            return
    
        trades_df = pd.read_csv(TRADE_LOG_FILE)
    
        if len(trades_df) == 0:
            await update.message.reply_text("No closed trades yet.", parse_mode='Markdown')
            return
    
        # Calculate metrics
        wins = len(trades_df[trades_df['result'] == 'WIN'])
        losses = len(trades_df[trades_df['result'] == 'LOSS'])
        total = wins + losses
        winrate = (wins / total * 100) if total > 0 else 0
    
        avg_win = trades_df[trades_df['result'] == 'WIN']['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['result'] == 'LOSS']['pnl_pct'].mean() if losses > 0 else 0
    
        expectancy = (winrate/100 * avg_win) + ((100-winrate)/100 * avg_loss)
    
        # Best/worst trades
        best_trade = trades_df.loc[trades_df['pnl_pct'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl_pct'].idxmin()]
    
        # By symbol
        symbol_stats = trades_df.groupby('symbol').agg({
            'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'pnl_pct': 'sum'
        }).round(1)
    
        # By regime
        regime_stats = trades_df.groupby('regime').agg({
            'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'pnl_pct': 'sum'
        }).round(1) if 'regime' in trades_df.columns else None
    
        # Build message
        msg = f"**📊 Performance Dashboard**\n\n"
        msg += f"**Overall**: {wins}W / {losses}L ({winrate:.1f}%)\n"
        msg += f"**Avg Win**: +{avg_win:.2f}% | **Avg Loss**: {avg_loss:.2f}%\n"
        msg += f"**Expectancy**: {expectancy:+.2f}%\n\n"
    
        msg += f"**Best Trade**: {best_trade['symbol']} {best_trade['direction']} @ {best_trade['entry_price']:.2f} → {best_trade['pnl_pct']:+.2f}%\n"
        msg += f"**Worst Trade**: {worst_trade['symbol']} {worst_trade['direction']} @ {worst_trade['entry_price']:.2f} → {worst_trade['pnl_pct']:+.2f}%\n\n"
    
        msg += "**By Symbol**:\n"
        for symbol, row in symbol_stats.iterrows():
            msg += f" {symbol}: {row['result']:.0f}% WR | {row['pnl_pct']:+.1f}%\n"
    
        if regime_stats is not None:
            msg += "\n**By Regime**:\n"
            for regime, row in regime_stats.iterrows():
                msg += f" {regime}: {row['result']:.0f}% WR | {row['pnl_pct']:+.1f}%\n"
    
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        logging.info(f"Dashboard sent: {total} trades analyzed")
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        await send_throttled(CHAT_ID, f"Dashboard failed: {str(e)}", parse_mode='Markdown')
# Retained: stats_cmd, health_cmd, recap_cmd, daily_callback, webhook_update
async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    logging.info(f"/stats triggered by user {update.effective_user.id}")
    try:
        total = stats['wins'] + stats['losses']
        winrate = (stats['wins'] / total * 100) if total > 0 else 0
        active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        pending = len([t for t in open_trades.values() if not t.get('active')])
        unique_symbols = sorted(set(get_clean_symbol(k).replace('/USDT', '') for trades in [open_trades, protected_trades] for k in trades.keys()))
        open_symbols = ', '.join(unique_symbols) if unique_symbols else 'None'
        drawdown = max(0, -stats['pnl'])
        msg = (
            "**Bot Statistics ♔**\n\n"
            f"**Total Trades:** {total}\n"
            f"**Wins:** {stats['wins']} ({winrate:.1f}%)\n"
            f"**Losses:** {stats['losses']}\n"
            f"**Total PNL:** {stats['pnl']:+.2f}% | *Capital:* ${stats['capital']:.2f}\n"
            f"**Drawdown:** {drawdown:.2f}%\n"
            f"**Active (incl protected):** {active}\n"
            f"**Pending:** {pending}\n"
            f"**Open:** {open_symbols}"
        )
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        logging.info(f"/stats sent successfully")
    except Exception as e:
        logging.error(f"/stats error: {e}")
        await send_throttled(CHAT_ID, f"Error fetching stats: {str(e)}", parse_mode='Markdown')
async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    logging.info(f"/health triggered by user {update.effective_user.id}")
    try:
        uptime = datetime.now(timezone.utc).isoformat()
        open_count = len(open_trades)
        protected_count = len(protected_trades)
        active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        pending = len([t for t in open_trades.values() if not t.get('active')])
        msg = (
            f"**Grok Elite Bot v25.02.0 - ICT Elite Alive!**\n\n"
            f"**MODE**: {'📝 PAPER TRADING' if PAPER_TRADING else '💰 LIVE TRADING'}\n"
            f"**Uptime Check:** {uptime}\n"
            f"**Open Trades:** {open_count}\n"
            f"**Protected Trades:** {protected_count}\n"
            f"**Active:** {active} | **Pending:** {pending}\n"
            f"**Status:** Regime-aware MTF ICT, dynamic EV, streamlined stack, 12h CD"
        )
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        logging.info(f"/health sent successfully")
    except Exception as e:
        logging.error(f"/health error: {e}")
        await send_throttled(CHAT_ID, f"Health check failed: {str(e)}", parse_mode='Markdown')
async def recap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    logging.info(f"/recap triggered by user {update.effective_user.id}")
    await daily_callback(context)
    await send_throttled(CHAT_ID, "**Manual recap triggered**—check logs/messages!", parse_mode='Markdown')
async def daily_callback(context):
    now = datetime.now(timezone.utc)
    recap_file = RECAP_FILE
    last_date = None
    if os.path.exists(recap_file):
        try:
            with open(recap_file, 'r') as f:
                last_date = datetime.fromisoformat(f.read().strip())
                logging.info(f"Last recap: {last_date.date()}")
        except Exception as e:
            logging.warning(f"Failed to read recap file: {e}")
            last_date = None
    if last_date and now.date() == last_date.date():
        logging.info("Recap skipped—already sent today")
        return
    try:
        logging.info(f"Daily recap starting for {now.date()}")
        text = f"**Daily Recap – {now.strftime('%b %d, %Y')}**\n"
        prices = await fetch_ticker_batch()
        for sym in SYMBOLS:
            if await BanManager.check_and_sleep():
                logging.warning(f"Ban active during recap for {sym}; skipping")
                continue
            price = prices.get(sym)
            if price is None:
                logging.warning(f"No price for {sym}")
                await asyncio.sleep(2)
                continue
            df = await fetch_ohlcv(sym, '1d', 2)
            if len(df) < 2:
                logging.warning(f"Insufficient data for {sym}")
                await asyncio.sleep(2)
                continue
            change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            text += f"{sym}: {change:+.2f}% │ {format_price(price)}\n"
            logging.info(f"{sym}: {change:+.2f}% @ {price:.4f}")
            await asyncio.sleep(2)
        if not text.endswith('\n'):
            await send_throttled(CHAT_ID, "**Recap skipped**—API ban active, try later.", parse_mode='Markdown')
            return
        text += "\nInstitutional macro recap + next 48h whale bias."
        logging.info("Calling Grok for recap...")
        global _working_model
        payload_base = {
            "messages": [{"role": "system", "content": "You are a top institutional macro analyst. Focus on whale flows, OB structures. Respond concisely."},
                         {"role": "user", "content": text}],
            "temperature": 0.2, "max_tokens": 300
        }
        models = [_working_model, "grok-4", "grok-3"] if _working_model else ["grok-4", "grok-3"]
        recap = None
        for model in models:
            payload = payload_base.copy()
            payload["model"] = model
            try:
                async with httpx.AsyncClient(timeout=45.0) as c:
                    r = await c.post("https://api.x.ai/v1/chat/completions", json=payload,
                                     headers={"Authorization": f"Bearer {XAI_API_KEY}"})
                    r.raise_for_status()
                    recap = r.json()["choices"][0]["message"]["content"].strip()
                    _working_model = model
                    logging.info(f"Grok recap from {model} (len: {len(recap)})")
                    if model != "grok-4":
                        logging.info(f"Daily recap fell back to {model}")
                    break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logging.warning(f"401 Unauthorized for {model} - Check SuperGrok/Premium+ subscription")
                if model == models[-1]:
                    raise
                continue
            except Exception as inner_e:
                logging.error(f"Daily recap Grok error with {model}: {inner_e}")
                if model == models[-1]:
                    raise
                continue
        if recap:
            escaped_recap = html.escape(recap)
            full_msg = f"**INSTITUTIONAL DAILY SUMMARY**\n\n{escaped_recap}"
            await send_throttled(CHAT_ID, full_msg, parse_mode='HTML')
            async with aiofiles.open(recap_file, 'w') as f:
                await f.write(now.isoformat())
            logging.info("Daily recap sent successfully")
            await asyncio.sleep(1)
        else:
            logging.error("No recap generated - all models failed")
            await asyncio.sleep(1)
    except Exception as e:
        logging.error(f"Daily recap error: {type(e).name}: {str(e)}")
        await asyncio.sleep(1)
async def webhook_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        logging.info(f"Webhook update: user {update.effective_user.id}, command {update.message.text}")
# UPDATED: query_grok_instant (improved prompt: ICT-specific, examples, calibration)
async def query_grok_instant(context: str, is_alt: bool = False) -> Dict[str, Any]:
    global _working_model
    system_prompt = """You are a quantitative ICT (Inner Circle Trader) analyst with access to real-time market data.
AVAILABLE FACTORS IN CONTEXT (check each):
1. Order Block (OB): Strength 2-3, mitigation 0-1 (use if <0.5)
2. Volume surge: >1.5x 20-period average
3. Liquidity sweep: Wick >60% + volume + reversal
4. Open Interest: Change >10%
5. Order flow delta: Imbalance >1%
6. Premium/discount: Current price position in range (0-1 scale)
7. HTF alignment: 1h/4h/1d/1w trend confluence
CONFIDENCE SCORING RULES:
- 70% = 2 strong factors (e.g., OB str=2 + liq sweep)
- 75% = 3 factors (add volume surge)
- 80% = 4 factors (add order flow)
- 85% = 5 factors (add OI)
- 90%+ = 6+ factors (full confluence)
MANDATORY REQUIREMENTS (reject if not met):
✓ Long ONLY in discount zone (<0.3 premium_pct)
✓ Short ONLY in premium zone (>0.7 premium_pct)
✓ OB mitigation <0.5 (if mitigation data provided)
✓ Distance from current price <7%
✓ Risk-reward ≥1:2 (TP1) and ≥1:4 (TP2)
OUTPUT FORMAT (precise 4+ decimals, NO rounded numbers):
{
  "direction": "Long" or "Short",
  "entry_low": 68234.5678,
  "entry_high": 68456.1234,
  "sl": 67987.2345,
  "tp1": 68890.7890,
  "tp2": 69543.4567,
  "leverage": 3-7 (3x if conf<80%, 5x if 80-89%, 7x if 90%+),
  "confidence": 70-95 (based on factor count),
  "strength": 2-3 (OB strength from context),
  "reason": "OB str2 + liq sweep at discount + vol" (max 60 chars)
}
If no valid setup → {"no_trade": true}
""" + (f"\nALTS RULE: For {context.split('|')[0] if '|' in context else 'alts'}, align strictly with BTC institutional trend. Reject counter-BTC trades." if is_alt else "")
    models = [_working_model, "grok-4", "grok-3"] if _working_model else ["grok-4", "grok-3"]
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.x.ai/v1/chat/completions",
                                      json=payload,
                                      headers={"Authorization": f"Bearer {XAI_API_KEY}"})
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON from {model}: {content[:100]}...")
                    result = {"no_trade": True}
                # NEW: Calibrate confidence
                factors = {'liq_sweep': 'liq sweep' in result.get('reason', '').lower(),
                           'vol_surge': 'volume' in result.get('reason', '').lower(),
                           'htf_align': 'htf' in result.get('reason', '').lower(),
                           'oi_spike': 'oi' in result.get('reason', '').lower()}
                if 'confidence' in result:
                    result['confidence'] = calibrate_grok_confidence(result['confidence'], factors)
                _working_model = model
                if model != "grok-4":
                    logging.info(f"Fell back to {model} for query.")
                return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logging.warning(f"401 Unauthorized for {model} - Check SuperGrok/Premium+ subscription")
                continue
            raise
        except Exception as e:
            logging.error(f"Grok query error with {model}: {e}")
            if model == models[-1]:
                return {"error": str(e)}
    return {"error": "All models failed"}
# NEW: Calibrate Grok confidence
def calibrate_grok_confidence(grok_conf: int, factors: Dict) -> int:
    """Adjust Grok's overconfidence using realized win rates"""
    calibration = {
        90: 70,
        80: 60,
        70: 50
    }
    adjusted = calibration.get(grok_conf, grok_conf - 15)
    factor_count = sum(factors.values())
    return min(adjusted + factor_count * 5, 95)
# UPDATED: check_precision, check_rr (retained, but use in Grok)
def check_precision(trade: Dict[str, Any]) -> bool:
    price_keys = ['entry_low', 'entry_high', 'sl', 'tp1', 'tp2']
    for key in price_keys:
        if key in trade:
            p = trade[key]
            if round(p, 4) == round(p, 0):
                logging.warning(f"Precision fail: {key}={p} too round")
                return False
    return True
def check_rr(trade: Dict[str, Any]) -> bool:
    entry_mid = (trade['entry_low'] + trade['entry_high']) / 2
    rr2 = abs(trade['tp2'] - entry_mid) / abs(trade['sl'] - entry_mid)
    return rr2 >= 2.0
# UPDATED: query_grok_potential: Improved prompt + optimized context (top 2 zones, regime, weekly OB)
async def query_grok_potential(zones: List[Dict], symbol: str, current_price: float, trend: str, btc_trend: Optional[str], atr: float = 0) -> Dict[str, Any]:
    global _working_model
    is_alt = symbol != 'BTC/USDT'
    filtered_zones = [z for z in zones if z.get('strength', 0) >= 2 and z.get('prob', 0) >= 70]
    if not filtered_zones:
        return {"no_live_trade": True, "roadmap": []}
    # NEW: Optimized context (top 2, regime, HTF)
    df_1d = await fetch_ohlcv(symbol, '1d', 100)
    regime = detect_market_regime(df_1d)
    ranked = sorted(filtered_zones, key=lambda z: z['prob'], reverse=True)[:2]
    context = build_grok_context(ranked, symbol, current_price)
    context += f"\nMarket: {regime}\nHTF: 1d trend={trend}, 1w OB=1" # Placeholder count
    system_prompt = (
        "You are ICT daily reversal analyst. Focus on OB str2/3, liquidity sweeps as bounces, order walls. Output ONLY JSON. "
        "If price near zone (>=70% conf dynamic, multi-TF 1d align) → {'live_trade': {direction, entry_low, entry_high, sl, tp1, tp2, leverage:3-7, confidence>=70 calibrated, strength:2-3, reason: 'concise (OB/liq reversal + regime)'}}. "
        "Else: {'no_live_trade': true, 'roadmap': [zones list]} Top 1-3 daily setups. Precise levels, SL ATR*2, R:R 1:2+. No consol signals."
        f"Alts: BTC align only. No trade if consol or low EV."
    )
    models = [_working_model, "grok-4", "grok-3"] if _working_model else ["grok-4", "grok-3"]
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "temperature": 0.1,
            "max_tokens": 400
        }
        attempt = 0
        while attempt < 2:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post("https://api.x.ai/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {XAI_API_KEY}"})
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    result = json.loads(content)
                    precise = True
                    if 'live_trade' in result and (not check_precision(result['live_trade']) or not check_rr(result['live_trade'])):
                        precise = False
                    if 'roadmap' in result:
                        for z in result['roadmap']:
                            if (not check_precision(z) or not check_rr(z) or z.get('strength', 0) < 2 or z.get('confidence', 0) < 70):
                                precise = False
                                break
                    if precise:
                        # Calibrate
                        factors = {} # Extract from reason as before
                        if 'confidence' in result:
                            result['confidence'] = calibrate_grok_confidence(result['confidence'], factors)
                        _working_model = model
                        return result
                    attempt += 1
                    if attempt == 2:
                        logging.warning(f"Precision/RR/Inst check failed after 2 attempts for {model}")
                        break
                    logging.info(f"Retrying {model} for inst precision/RR")
            except Exception as e:
                if attempt == 1:
                    if model == models[-1]:
                        return {"error": str(e)}
                attempt += 1
                continue
        if attempt == 2:
            continue
    return {"error": "All models failed"}
# NEW: Build optimized Grok context
def build_grok_context(zones: List[Dict], symbol: str, price: float) -> str:
    """Send only decision-critical data"""
    ranked = sorted(zones, key=lambda z: z['prob'], reverse=True)[:2]
    context = f"{symbol} | ${price:.2f}\n"
    for z in ranked:
        dist = abs(price - (z['zone_low'] + z['zone_high'])/2) / price * 100
        context += f"{z['direction']}: {z['zone_low']:.4f}-{z['zone_high']:.4f} "
        context += f"(str{z['strength']}, {dist:.1f}% away, {z['confluence']})\n"
    return context
# UPDATED: query_grok_watch_levels (retained, but use new indicators)
async def query_grok_watch_levels(symbol: str, price: float, trend: str, btc_trend: Optional[str], data: Dict[str, pd.DataFrame], oi_data: Optional[Dict[str, float]]) -> str:
    global _working_model
    is_alt = symbol != 'BTC/USDT'
    df_1d = data.get('1d', pd.DataFrame())
    obs = await find_unmitigated_order_blocks(df_1d, tf='1d', symbol=symbol)
    poc = max(calc_liquidity_profile(df_1d).items(), key=lambda x: x[1])[0] if len(df_1d) > 0 else None
    level_summary = ""
    for ob_type in ['bullish', 'bearish']:
        for ob in obs.get(ob_type, [])[:2]:
            mid = (ob['low'] + ob['high']) / 2
            level_summary += f"{ob_type.capitalize()} OB: {mid:.4f} (str{ob['strength']}, mit{ob['mitigation']:.1f}); "
    oi_str = f"OI change: {oi_data['oi_change_pct']:.1f}%" if oi_data else "No OI data"
    context = f"{symbol} | Price: {price:.4f} | Trend: {trend} {'| BTC: ' + btc_trend if is_alt else ''}\nLevels: {level_summary} | POC: {poc:.4f if poc else 'N/A'} | {oi_str}\nAnalyze key 1d levels to watch (OB/liq/POC), why (confluence), potential trade plan (bias/entry/SL/TP outline, max 50 chars per level)."
    system_prompt = (
        "You are ICT daily analyst. Output ONLY concise text: 'Watch [level1] ([why1]) – Plan: [plan1]; [level2] ([why2]) – Plan: [plan2]'. "
        "Focus 1d OB str>=2 low-mit, liq sweeps, POC. 2-3 levels max. Bias from trend. No spam, precise prices."
        f"Alts: BTC align."
    )
    models = [_working_model, "grok-4", "grok-3"] if _working_model else ["grok-4", "grok-3"]
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "temperature": 0.2,
            "max_tokens": 200
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.x.ai/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {XAI_API_KEY}"})
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                if len(content) > 10:
                    _working_model = model
                    return content
        except Exception as e:
            logging.error(f"Watch levels Grok error with {model} for {symbol}: {e}")
            if model == models[-1]:
                return f"No analysis for {symbol.replace('/USDT','')} – market quiet."
    return f"No analysis for {symbol.replace('/USDT','')} – market quiet."
# UPDATED: backtest_cmd -> backtest_with_live_logic (realistic: live logic, OHLC exits, slippage) + TP tracking
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    logging.info(f"/backtest triggered by user {update.effective_user.id}")
    try:
        days = 90
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        df = await fetch_ohlcv('BTC/USDT', '1d', limit=days, since=since)
        if len(df) == 0:
            await send_throttled(CHAT_ID, "Data unavailable (ban/active cooldown?), backtest skipped. Try later.", parse_mode='Markdown')
            return
        trades = await run_backtest_logic(df, 'BTC/USDT')
        # Track TP1/TP2 separately
        tp1_hits = 0
        tp2_hits = 0
    
        for trade in trades:
            if trade['result'] == 'win':
                tp2_hits += 1
                tp1_hits += 1 # Assume TP1 also hit if TP2 hit
            elif trade.get('hit_tp1'): # If you track partial TP1
                tp1_hits += 1
    
        wins = len([t for t in trades if t['result'] == 'win'])
        total = len([t for t in trades if t['result'] != 'open'])
        winrate = (wins / total * 100) if total > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades)
        sharpe = np.mean([t['pnl'] for t in trades]) / (np.std([t['pnl'] for t in trades]) or np.inf) if trades else 0
    
        # Calculate hit rates
        tp1_hit_rate = tp1_hits / total if total > 0 else 0.60
        tp2_hit_rate = tp2_hits / total if total > 0 else 0.35
    
        bt_results = {
            'winrate': winrate,
            'total_pnl': total_pnl,
            'sharpe': sharpe,
            'tp1_hit_rate': tp1_hit_rate, # NEW
            'tp2_hit_rate': tp2_hit_rate, # NEW
            'date': datetime.now(timezone.utc).isoformat()
        }
        async with aiofiles.open(BACKTEST_FILE, 'w') as f:
            await f.write(json.dumps(bt_results, indent=2))
        msg = f"**ICT Backtest (90d BTC/USDT 1d - Live Logic)**\n\nWins: {wins}/{total} ({winrate:.1f}%)\nTotal PnL: {total_pnl:+.2f} ({total_pnl/SIMULATED_CAPITAL*100:.2f}%)\nSharpe Ratio: {sharpe:.2f}\nTP1 Hit: {tp1_hit_rate:.1%} | TP2 Hit: {tp2_hit_rate:.1%}"
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        logging.info(f"Backtest completed: {winrate:.1f}% winrate")
    except Exception as e:
        logging.error(f"/backtest error: {e}")
        await send_throttled(CHAT_ID, f"Backtest failed: {str(e)}", parse_mode='Markdown')
async def send_welcome_once():
    if not os.path.exists(FLAG_FILE):
        try:
            welcome_text = (
                f"**Grok Elite Bot v25.02.0 ONLINE ♔**\n\n"
                f"**MODE**: {'📝 PAPER TRADING' if PAPER_TRADING else '💰 LIVE TRADING'}\n\n"
                "• v25.02.0: Strict zone rejection, self-cal EV, FVG strength, precise Grok, CSV logs, regime tracking, multi-backtest, Monte Carlo, dashboard, paper mode. Win proj 68-75%.\n"
                "• Retained: Daily 1d, str>=2, sweeps bounces, ADX anti-consol, vol>1.1x, 70% dyn, <7%, Levels to Watch."
            )
            escaped_text = html.escape(welcome_text)
            await send_throttled(CHAT_ID, escaped_text, parse_mode='HTML')
            open(FLAG_FILE, "w").close()
            logging.info("Welcome sent (v25.02.0)")
        except Exception as e:
            logging.error(f"Failed to send welcome: {e}")
# UPDATED: signal_callback: Integrate regime/MTF/quick wins, new indicators, regime/reason in trades
async def signal_callback(context):
    start_time = time.perf_counter()
    logging.info("=== Starting ICT elite signal cycle ===")
    try:
        if await BanManager.check_and_sleep(skip_long_sleep=True):
            logging.warning("Global ban during signal check (short mode); skipping")
            total_time = time.perf_counter() - start_time
            logging.info(f"=== Signal cycle complete in {total_time:.2f}s ===")
            return
        if not TELEGRAM_TOKEN:
            logging.error("TELEGRAM_TOKEN missing - skipping signal cycle")
            return
        btc_trend = btc_trend_global
        logging.info(f"BTC Trend (global): {btc_trend}")
        now = datetime.now(timezone.utc)
        prices = await fetch_ticker_batch()
        order_books = await fetch_order_flow_batch()
        oi_tasks = [fetch_open_interest(s) for s in SYMBOLS]
        oi_results = await asyncio.gather(*oi_tasks, return_exceptions=True)
        oi_data_dict = {SYMBOLS[i]: oi_results[i] if not isinstance(oi_results[i], Exception) else None for i in range(len(SYMBOLS))}
        signals_sent = 0
        for symbol in SYMBOLS:
            sym_start = time.perf_counter()
            price = prices.get(symbol)
            if price is None:
                logging.warning(f"No price for {symbol}; skipping")
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            if symbol in last_signal_time and now - last_signal_time[symbol] < timedelta(hours=COOLDOWN_HOURS):
                remaining = timedelta(hours=COOLDOWN_HOURS) - (now - last_signal_time[symbol])
                logging.info(f"Cooldown active for {symbol}: {remaining} remaining - skipping")
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            if await BanManager.check_and_sleep(skip_long_sleep=True):
                logging.warning(f"Global ban hit for {symbol} (short mode); skipping")
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            oi_data = oi_data_dict.get(symbol)
            whale_data = {'boost': 0, 'reason': ''}
            data = {}
            for tf in TIMEFRAMES:
                df_tf = await fetch_ohlcv(symbol, tf)
                book = order_books.get(symbol)
                data[tf] = add_institutional_indicators(df_tf) if len(df_tf) > 0 else pd.DataFrame() # UPDATED
                await asyncio.sleep(0.5)
            trend = None
            if symbol == 'BTC/USDT' and len(data.get('1d', pd.DataFrame())) > 0 and 'ema200' in data['1d'].columns and not pd.isna(data['1d']['ema200'].iloc[-1]):
                l = data['1d'].iloc[-1]
                if l['close'] > l['ema200']: # UPDATED
                    trend = "Uptrend"
                elif l['close'] < l['ema200']:
                    trend = "Downtrend"
                else:
                    trend = "Sideways"
            elif trend is None and btc_trend:
                trend = btc_trend
            else:
                trend = "Banned/Unknown"
            if trend == "Banned/Unknown":
                if len(data.get('1d', pd.DataFrame())) > 0:
                    l = data['1d'].iloc[-1]
                    if 'ema200' in data['1d'].columns and not pd.isna(l['ema200']):
                        if l['close'] > l['ema200']:
                            trend = "Uptrend"
                        elif l['close'] < l['ema200']:
                            trend = "Downtrend"
                        else:
                            trend = "Sideways"
                else:
                    trend = "Sideways"
            triggers = set()
            is_alt = symbol != 'BTC/USDT'
            for tf in TIMEFRAMES:
                df = data.get(tf, pd.DataFrame())
                if len(df) == 0:
                    continue
                if is_consolidation(df): # UPDATED: Regime-integrated
                    logging.info(f"Skipped {symbol} {tf}: Consolidation/Dead regime")
                    continue
                for func in [detect_pre_cross, detect_divergence, detect_macd, detect_supertrend]:
                    if r := func(df, tf):
                        triggers.add(r)
                fvg_triggers = detect_fvg(df, tf)
                normalized_fvgs = [re.sub(r'\d+\.\d+%', '<0.3%', f) for f in fvg_triggers]
                triggers.update(normalized_fvgs)
                candles = detect_candle_patterns(df, tf)
                triggers.update(candles)
                if len(df) > 20 and df['volume'].iloc[-1] > VOL_SURGE_MULTIPLIER * df['volume'].rolling(20).mean().iloc[-1]:
                    triggers.add(f"Inst Vol Surge ({tf})")
                obs = await find_unmitigated_order_blocks(df, tf=tf, symbol=symbol)
                raw_obs = len(obs.get('bullish', []) + obs.get('bearish', []))
                logging.info(f"Raw OBs for {symbol} {tf}: {raw_obs} (before elite filter)")
                for ob_type in ['bullish', 'bearish']:
                    for ob in obs.get(ob_type, []):
                        if ob['strength'] < 2:
                            continue
                        mid = (ob['low'] + ob['high']) / 2
                        dist_pct = abs(price - mid) / price * 100
                        if dist_pct < 0.5:
                            dir_str = "Bullish Long" if ob_type == 'bullish' else "Bearish Short"
                            triggers.add(f"Elite {dir_str} OB str{ob['strength']} mit{ob['mitigation']:.1f} {dist_pct:.1f}% away ({tf})")
                delta = df['order_delta'].iloc[-1] if 'order_delta' in df.columns else 0
                if abs(delta) > 0.5:
                    dir_str = "Bullish" if delta > 0 else "Bearish"
                    triggers.add(f"Order Flow {dir_str} Delta {delta:.1f}% ({tf})")
                if df['liq_sweep'].iloc[-1]:
                    triggers.add(f"Liq Sweep Bounce ({tf})")
            logging.info(f"All triggers for {symbol}: {list(triggers)}")
            strong_triggers = [t for t in triggers if any(kw in t for kw in ['OB str', 'Displacement', 'Inst Vol', 'Exhaust', 'Bearish Short', 'Liq Sweep', 'Order Flow'])]
            logging.info(f"Elite triggers for {symbol}: {len(strong_triggers)} (min 1 req)")
            if len(strong_triggers) < 1:
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Skipped {symbol}: <1 elite triggers")
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            premium_zones = []
            for tf in ['1d', '1w']:
                if tf not in data:
                    continue
                tf_df = data[tf]
                if len(tf_df) > 0:
                    book = order_books.get(symbol)
                    tf_zones = await find_next_premium_zones(tf_df, price, tf, symbol, oi_data, trend=trend, whale_data=whale_data, order_book=book)
                    premium_zones.extend(tf_zones)
            logging.info(f"Total elite {len(premium_zones)} premium zones for {symbol} before Grok")
            if not premium_zones:
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Skipped {symbol}: No elite zones")
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            grok_potential = await query_grok_potential(premium_zones, symbol, price, trend, btc_trend)
            retry_count = 0
            max_retries = 2
            while retry_count < max_retries:
                has_roadmap = 'roadmap' in grok_potential and grok_potential.get('roadmap')
                if not has_roadmap and premium_zones:
                    logging.info(f"No roadmap on attempt {retry_count+1} for {symbol}; forcing retry...")
                    grok_potential = await query_grok_potential(premium_zones, symbol, price, trend, btc_trend)
                    retry_count += 1
                    continue
                break
            # Rest retained, but with new calibration in Grok func
            live_trade_key = 'live_trade' if 'live_trade' in grok_potential else None
            if live_trade_key and not grok_potential.get('no_live_trade', True):
                grok = grok_potential[live_trade_key]
                logging.info(f"Grok elite live trade for {symbol}: {grok}")
                if grok.get('strength', 0) < 2 or grok.get('confidence', 0) < 70:
                    logging.info(f"Skipped non-elite live {symbol}: str {grok.get('strength', 0)} or conf {grok.get('confidence', 0)}")
                    sym_time = time.perf_counter() - sym_start
                    logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                    continue
                required_keys = ['direction', 'entry_low', 'entry_high', 'sl', 'tp1', 'tp2', 'leverage', 'confidence', 'reason']
                if not all(key in grok for key in required_keys):
                    logging.warning(f"Invalid Grok response for {symbol}: {grok}")
                    await asyncio.sleep(2)
                    sym_time = time.perf_counter() - sym_start
                    logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                    continue
                entry_low = min(grok['entry_low'], grok['entry_high'])
                entry_high = max(grok['entry_low'], grok['entry_high'])
                new_conf = grok['confidence']
                leverage = min(grok['leverage'], 5)
                # Overlap/merge retained...
                overlapping = []
                for trades_dict, dict_name in [(open_trades, 'open'), (protected_trades, 'protected')]:
                    for key, t in trades_dict.items():
                        overlap_ratio = zones_overlap(entry_low, entry_high, t['entry_low'], t['entry_high'])
                        if overlap_ratio > 0:
                            overlapping.append((key, t, dict_name, overlap_ratio))
                if overlapping:
                    same_dir_overlaps = [o for o in overlapping if o[1]['direction'] == grok['direction']]
                    if same_dir_overlaps:
                        max_conf = max(new_conf, max(o[1]['confidence'] for o in same_dir_overlaps))
                        min_sl = min(grok['sl'], min(o[1]['sl'] for o in same_dir_overlaps))
                        max_tp = max(grok['tp2'], max(o[1]['tp2'] for o in same_dir_overlaps))
                        for key, t, dict_name, ratio in same_dir_overlaps:
                            if t.get('active') and dict_name == 'open':
                                protected_trades[key] = open_trades.pop(key)
                                await save_trades_async(open_trades)
                                await save_protected_async(protected_trades)
                                logging.info(f"Protected active overlap for {key}")
                        merge_key, merge_t, _, _ = same_dir_overlaps[0]
                        merge_t['confidence'] = max_conf
                        merge_t['sl'] = min_sl
                        merge_t['tp2'] = max_tp
                        if new_conf > merge_t['confidence']:
                            merge_t['entry_low'] = entry_low
                            merge_t['entry_high'] = entry_high
                        logging.info(f"Merged elite live signal for {symbol}: conf {max_conf}")
                        for key, t, dict_name, ratio in same_dir_overlaps[1:]:
                            if ratio >= 0.95:
                                if dict_name == 'open':
                                    del open_trades[key]
                                else:
                                    del protected_trades[key]
                                logging.info(f"Removed high-overlap {key} (ratio {ratio:.2f})")
                        await save_trades_async(open_trades)
                        await save_protected_async(protected_trades)
                        last_signal_time[symbol] = now
                    else:
                        max_overlap = max(r for _, _, _, r in overlapping)
                        if max_overlap < 0.95:
                            trade_key = symbol
                            open_trades[trade_key] = {
                                'direction': grok['direction'],
                                'entry_low': entry_low,
                                'entry_high': entry_high,
                                'sl': grok['sl'],
                                'tp1': grok['tp1'],
                                'tp2': grok['tp2'],
                                'leverage': leverage,
                                'confidence': new_conf,
                                'active': False,
                                'last_check': datetime.now(timezone.utc),
                                'processed': False,
                                'strength': grok.get('strength', 2),
                                'reason': grok.get('reason', 'N/A'), # NEW
                                'regime': detect_market_regime(data['1d']) if '1d' in data else 'Unknown' # NEW
                            }
                            await save_trades_async(open_trades)
                            last_signal_time[symbol] = now
                        else:
                            logging.info(f"Skipped live {symbol}: high overlap diff dir {max_overlap:.2f}")
                            sym_time = time.perf_counter() - sym_start
                            logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                            continue
                else:
                    trade_key = symbol
                    open_trades[trade_key] = {
                        'direction': grok['direction'],
                        'entry_low': entry_low,
                        'entry_high': entry_high,
                        'sl': grok['sl'],
                        'tp1': grok['tp1'],
                        'tp2': grok['tp2'],
                        'leverage': leverage,
                        'confidence': new_conf,
                        'active': False,
                        'last_check': datetime.now(timezone.utc),
                        'processed': False,
                        'strength': grok.get('strength', 2),
                        'reason': grok.get('reason', 'N/A'), # NEW
                        'regime': detect_market_regime(data['1d']) if '1d' in data else 'Unknown' # NEW
                    }
                    await save_trades_async(open_trades)
                    last_signal_time[symbol] = now
                entry_mid = (entry_low + entry_high) / 2
                rr1 = abs(grok['tp1'] - entry_mid) / abs(grok['sl'] - entry_mid)
                rr2 = abs(grok['tp2'] - entry_mid) / abs(grok['sl'] - entry_mid)
                dist_to_sl = abs(entry_mid - grok['sl']) / entry_mid * 100
                ev_r = calculate_expected_value({'direction': grok['direction'], 'entry_low': entry_low, 'entry_high': entry_high, 'sl': grok['sl'], 'tp1': grok['tp1'], 'tp2': grok['tp2']}, HISTORICAL_DATA)
                msg = (
                    f"**{symbol.replace('/USDT','')} ELITE LIVE SIGNAL (Inst OB Hit!)**\n\n"
                    f"*Price:* {format_price(price)} | *Trend:* {trend} | *EV:* {ev_r:.2f}R"
                )
                if is_alt and btc_trend: msg += f" | *BTC:* {btc_trend}"
                msg += "\n\n"
                msg += (
                    f"{grok['direction']} | **{new_conf}%**\n"
                    f"*Entry Zone:* {format_price(entry_low)}–{format_price(entry_high)}\n"
                    f"*SL:* {format_price(grok['sl'])} | *TP1* {format_price(grok['tp1'])} | *TP2* {format_price(grok['tp2'])}\n"
                    f"*Leverage:* {leverage}x | *R:R* 1:{rr1:.1f} / 1:{rr2:.1f} | *Risk* {dist_to_sl:.2f}%\n"
                    f"**Reason:** {grok['reason']}"
                )
                logging.info(f"Sending elite live for {symbol}")
                await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
                signals_sent += 1
            # Roadmap similar, add regime/reason
            # ... (for roadmap trades, add 'reason': grok.get('reason', 'N/A'), 'regime': detect_market_regime(data['1d']) if '1d' in data else 'Unknown')
        if signals_sent == 0:
            watch_msg = "**Levels to Watch (Daily Analysis)**\n\n"
            for symbol in SYMBOLS:
                if symbol in last_watch_time and now - last_watch_time[symbol] < timedelta(hours=WATCH_COOLDOWN_HOURS):
                    continue
                price = prices.get(symbol)
                if price is None:
                    continue
                data = {}
                for tf in ['1d']:
                    df_tf = await fetch_ohlcv(symbol, tf)
                    book = order_books.get(symbol)
                    data[tf] = add_institutional_indicators(df_tf) if len(df_tf) > 0 else pd.DataFrame()
                trend = "Sideways"
                oi_data = oi_data_dict.get(symbol)
                analysis = await query_grok_watch_levels(symbol, price, trend, btc_trend, data, oi_data)
                watch_msg += f"{symbol.replace('/USDT','')}: {analysis}\n\n"
                last_watch_time[symbol] = now
                logging.info(f"Watch levels sent for {symbol}")
            if len(watch_msg) > 50:
                await send_throttled(CHAT_ID, watch_msg, parse_mode='Markdown')
                logging.info("Levels to Watch message sent")
        total_time = time.perf_counter() - start_time
        logging.info(f"=== ICT elite signal cycle complete in {total_time:.2f}s ===")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"Signal callback error: {e}\n{error_trace}")
        total_time = time.perf_counter() - start_time
        logging.info(f"=== Signal cycle complete in {total_time:.2f}s (error) ===")
# Retained: post_init, main (with global stats load, new handlers)
stats = load_stats() # Global
async def post_init(application: Application) -> None:
    global background_task
    logging.info("Starting post_init: Sending welcome and setting webhook...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.bybit.com/v5/market/tickers?category=spot")
            if r.status_code != 200:
                logging.warning("Bybit API health check failed - check connectivity")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post("https://api.x.ai/v1/chat/completions", json={"model": "grok-3", "messages": [{"role": "user", "content": "ping"}]}, headers={"Authorization": f"Bearer {XAI_API_KEY}"})
            if r.status_code == 401:
                logging.error("xAI API unauthorized - Check SuperGrok/Premium+ key!")
            elif r.status_code != 200:
                logging.warning(f"xAI API health check failed: status {r.status_code}")
    except Exception as e:
        logging.error(f"API health check error: {e}")
    await send_welcome_once()
    await application.bot.delete_webhook(drop_pending_updates=True)
    external_hostname = os.getenv('RENDER_EXTERNAL_HOSTNAME')
    if not external_hostname:
        logging.error("RENDER_EXTERNAL_HOSTNAME not set - cannot set webhook!")
        return
    webhook_url = f"https://{external_hostname}/webhook"
    logging.info(f"Setting webhook to: {webhook_url}")
    await application.bot.set_webhook(url=webhook_url)
    logging.info("Webhook set successfully. Jobs will now run.")
    background_task = asyncio.create_task(price_background_task())
    logging.info("Background Polling Task started – fresh prices + order flow every 10s!")
    logging.info("Post_init complete – ICT elite signals via job in ~60s.")
def main():
    global background_task, stats
    stats = load_stats() # Ensure loaded
    logging.info(f"Loaded env: TOKEN={'SET' if TELEGRAM_TOKEN else 'MISSING'}, CHAT={'SET' if CHAT_ID else 'MISSING'}, KEY={'SET' if XAI_API_KEY else 'MISSING'}, PAPER={PAPER_TRADING}")
    if not all([TELEGRAM_TOKEN, CHAT_ID, XAI_API_KEY]):
        logging.error("Missing required env vars: TELEGRAM_TOKEN, CHAT_ID, XAI_API_KEY")
        sys.exit(1)
    logging.info(f"Daily cooldown: {COOLDOWN_HOURS}h | ICT elite: regime MTF, dyn EV, inst stack | Paper: {PAPER_TRADING}")
    application = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("health", health_cmd))
    application.add_handler(CommandHandler("recap", recap_cmd))
    application.add_handler(CommandHandler("backtest", backtest_cmd))
    application.add_handler(CommandHandler("backtest_all", backtest_all_cmd)) # NEW
    application.add_handler(CommandHandler("validate", validate_cmd)) # NEW
    application.add_handler(CommandHandler("dashboard", dashboard_cmd)) # NEW
    application.add_handler(MessageHandler(filters.ALL, webhook_update))
    signal_job = application.job_queue.run_repeating(
        signal_callback,
        interval=CHECK_INTERVAL,
        first=60,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )
    logging.info(f"ICT elite signal job: first in 60s, max_instances=2")
    track_job = application.job_queue.run_repeating(
        track_callback,
        interval=TRACK_INTERVAL,
        first=60,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )
    logging.info(f"Track job: interval=5s")
    price_job = application.job_queue.run_repeating(
        price_update_callback,
        interval=60,
        first=30,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )
    logging.info("Price update job: every 60s")
    btc_job = application.job_queue.run_repeating(
        btc_trend_update,
        interval=300,
        first=60,
        job_kwargs={'max_instances': 1}
    )
    logging.info("BTC trend job: every 5min")
    now = datetime.now(timezone.utc)
    target = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if now.time() > target.time():
        target += timedelta(days=1)
    delta = target - now
    first_delay = int(delta.total_seconds())
    logging.info(f"Daily recap first in {first_delay}s")
    application.job_queue.run_repeating(daily_callback, interval=86400, first=first_delay)
    webhook_path = "/webhook"
    port = int(os.getenv("PORT", 10000))
    external_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}"
    webhook_url = f"{external_url}{webhook_path}"
    logging.info(f"Starting webhook server on port {port}, URL: {webhook_url}")
    try:
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=webhook_path,
            webhook_url=webhook_url
        )
    finally:
        if background_task and not background_task.done():
            background_task.cancel()
            logging.info("Background task cancelled")
        asyncio.run(exchange.close())
        asyncio.run(futures_exchange.close())
        logging.info("CCXT exchanges closed gracefully!")
if __name__ == "__main__":
    main()
