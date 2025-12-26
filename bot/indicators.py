# indicators.py - Grok Elite Signal Bot v27.8.2 - ICT Indicators + Advanced Filtering
"""
ICT (Inner Circle Trader) indicators and market structure analysis:
- Institutional orderflow indicators
- Market regime detection (RELAXED v27.8.2)
- FVG (Fair Value Gap) analysis
- Premium/discount zones
- Liquidity sweeps
- Pattern detection
- Order flow calculation
- Session-aware analysis
- Momentum indicators (v27.8.0)
- Trend strength calculation (v27.8.0)

v27.8.2: Relaxed dead regime threshold to prevent blocking valid 4h signals
"""
import logging
from typing import List, Optional, Dict, Tuple
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import time as dtime

from bot.config import (
    VOL_SURGE_MULTIPLIER, FVG_DISPLACEMENT_MULT, ZONE_LOOKBACK,
    PRE_CROSS_THRESHOLD_PCT, DISPLACEMENT_BODY_RATIO, PINBAR_WICK_RATIO,
    OB_MIN_STRENGTH, ORDERBOOK_DEPTH, WALL_MULTIPLIER,
    MOMENTUM_ROC_PERIOD, MOMENTUM_LOOKBACK, MOMENTUM_STRONG_THRESHOLD
)

# ============================================================================
# MOMENTUM INDICATORS (v27.8.0)
# ============================================================================
def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum indicators for signal confirmation.
    
    Adds:
    - roc: Rate of Change
    - momentum: Price momentum
    - momentum_signal: 'bullish', 'bearish', or 'neutral'
    - macd, macd_signal, macd_hist: MACD indicator
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with momentum columns added
    """
    if len(df) < 30:
        df['roc'] = 0
        df['momentum'] = 0
        df['momentum_signal'] = 'neutral'
        df['macd'] = 0
        df['macd_signal'] = 0
        df['macd_hist'] = 0
        return df
    
    df = df.copy()
    
    # Rate of Change
    df['roc'] = ta.roc(df['close'], length=MOMENTUM_ROC_PERIOD)
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(MOMENTUM_LOOKBACK)
    
    # MACD
    macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_result is not None and len(macd_result.columns) >= 3:
        df['macd'] = macd_result.iloc[:, 0]
        df['macd_signal'] = macd_result.iloc[:, 1]
        df['macd_hist'] = macd_result.iloc[:, 2]
    else:
        df['macd'] = 0
        df['macd_signal'] = 0
        df['macd_hist'] = 0
    
    # Classify momentum signal
    df['momentum_signal'] = 'neutral'
    
    # Bullish momentum: ROC > threshold and MACD hist > 0
    bullish_mask = (df['roc'] > MOMENTUM_STRONG_THRESHOLD) & (df['macd_hist'] > 0)
    df.loc[bullish_mask, 'momentum_signal'] = 'bullish'
    
    # Bearish momentum: ROC < -threshold and MACD hist < 0
    bearish_mask = (df['roc'] < -MOMENTUM_STRONG_THRESHOLD) & (df['macd_hist'] < 0)
    df.loc[bearish_mask, 'momentum_signal'] = 'bearish'
    
    return df

def get_momentum_confluence(df: pd.DataFrame, direction: str) -> Tuple[float, str]:
    """
    Calculate momentum confluence for a trade direction.
    
    Args:
        df: DataFrame with momentum indicators
        direction: 'Long' or 'Short'
    
    Returns:
        Tuple of (confluence_score, confluence_string)
    """
    if len(df) == 0 or 'momentum_signal' not in df.columns:
        return 0.0, ""
    
    mom_signal = df['momentum_signal'].iloc[-1]
    roc = df['roc'].iloc[-1] if 'roc' in df.columns else 0
    macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else 0
    
    if pd.isna(roc):
        roc = 0
    if pd.isna(macd_hist):
        macd_hist = 0
    
    confluence_score = 0.0
    confluence_str = ""
    
    # Strong alignment
    if direction == 'Long' and mom_signal == 'bullish':
        confluence_score = 5.0
        confluence_str = f"+StrongMom({roc:+.1f}%)"
    elif direction == 'Short' and mom_signal == 'bearish':
        confluence_score = 5.0
        confluence_str = f"+StrongMom({roc:+.1f}%)"
    
    # Moderate alignment
    elif direction == 'Long' and roc > 0:
        confluence_score = 2.0
        confluence_str = f"+Mom({roc:+.1f}%)"
    elif direction == 'Short' and roc < 0:
        confluence_score = 2.0
        confluence_str = f"+Mom({roc:+.1f}%)"
    
    # Counter-momentum (reduce confidence)
    elif direction == 'Long' and mom_signal == 'bearish':
        confluence_score = -3.0
        confluence_str = f"-CounterMom({roc:+.1f}%)"
    elif direction == 'Short' and mom_signal == 'bullish':
        confluence_score = -3.0
        confluence_str = f"-CounterMom({roc:+.1f}%)"
    
    return confluence_score, confluence_str

def get_trend_strength(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate comprehensive trend strength metrics.
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        Dict with trend analysis
    """
    if len(df) < 50:
        return {
            'strength': 0,
            'direction': 'neutral',
            'ema_alignment': False,
            'adx_value': 0,
            'momentum_aligned': False
        }
    
    df = df.copy()
    
    # Calculate ADX if not present
    if 'adx' not in df.columns:
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_result is not None and 'ADX_14' in adx_result.columns:
            adx_val = adx_result['ADX_14'].iloc[-1]
        else:
            adx_val = 20
    else:
        adx_val = df['adx'].iloc[-1] if pd.notna(df['adx'].iloc[-1]) else 20
    
    # EMA alignment
    ema_20 = ta.ema(df['close'], 20).iloc[-1]
    ema_50 = ta.ema(df['close'], 50).iloc[-1]
    ema_200 = df['ema200'].iloc[-1] if 'ema200' in df.columns else ta.ema(df['close'], 200).iloc[-1]
    
    current_price = df['close'].iloc[-1]
    
    # Determine direction
    if pd.notna(ema_200):
        if current_price > ema_200 and ema_20 > ema_50:
            direction = 'bullish'
            ema_alignment = True
        elif current_price < ema_200 and ema_20 < ema_50:
            direction = 'bearish'
            ema_alignment = True
        else:
            direction = 'neutral'
            ema_alignment = False
    else:
        direction = 'neutral'
        ema_alignment = False
    
    # Momentum alignment
    mom_signal = df['momentum_signal'].iloc[-1] if 'momentum_signal' in df.columns else 'neutral'
    momentum_aligned = (
        (direction == 'bullish' and mom_signal == 'bullish') or
        (direction == 'bearish' and mom_signal == 'bearish')
    )
    
    # Calculate strength (0-100)
    strength = 0
    
    # ADX contribution (0-40)
    if pd.notna(adx_val):
        strength += min(adx_val, 40)
    
    # EMA alignment contribution (0-30)
    if ema_alignment:
        strength += 30
    
    # Momentum alignment contribution (0-30)
    if momentum_aligned:
        strength += 30
    
    return {
        'strength': int(strength),
        'direction': direction,
        'ema_alignment': ema_alignment,
        'adx_value': adx_val if pd.notna(adx_val) else 0,
        'momentum_aligned': momentum_aligned
    }

# ============================================================================
# KALMAN FILTER FOR TREND
# ============================================================================
class KalmanFilter:
    """
    1D Kalman filter for price trend estimation.
    Reduces noise and provides smooth trend indication.
    """
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement and return filtered value.
        """
        if not self.initialized:
            self.posteri_estimate = measurement
            self.initialized = True
            return measurement
        
        # Prediction
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # Update
        kalman_gain = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate
        
        return self.posteri_estimate

def apply_kalman_filter(series: pd.Series) -> pd.Series:
    """
    Apply Kalman filter to price series for trend estimation.
    
    Args:
        series: Price series
    
    Returns:
        Filtered (smoothed) series
    """
    kf = KalmanFilter()
    filtered = []
    
    for val in series:
        if pd.notna(val):
            filtered.append(kf.update(val))
        else:
            filtered.append(np.nan)
    
    return pd.Series(filtered, index=series.index)

# ============================================================================
# VWAP BANDS
# ============================================================================
def calculate_vwap_bands(df: pd.DataFrame, std_mult: float = 2.0) -> pd.DataFrame:
    """
    Calculate VWAP with standard deviation bands.
    Helps identify premium/discount zones relative to volume-weighted average.
    
    Args:
        df: DataFrame with OHLCV and vwap
        std_mult: Standard deviation multiplier for bands
    
    Returns:
        DataFrame with vwap_upper, vwap_lower, vwap_position
    """
    if 'vwap' not in df.columns or len(df) < 20:
        df['vwap_upper'] = np.nan
        df['vwap_lower'] = np.nan
        df['vwap_position'] = 0.5
        return df
    
    df = df.copy()
    
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate squared deviation from VWAP
    squared_dev = (typical_price - df['vwap']) ** 2
    
    # Volume-weighted variance
    cum_volume = df['volume'].rolling(20, min_periods=1).sum()
    cum_squared_dev = (squared_dev * df['volume']).rolling(20, min_periods=1).sum()
    
    vwap_variance = cum_squared_dev / cum_volume
    vwap_std = np.sqrt(vwap_variance)
    
    # Calculate bands
    df['vwap_upper'] = df['vwap'] + (vwap_std * std_mult)
    df['vwap_lower'] = df['vwap'] - (vwap_std * std_mult)
    
    # Calculate position within bands (0 = at lower band, 1 = at upper band)
    band_range = df['vwap_upper'] - df['vwap_lower']
    df['vwap_position'] = np.where(
        band_range > 0,
        (df['close'] - df['vwap_lower']) / band_range,
        0.5
    )
    df['vwap_position'] = df['vwap_position'].clip(0, 1)
    
    return df

# ============================================================================
# SESSION QUALITY
# ============================================================================
def get_session_quality_score(timestamp: pd.Timestamp) -> float:
    """
    Score trading quality based on time of day (UTC).
    
    High-quality sessions:
    - London open (08:00-14:00 UTC): 1.0
    - NY open (14:00-21:00 UTC): 1.0
    
    Medium-quality sessions:
    - Asia close (21:00-24:00 UTC): 0.7
    
    Low-quality sessions:
    - Dead zone (00:00-08:00 UTC): 0.5
    
    Args:
        timestamp: Datetime to score
    
    Returns:
        Quality score (0.5 - 1.0)
    """
    hour = timestamp.hour
    
    # London session (08:00-14:00)
    if 8 <= hour < 14:
        return 1.0
    
    # NY session (14:00-21:00)
    elif 14 <= hour < 21:
        return 1.0
    
    # Asia close (21:00-24:00)
    elif 21 <= hour < 24:
        return 0.7
    
    # Dead zone (00:00-08:00)
    else:
        return 0.5

def add_session_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add session quality scores to dataframe.
    
    Args:
        df: DataFrame with 'date' column
    
    Returns:
        DataFrame with 'session_quality' column
    """
    if 'date' not in df.columns:
        df['session_quality'] = 1.0
        return df
    
    df = df.copy()
    df['session_quality'] = df['date'].apply(get_session_quality_score)
    
    return df

# ============================================================================
# LIQUIDITY SWEEP DETECTION
# ============================================================================
def detect_liquidity_sweep(df: pd.DataFrame, atr_window: int = 14) -> pd.Series:
    """
    ICT-compliant sweep: breaches key level with volume + reversal.
    """
    if len(df) < 20:
        return pd.Series([False] * len(df), index=df.index)
    
    df = df.copy()
    
    if 'atr' not in df.columns:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], atr_window)
    
    atr_val = df['atr'].iloc[-1]
    close_val = df['close'].iloc[-1]
    
    if pd.isna(atr_val) or close_val == 0:
        lookback = 20
    else:
        lookback = int(atr_val / (close_val * 0.01) * 5)
        lookback = max(20, min(lookback, 100))
    
    swing_high = df['high'].rolling(lookback, min_periods=1).max()
    swing_low = df['low'].rolling(lookback, min_periods=1).min()
    
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    candle_range = df['high'] - df['low']
    candle_range = candle_range.replace(0, np.nan)
    
    vol_mean = df['volume'].rolling(20, min_periods=1).mean()
    vol_surge = df['volume'] > 1.5 * vol_mean
    
    bull_sweep = (
        (df['low'] < swing_low.shift(1)) &
        (df['close'] > swing_low.shift(1) * 1.002) &
        (lower_wick > candle_range * 0.6) &
        vol_surge
    )
    
    bear_sweep = (
        (df['high'] > swing_high.shift(1)) &
        (df['close'] < swing_high.shift(1) * 0.998) &
        (upper_wick > candle_range * 0.6) &
        vol_surge
    )
    
    result = (bull_sweep | bear_sweep).fillna(False)
    return result

# ============================================================================
# ORDER FLOW CALCULATION
# ============================================================================
def calculate_order_flow(df: pd.DataFrame, order_book: Optional[Dict] = None) -> pd.DataFrame:
    """Calculate order flow metrics from OHLCV data and orderbook."""
    if len(df) == 0:
        return df
    
    df = df.copy()
    
    if 'atr' not in df.columns:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
    
    df['liq_sweep'] = detect_liquidity_sweep(df)
    
    if not order_book:
        df['order_delta'] = 0.0
        df['cum_delta'] = 0.0
        df['footprint_imbalance'] = False
        df['bid_walls'] = None
        df['ask_walls'] = None
        return df
    
    bids = order_book.get('bids', [])[:ORDERBOOK_DEPTH]
    asks = order_book.get('asks', [])[:ORDERBOOK_DEPTH]
    
    if not bids or not asks:
        df['order_delta'] = 0.0
        df['cum_delta'] = 0.0
        df['footprint_imbalance'] = False
        df['bid_walls'] = None
        df['ask_walls'] = None
        return df
    
    buy_vol = sum(amount for _, amount in bids)
    sell_vol = sum(amount for _, amount in asks)
    total_vol = buy_vol + sell_vol
    
    delta = (buy_vol - sell_vol) / total_vol * 100 if total_vol > 0 else 0
    df['order_delta'] = delta
    
    buy_candle_vol = np.where(df['close'] > df['open'], df['volume'], 0)
    sell_candle_vol = np.where(df['close'] < df['open'], df['volume'], 0)
    df['cum_delta'] = (buy_candle_vol - sell_candle_vol).cumsum()
    
    # Wall detection
    mid_price = df['close'].iloc[-1]
    bid_walls, ask_walls = detect_orderbook_walls(bids, asks, mid_price)
    
    df['bid_walls'] = [bid_walls] * len(df)
    df['ask_walls'] = [ask_walls] * len(df)
    
    active_levels = len([p for p, _ in bids + asks if abs(p - mid_price) / mid_price < 0.01])
    df['footprint_imbalance'] = active_levels > 10
    
    logging.debug(f"Order flow: delta {delta:.2f}% | Walls: {len(bid_walls)} bids, {len(ask_walls)} asks")
    
    return df

# ============================================================================
# ORDERBOOK WALL DETECTION
# ============================================================================
def detect_orderbook_walls(
    bids: List[Tuple[float, float]], 
    asks: List[Tuple[float, float]], 
    current_price: float
) -> Tuple[Dict[float, float], Dict[float, float]]:
    """
    Find significant liquidity walls in orderbook.
    """
    def find_walls(orders: List[Tuple[float, float]], bucket_size: float = 100) -> Dict[float, float]:
        if not orders:
            return {}
        
        buckets = {}
        for price, vol in orders:
            bucket = round(price / bucket_size) * bucket_size
            buckets[bucket] = buckets.get(bucket, 0) + vol
        
        if not buckets:
            return {}
        
        avg_vol = sum(buckets.values()) / len(buckets)
        walls = {p: v for p, v in buckets.items() if v > avg_vol * WALL_MULTIPLIER}
        
        return walls
    
    if current_price > 10000:
        bucket_size = 100
    elif current_price > 1000:
        bucket_size = 10
    elif current_price > 100:
        bucket_size = 1
    else:
        bucket_size = 0.1
    
    bid_walls = find_walls(bids, bucket_size)
    ask_walls = find_walls(asks, bucket_size)
    
    return bid_walls, ask_walls

# ============================================================================
# MAIN INDICATOR FUNCTION (v27.8.2)
# ============================================================================
def add_institutional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add institutional-grade indicators focused on orderflow + structure.
    
    v27.8.2: Includes momentum indicators, MACD
    """
    if len(df) == 0:
        return df
    
    if not isinstance(df.index, (pd.DatetimeIndex, pd.RangeIndex)):
        df = df.reset_index(drop=True)
    
    df = df.copy()
    
    # ========================================================================
    # CORE ICT INDICATORS
    # ========================================================================
    
    # 1. PRICE STRUCTURE
    df['swing_high'] = df['high'].rolling(21, center=True, min_periods=1).max() == df['high']
    df['swing_low'] = df['low'].rolling(21, center=True, min_periods=1).min() == df['low']
    
    # 2. VOLUME PROFILE
    df['vol_delta'] = df['volume'] - df['volume'].shift(1)
    df['vol_acceleration'] = df['vol_delta'] - df['vol_delta'].shift(1)
    
    # 3. CUMULATIVE DELTA
    df['buy_vol'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['sell_vol'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['cum_delta'] = (df['buy_vol'] - df['sell_vol']).cumsum()
    
    # 4. ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
    df['atr_pct'] = df['atr'].rank(pct=True)
    
    # 5. SPREAD PROXY
    df['spread_proxy'] = (df['high'] - df['low']) / df['close'] * 100
    
    # 6. EMA200 + RSI
    df['ema200'] = ta.ema(df['close'], 200)
    df['rsi'] = ta.rsi(df['close'], 14)
    
    # 7. FVG strength
    df = calculate_fvg_strength(df)
    
    # 8. Premium/discount zones
    df = calculate_premium_discount(df, lookback=ZONE_LOOKBACK)
    
    # ========================================================================
    # MOMENTUM INDICATORS (v27.8.0)
    # ========================================================================
    df = calculate_momentum_indicators(df)
    
    # ========================================================================
    # OPTIONAL FEATURES
    # ========================================================================
    
    # VWAP Bands (if VWAP exists)
    if 'vwap' in df.columns:
        df = calculate_vwap_bands(df, std_mult=2.0)
    else:
        df['vwap_upper'] = np.nan
        df['vwap_lower'] = np.nan
        df['vwap_position'] = 0.5
    
    # Session quality
    if 'date' in df.columns:
        df = add_session_quality(df)
    else:
        df['session_quality'] = 1.0
    
    # ========================================================================
    
    # Placeholders for order flow (calculated separately)
    if 'liq_sweep' not in df.columns:
        df['liq_sweep'] = False
    if 'order_delta' not in df.columns:
        df['order_delta'] = 0.0
    if 'footprint_imbalance' not in df.columns:
        df['footprint_imbalance'] = False
    
    # Ensure 'date' column exists
    if 'date' not in df.columns and 'ts' in df.columns:
        df['date'] = pd.to_datetime(df['ts'], unit='ms')
    
    return df

# ============================================================================
# FVG ANALYSIS
# ============================================================================
def calculate_fvg_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Grade FVG quality using volume + displacement."""
    df['fvg_strength'] = 0.0
    
    if len(df) < 3:
        return df
    
    if 'atr' not in df.columns:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
    
    fvgs = []
    vol_rolling = df['volume'].rolling(20, min_periods=1).mean()
    
    for i in range(2, len(df)):
        atr_val = df['atr'].iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue
        
        vol_avg = vol_rolling.iloc[i]
        if pd.isna(vol_avg) or vol_avg == 0:
            continue
        
        # Bullish FVG
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            vol_at_gap = df['volume'].iloc[i-1]
            displacement = abs(df['close'].iloc[i] - df['open'].iloc[i-2]) / df['close'].iloc[i]
            
            strength = (
                (vol_at_gap / vol_avg) * 2 +
                (gap_size / atr_val) * 3 +
                (displacement * 100) * FVG_DISPLACEMENT_MULT
            )
            
            fvgs.append({'idx': i, 'type': 'bull', 'strength': strength})
        
        # Bearish FVG
        elif df['low'].iloc[i-2] > df['high'].iloc[i]:
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            vol_at_gap = df['volume'].iloc[i-1]
            displacement = abs(df['close'].iloc[i] - df['open'].iloc[i-2]) / df['close'].iloc[i]
            
            strength = (
                (vol_at_gap / vol_avg) * 2 +
                (gap_size / atr_val) * 3 +
                (displacement * 100) * FVG_DISPLACEMENT_MULT
            )
            
            fvgs.append({'idx': i, 'type': 'bear', 'strength': strength})
    
    for fvg in fvgs:
        try:
            df.iloc[fvg['idx'], df.columns.get_loc('fvg_strength')] = fvg['strength']
        except (KeyError, IndexError):
            continue
    
    return df

def detect_fvg(df: pd.DataFrame, tf: str, lookback: int = 50, proximity_pct: float = 0.3) -> List[str]:
    """Detect FVGs near current price for signal generation."""
    if len(df) < 5:
        return []
    
    df_local = df.tail(lookback).copy()
    fvgs = []
    price = df_local['close'].iloc[-1]
    
    for i in range(2, len(df_local) - 1):
        # Bullish FVG
        if df_local['high'].iloc[i-2] < df_local['low'].iloc[i]:
            fvg_low = df_local['high'].iloc[i-2]
            fvg_high = df_local['low'].iloc[i]
            mid = (fvg_low + fvg_high) / 2
            dist_pct = abs(price - mid) / price * 100
            
            if dist_pct < proximity_pct:
                fvgs.append(f"Near Bullish FVG {dist_pct:.1f}% away ({tf})")
        
        # Bearish FVG
        elif df_local['low'].iloc[i-2] > df_local['high'].iloc[i]:
            fvg_low = df_local['high'].iloc[i]
            fvg_high = df_local['low'].iloc[i-2]
            mid = (fvg_low + fvg_high) / 2
            dist_pct = abs(price - mid) / price * 100
            
            if dist_pct < proximity_pct:
                fvgs.append(f"Near Bearish FVG {dist_pct:.1f}% away ({tf})")
    
    return fvgs

# ============================================================================
# PREMIUM/DISCOUNT ZONES
# ============================================================================
def calculate_premium_discount(df: pd.DataFrame, lookback: int = ZONE_LOOKBACK) -> pd.DataFrame:
    """Determine if price is in premium/discount zone relative to range."""
    if len(df) < lookback:
        df = df.copy()
        df['premium_pct'] = 0.5
        df['zone'] = 'equilibrium'
        return df
    
    df = df.copy()
    
    df['range_high'] = df['high'].rolling(lookback, min_periods=1).max()
    df['range_low'] = df['low'].rolling(lookback, min_periods=1).min()
    df['range_mid'] = (df['range_high'] + df['range_low']) / 2
    
    range_size = df['range_high'] - df['range_low']
    df['premium_pct'] = np.where(
        range_size > 0,
        (df['close'] - df['range_low']) / range_size,
        0.5
    )
    
    df['premium_pct'] = df['premium_pct'].clip(0, 1)
    
    df['zone'] = 'equilibrium'
    df.loc[df['premium_pct'] <= 0.3, 'zone'] = 'discount'
    df.loc[df['premium_pct'] >= 0.7, 'zone'] = 'premium'
    
    return df

# ============================================================================
# MARKET REGIME DETECTION (v27.8.2 FIXED)
# ============================================================================
def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Classify market into trending/ranging/explosive/dead regimes.
    v27.8.2: RELAXED thresholds to prevent blocking valid 4h signals
    """
    if len(df) < 50:
        return 'ranging'
    
    adx_result = ta.adx(df['high'], df['low'], df['close'], 14)
    if adx_result is None or 'ADX_14' not in adx_result.columns:
        return 'ranging'
    
    adx = adx_result['ADX_14'].iloc[-1]
    if pd.isna(adx):
        return 'ranging'
    
    df_copy = df.copy()
    if 'atr' not in df_copy.columns:
        df_copy['atr'] = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], 14)
    
    atr_current = df_copy['atr'].iloc[-1]
    atr_mean = df_copy['atr'].rolling(50, min_periods=1).mean().iloc[-1]
    
    if pd.isna(atr_current) or pd.isna(atr_mean) or atr_mean == 0:
        atr_ratio = 1.0
    else:
        atr_ratio = atr_current / atr_mean
    
    # RELAXED thresholds - don't kill 4h signals (v27.8.2)
    if adx > 40 and atr_ratio > 1.5:
        return 'explosive'
    elif adx > 20:  # Was 25 - lowered for more signals
        return 'trending'
    elif adx < 15 and atr_ratio < 0.6:  # Was 20 and 0.8 - much stricter now
        return 'dead'
    else:
        return 'ranging'

def is_consolidation(df: pd.DataFrame) -> bool:
    """Check if market is consolidating (skip trading)."""
    regime = detect_market_regime(df)
    
    if regime == 'dead':
        return True
    
    if len(df) < 14:
        return True
    
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is None or 'ADX_14' not in adx.columns:
        return False
    
    adx_val = adx['ADX_14'].iloc[-1]
    if pd.isna(adx_val):
        return False
    
    vol_mean = df['volume'].rolling(20, min_periods=1).mean().iloc[-1]
    vol_ratio = df['volume'].iloc[-1] / vol_mean if vol_mean > 0 else 1.0
    
    consol = (adx_val < 25) and (vol_ratio < VOL_SURGE_MULTIPLIER)
    
    if regime == 'ranging':
        consol = False
    
    logging.debug(f"Regime: {regime} | ADX={adx_val:.1f}, vol_ratio={vol_ratio:.2f} -> consol={consol}")
    return consol

# ============================================================================
# PATTERN DETECTION
# ============================================================================
def detect_candle_patterns(df: pd.DataFrame, tf: str) -> List[str]:
    """Detect institutional candle patterns."""
    if len(df) < 3:
        return []
    
    patterns = set()
    c, p = df.iloc[-1], df.iloc[-2]
    
    body = abs(c['close'] - c['open'])
    upper = c['high'] - max(c['open'], c['close'])
    lower = min(c['open'], c['close']) - c['low']
    range_size = c['high'] - c['low']
    
    if range_size == 0:
        return []
    
    vol_mean = df['volume'].rolling(20, min_periods=1).mean().iloc[-1]
    
    # Displacement candle
    if body > DISPLACEMENT_BODY_RATIO * range_size and vol_mean > 0 and df['volume'].iloc[-1] > VOL_SURGE_MULTIPLIER * vol_mean:
        dir_str = "Bullish" if c['close'] > c['open'] else "Bearish"
        patterns.add(f"{dir_str} Displacement Candle ({tf})")
    
    # Pinbars
    if body > 0 and lower > PINBAR_WICK_RATIO * body and upper < body * 0.3 and c['close'] > p['close']:
        patterns.add(f"Bullish Pinbar ({tf})")
    
    if body > 0 and upper > PINBAR_WICK_RATIO * body and lower < body * 0.3 and c['close'] < p['close']:
        patterns.add(f"Bearish Pinbar ({tf})")
    
    # Engulfing patterns
    if p['close'] < p['open'] and c['close'] > c['open'] and c['open'] < p['close'] and c['close'] > p['open']:
        patterns.add(f"Bullish Engulfing ({tf})")
    
    if p['close'] > p['open'] and c['close'] < c['open'] and c['open'] > p['close'] and c['close'] < p['open']:
        patterns.add(f"Bearish Engulfing ({tf})")
    
    # Inside bar
    if c['high'] < p['high'] and c['low'] > p['low']:
        patterns.add(f"Inside Bar ({tf})")
    
    return list(patterns)

# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================
def detect_divergence(df: pd.DataFrame, tf: str) -> Optional[str]:
    """Detect RSI divergences."""
    if len(df) < 50 or tf not in ['4h', '1d']:
        return None
    
    if 'rsi' not in df.columns or pd.isna(df['rsi'].iloc[-1]):
        return None
    
    price = df['close'].iloc[-40:]
    rsi = df['rsi'].iloc[-40:]
    
    swing_low_idx = price.iloc[-10:].idxmin()
    swing_high_idx = price.iloc[-10:].idxmax()
    
    if pd.isna(rsi.loc[swing_low_idx]) or pd.isna(rsi.loc[swing_high_idx]):
        return None
    
    # Hidden bullish divergence
    if price.iloc[-1] > price.loc[swing_low_idx] and rsi.iloc[-1] < rsi.loc[swing_low_idx] and rsi.loc[swing_low_idx] < 30:
        return f"Hidden Bullish RSI Divergence ({tf})"
    
    # Hidden bearish divergence
    if price.iloc[-1] < price.loc[swing_high_idx] and rsi.iloc[-1] > rsi.loc[swing_high_idx] and rsi.loc[swing_high_idx] > 70:
        return f"Hidden Bearish RSI Divergence ({tf})"
    
    # Regular bullish divergence
    if price.iloc[-1] < price.loc[swing_low_idx] and rsi.iloc[-1] > rsi.loc[swing_low_idx] and rsi.iloc[-1] < 25:
        return f"Bullish RSI Divergence ({tf})"
    
    # Regular bearish divergence
    if price.iloc[-1] > price.loc[swing_high_idx] and rsi.iloc[-1] < rsi.loc[swing_high_idx] and rsi.iloc[-1] > 75:
        return f"Bearish RSI Divergence ({tf})"
    
    return None

# ============================================================================
# EMA CROSS DETECTION
# ============================================================================
def detect_pre_cross(df: pd.DataFrame, tf: str) -> Optional[str]:
    """Detect when price is approaching EMA200."""
    if len(df) < 2 or tf not in ['4h', '1d']:
        return None
    
    if 'ema200' not in df.columns or pd.isna(df['ema200'].iloc[-1]):
        return None
    
    l = df.iloc[-1]
    if l['close'] == 0:
        return None
    
    diff = abs(l['ema200'] - l['close']) / l['close']
    
    if diff < PRE_CROSS_THRESHOLD_PCT:
        return "EMA200 Bias Shift Incoming"
    
    return None

# ============================================================================
# LIQUIDITY PROFILE
# ============================================================================
def calc_liquidity_profile(df: pd.DataFrame, bins: int = 15) -> dict:
    """Calculate volume profile to identify high-liquidity zones (POC)."""
    if len(df) < 50:
        return {}
    
    prices = df['close']
    volumes = df['volume']
    
    min_p, max_p = prices.min(), prices.max()
    
    if min_p == max_p:
        return {}
    
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

# ============================================================================
# VOLATILITY CHECK
# ============================================================================
def get_current_volatility(df: pd.DataFrame) -> float:
    """
    Get current volatility as ATR percentage of price.
    """
    if len(df) < 14 or 'atr' not in df.columns:
        return 2.0
    
    atr = df['atr'].iloc[-1]
    price = df['close'].iloc[-1]
    
    if pd.isna(atr) or price == 0:
        return 2.0
    
    return (atr / price) * 100
