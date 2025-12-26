# structure_breaker.py - Grok Elite Signal Bot v27.9.5 - Structure Break Detection
"""
Detects Break of Structure (BOS) and Change of Character (CHoCH).
Also identifies manipulation patterns (stop hunts, fake breakouts).

When CHoCH detected → Triggers immediate signal check (high probability reversal)
When BOS detected → Confirms trend continuation

v27.9.5: Structure-based confluence + manipulation detection
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class StructureBreak:
    """Structure break information"""
    type: str  # 'BOS' or 'CHoCH'
    direction: str  # 'bullish' or 'bearish'
    break_level: float
    prior_trend: str  # 'uptrend', 'downtrend', 'ranging'
    strength: str  # 'high', 'medium', 'low'
    signal: str  # 'LONG' or 'SHORT'
    reason: str
    timestamp: Optional[datetime] = None
    
    def __str__(self):
        return f"{self.type} {self.direction} @ {self.break_level:.2f}"


@dataclass
class SwingPoint:
    """Swing high or swing low point"""
    index: int
    price: float
    type: str  # 'high' or 'low'
    timestamp: Optional[datetime] = None


# ============================================================================
# SWING POINT DETECTION
# ============================================================================

def find_swing_points(
    df: pd.DataFrame, 
    lookback: int = 5,
    min_swing_size_pct: float = 0.3
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Find swing highs and swing lows in price data.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Bars on each side to confirm swing
        min_swing_size_pct: Minimum swing size as % of price
    
    Returns:
        Tuple of (swing_highs, swing_lows)
    """
    swing_highs = []
    swing_lows = []
    
    if len(df) < lookback * 2 + 1:
        return [], []
    
    for i in range(lookback, len(df) - lookback):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        
        # Check for swing high
        is_swing_high = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and df['high'].iloc[j] >= current_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            # Check minimum size
            local_low = df['low'].iloc[max(0, i-lookback):i].min()
            swing_size_pct = (current_high - local_low) / current_close * 100
            
            if swing_size_pct >= min_swing_size_pct:
                swing_highs.append(SwingPoint(
                    index=i,
                    price=current_high,
                    type='high',
                    timestamp=df['date'].iloc[i] if 'date' in df.columns else None
                ))
        
        # Check for swing low
        is_swing_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and df['low'].iloc[j] <= current_low:
                is_swing_low = False
                break
        
        if is_swing_low:
            # Check minimum size
            local_high = df['high'].iloc[max(0, i-lookback):i].max()
            swing_size_pct = (local_high - current_low) / current_close * 100
            
            if swing_size_pct >= min_swing_size_pct:
                swing_lows.append(SwingPoint(
                    index=i,
                    price=current_low,
                    type='low',
                    timestamp=df['date'].iloc[i] if 'date' in df.columns else None
                ))
    
    return swing_highs, swing_lows


def determine_trend_from_swings(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint]
) -> str:
    """
    Determine trend based on swing structure.
    
    Returns:
        'uptrend', 'downtrend', or 'ranging'
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'ranging'
    
    # Get last two swings of each type
    last_high = swing_highs[-1].price
    prev_high = swing_highs[-2].price
    last_low = swing_lows[-1].price
    prev_low = swing_lows[-2].price
    
    # Higher highs + higher lows = uptrend
    if last_high > prev_high and last_low > prev_low:
        return 'uptrend'
    
    # Lower highs + lower lows = downtrend
    elif last_high < prev_high and last_low < prev_low:
        return 'downtrend'
    
    else:
        return 'ranging'


# ============================================================================
# STRUCTURE BREAK DETECTION
# ============================================================================

def detect_structure_break(
    df: pd.DataFrame,
    lookback: int = 50,
    swing_lookback: int = 5,
    confirmation_candles: int = 2
) -> Optional[StructureBreak]:
    """
    Detect Break of Structure (BOS) or Change of Character (CHoCH).
    
    BOS: Break confirming existing trend
    CHoCH: Break signaling potential reversal (HIGH VALUE!)
    
    Args:
        df: DataFrame with OHLCV data
        lookback: How far back to analyze
        swing_lookback: Bars for swing point detection
        confirmation_candles: Candles needed to confirm break
    
    Returns:
        StructureBreak if detected, None otherwise
    """
    if len(df) < lookback + 10:
        return None
    
    # Use recent data
    recent_df = df.tail(lookback).copy().reset_index(drop=True)
    
    # Find swing points
    swing_highs, swing_lows = find_swing_points(recent_df, swing_lookback)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    
    # Determine prior trend
    prior_trend = determine_trend_from_swings(swing_highs, swing_lows)
    
    # Current price and recent candles
    current_price = df['close'].iloc[-1]
    recent_high = df['high'].iloc[-confirmation_candles:].max()
    recent_low = df['low'].iloc[-confirmation_candles:].min()
    
    # Last significant swing levels
    last_swing_high = swing_highs[-1].price
    last_swing_low = swing_lows[-1].price
    
    # ========================================================================
    # CHECK FOR BULLISH BREAK
    # ========================================================================
    if recent_high > last_swing_high:
        if prior_trend == 'downtrend':
            # CHoCH - Reversal signal (HIGH VALUE)
            return StructureBreak(
                type='CHoCH',
                direction='bullish',
                break_level=last_swing_high,
                prior_trend=prior_trend,
                strength='high',
                signal='LONG',
                reason=f"Bullish CHoCH: Broke ${last_swing_high:.2f} after downtrend",
                timestamp=datetime.now(timezone.utc)
            )
        else:
            # BOS - Continuation
            return StructureBreak(
                type='BOS',
                direction='bullish',
                break_level=last_swing_high,
                prior_trend=prior_trend,
                strength='medium',
                signal='LONG',
                reason=f"Bullish BOS: Continuation above ${last_swing_high:.2f}",
                timestamp=datetime.now(timezone.utc)
            )
    
    # ========================================================================
    # CHECK FOR BEARISH BREAK
    # ========================================================================
    if recent_low < last_swing_low:
        if prior_trend == 'uptrend':
            # CHoCH - Reversal signal (HIGH VALUE)
            return StructureBreak(
                type='CHoCH',
                direction='bearish',
                break_level=last_swing_low,
                prior_trend=prior_trend,
                strength='high',
                signal='SHORT',
                reason=f"Bearish CHoCH: Broke ${last_swing_low:.2f} after uptrend",
                timestamp=datetime.now(timezone.utc)
            )
        else:
            # BOS - Continuation
            return StructureBreak(
                type='BOS',
                direction='bearish',
                break_level=last_swing_low,
                prior_trend=prior_trend,
                strength='medium',
                signal='SHORT',
                reason=f"Bearish BOS: Continuation below ${last_swing_low:.2f}",
                timestamp=datetime.now(timezone.utc)
            )
    
    return None


# ============================================================================
# MANIPULATION PATTERN DETECTION
# ============================================================================

def detect_stop_hunt(
    df: pd.DataFrame,
    lookback: int = 20
) -> Optional[Dict]:
    """
    Detect stop hunt pattern (liquidity sweep + reversal).
    
    Pattern:
    1. Price sweeps above/below recent swing
    2. Immediately reverses (wick rejection)
    3. Close back inside range
    
    Returns:
        Dict with stop hunt info or None
    """
    if len(df) < lookback + 5:
        return None
    
    recent = df.tail(lookback).copy()
    
    # Get swing levels (excluding last few bars)
    swing_high = recent['high'].iloc[:-3].max()
    swing_low = recent['low'].iloc[:-3].min()
    
    # Check last 3 candles for sweep + reversal
    for i in range(-3, 0):
        candle = df.iloc[i]
        
        # Bullish stop hunt (sweep lows, reverse up)
        if candle['low'] < swing_low * 0.998:  # Swept below
            wick_down = candle['open'] - candle['low'] if candle['close'] > candle['open'] else candle['close'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            
            if wick_down > body * 1.5 and candle['close'] > swing_low:
                return {
                    'type': 'stop_hunt',
                    'direction': 'bullish',
                    'sweep_level': candle['low'],
                    'swing_level': swing_low,
                    'signal': 'LONG',
                    'reason': f"Stop hunt below ${swing_low:.2f}, reversal candle"
                }
        
        # Bearish stop hunt (sweep highs, reverse down)
        if candle['high'] > swing_high * 1.002:  # Swept above
            wick_up = candle['high'] - candle['open'] if candle['close'] < candle['open'] else candle['high'] - candle['close']
            body = abs(candle['close'] - candle['open'])
            
            if wick_up > body * 1.5 and candle['close'] < swing_high:
                return {
                    'type': 'stop_hunt',
                    'direction': 'bearish',
                    'sweep_level': candle['high'],
                    'swing_level': swing_high,
                    'signal': 'SHORT',
                    'reason': f"Stop hunt above ${swing_high:.2f}, reversal candle"
                }
    
    return None


def detect_fake_breakout(
    df: pd.DataFrame,
    lookback: int = 30
) -> Optional[Dict]:
    """
    Detect fake breakout pattern.
    
    Pattern:
    1. Price breaks structure level
    2. Fails to hold (closes back inside within 3 candles)
    3. Signal to fade the breakout
    
    Returns:
        Dict with fake breakout info or None
    """
    if len(df) < lookback + 5:
        return None
    
    recent = df.tail(lookback).copy()
    
    # Find range before potential breakout
    range_high = recent['high'].iloc[:-5].max()
    range_low = recent['low'].iloc[:-5].min()
    
    # Check if we had a breakout that failed
    last_5 = df.tail(5)
    
    # Check for failed bullish breakout
    broke_high = last_5['high'].max() > range_high * 1.002
    closed_inside = last_5['close'].iloc[-1] < range_high
    
    if broke_high and closed_inside:
        return {
            'type': 'fake_breakout',
            'direction': 'bearish',  # Fade the failed breakout
            'break_level': range_high,
            'signal': 'SHORT',
            'reason': f"Failed breakout above ${range_high:.2f}"
        }
    
    # Check for failed bearish breakout
    broke_low = last_5['low'].min() < range_low * 0.998
    closed_inside = last_5['close'].iloc[-1] > range_low
    
    if broke_low and closed_inside:
        return {
            'type': 'fake_breakout',
            'direction': 'bullish',  # Fade the failed breakout
            'break_level': range_low,
            'signal': 'LONG',
            'reason': f"Failed breakout below ${range_low:.2f}"
        }
    
    return None


# ============================================================================
# CONFLUENCE CHECK
# ============================================================================

def get_structure_confluence(
    df: pd.DataFrame,
    proposed_direction: str,
    timeframe: str = '4h'
) -> Tuple[bool, str, Optional[StructureBreak]]:
    """
    Check if proposed trade direction aligns with structure.
    
    Args:
        df: DataFrame with OHLCV
        proposed_direction: 'Long' or 'Short'
        timeframe: Timeframe label for logging
    
    Returns:
        Tuple of (is_aligned, confluence_string, structure_break)
    """
    # Check for structure break
    structure = detect_structure_break(df)
    
    if structure:
        signal_dir = structure.signal.upper()
        proposed_dir = proposed_direction.upper()
        
        if (proposed_dir == 'LONG' and signal_dir == 'LONG') or \
           (proposed_dir == 'SHORT' and signal_dir == 'SHORT'):
            
            confluence_str = f"+{structure.type}({timeframe})"
            return True, confluence_str, structure
        else:
            return False, f"-Structure({structure.type} {structure.direction})", structure
    
    # Check for manipulation patterns
    stop_hunt = detect_stop_hunt(df)
    if stop_hunt:
        if (proposed_direction.upper() == 'LONG' and stop_hunt['signal'] == 'LONG') or \
           (proposed_direction.upper() == 'SHORT' and stop_hunt['signal'] == 'SHORT'):
            return True, f"+StopHunt({timeframe})", None
    
    fake_breakout = detect_fake_breakout(df)
    if fake_breakout:
        if (proposed_direction.upper() == 'LONG' and fake_breakout['signal'] == 'LONG') or \
           (proposed_direction.upper() == 'SHORT' and fake_breakout['signal'] == 'SHORT'):
            return True, f"+FakeBO({timeframe})", None
    
    return False, "", None


def should_trigger_immediate_check(df: pd.DataFrame) -> Tuple[bool, Optional[StructureBreak]]:
    """
    Determine if current market conditions warrant immediate signal check.
    
    Triggers on:
    - CHoCH detection (high probability reversal)
    - Strong stop hunt with reversal
    
    Returns:
        Tuple of (should_trigger, trigger_reason)
    """
    # Check for CHoCH
    structure = detect_structure_break(df)
    if structure and structure.type == 'CHoCH':
        return True, structure
    
    return False, None


# ============================================================================
# MULTI-TIMEFRAME STRUCTURE
# ============================================================================

async def check_mtf_structure(
    symbol: str,
    direction: str,
    data_cache: Dict
) -> Dict[str, Tuple[bool, str]]:
    """
    Check structure alignment across multiple timeframes.
    
    Args:
        symbol: Trading pair
        direction: Proposed direction
        data_cache: Cached OHLCV data
    
    Returns:
        Dict mapping timeframe -> (is_aligned, reason)
    """
    results = {}
    
    for tf in ['1h', '4h', '1d']:
        df = data_cache.get(symbol, {}).get(tf)
        
        if df is not None and len(df) > 50:
            aligned, reason, _ = get_structure_confluence(df, direction, tf)
            results[tf] = (aligned, reason)
        else:
            results[tf] = (False, "No data")
    
    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create test data with a clear structure break
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4h')
    
    # Simulate downtrend then reversal (CHoCH)
    prices = [100]
    for i in range(99):
        if i < 70:
            # Downtrend
            change = np.random.uniform(-1.5, 0.5)
        else:
            # Reversal
            change = np.random.uniform(-0.5, 2.0)
        prices.append(prices[-1] + change)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + np.random.uniform(0, 1) for p in prices],
        'low': [p - np.random.uniform(0, 1) for p in prices],
        'close': [p + np.random.uniform(-0.5, 0.5) for p in prices],
        'volume': [np.random.uniform(100, 1000) for _ in prices]
    })
    
    # Test structure break detection
    result = detect_structure_break(df)
    if result:
        print(f"Structure break detected: {result}")
    else:
        print("No structure break")
    
    # Test stop hunt detection
    stop_hunt = detect_stop_hunt(df)
    if stop_hunt:
        print(f"Stop hunt detected: {stop_hunt}")
    
    # Test fake breakout
    fake_bo = detect_fake_breakout(df)
    if fake_bo:
        print(f"Fake breakout detected: {fake_bo}")
