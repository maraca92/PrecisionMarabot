# early_reversal.py - Grok Elite Signal Bot v27.12.14 - Early Reversal Detection
# -*- coding: utf-8 -*-
"""
Early Reversal Detection System

Detects potential reversals BEFORE they happen using:
1. Momentum Divergences (RSI, MACD) - Earliest signal
2. Candlestick Reversal Patterns - Confirmation
3. Classical Chart Patterns - Strong confirmation
4. Volume Analysis - Validation

Based on the "Compact Guide: Early Reversal Detection" methodology.

v27.12.14: Initial implementation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DivergenceSignal:
    """Represents a detected divergence."""
    divergence_type: str  # 'regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish'
    indicator: str  # 'RSI', 'MACD'
    strength: float  # 0-100
    price_swing1: float
    price_swing2: float
    indicator_swing1: float
    indicator_swing2: float
    timeframe: str
    description: str


@dataclass
class CandlestickPattern:
    """Represents a detected candlestick pattern."""
    pattern_name: str
    pattern_type: str  # 'bullish_reversal', 'bearish_reversal', 'continuation'
    strength: float  # 0-100
    volume_confirmed: bool
    at_key_level: bool  # Near support/resistance
    index: int  # Candle index
    description: str


@dataclass
class ChartPattern:
    """Represents a classical chart pattern."""
    pattern_name: str  # 'head_shoulders', 'inverse_head_shoulders', 'double_bottom', etc.
    pattern_type: str  # 'bullish_reversal', 'bearish_reversal'
    completion_pct: float  # How complete is the pattern (0-100)
    neckline: float
    target: float
    invalidation: float
    description: str


@dataclass
class ReversalSignal:
    """Complete reversal signal with all components."""
    direction: str  # 'LONG', 'SHORT'
    confidence: float  # 0-100
    divergences: List[DivergenceSignal] = field(default_factory=list)
    candlestick_patterns: List[CandlestickPattern] = field(default_factory=list)
    chart_patterns: List[ChartPattern] = field(default_factory=list)
    volume_confirmed: bool = False
    rsi_extreme: bool = False
    confluence_count: int = 0
    summary: str = ""


# ============================================================================
# 1. MOMENTUM DIVERGENCE DETECTION (Earliest Signal)
# ============================================================================

def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = 30,
    swing_lookback: int = 5
) -> Optional[DivergenceSignal]:
    """
    Detect RSI divergences - the earliest reversal signal.
    
    Regular Bullish: Price lower lows, RSI higher lows (at oversold)
    Regular Bearish: Price higher highs, RSI lower highs (at overbought)
    Hidden Bullish: Price higher lows, RSI lower lows (trend continuation)
    Hidden Bearish: Price lower highs, RSI higher highs (trend continuation)
    
    Prioritizes REGULAR divergences over hidden ones per the guide.
    """
    if df is None or len(df) < lookback + rsi_period:
        return None
    
    try:
        # Calculate RSI if not present
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df = df.copy()
            df['rsi'] = 100 - (100 / (1 + rs))
        
        price = df['close'].iloc[-lookback:].values
        rsi = df['rsi'].iloc[-lookback:].values
        low = df['low'].iloc[-lookback:].values
        high = df['high'].iloc[-lookback:].values
        
        # Find swing points
        swing_lows = _find_swing_lows(low, swing_lookback)
        swing_highs = _find_swing_highs(high, swing_lookback)
        
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            return None
        
        # Get the two most recent swing points
        recent_low_indices = swing_lows[-2:]
        recent_high_indices = swing_highs[-2:]
        
        current_rsi = rsi[-1]
        
        # === REGULAR BULLISH DIVERGENCE (Priority) ===
        # Price makes lower low, RSI makes higher low
        if len(recent_low_indices) >= 2:
            idx1, idx2 = recent_low_indices[-2], recent_low_indices[-1]
            if idx1 < len(price) and idx2 < len(price) and idx1 < len(rsi) and idx2 < len(rsi):
                price_ll = low[idx2] < low[idx1]  # Lower low in price
                rsi_hl = rsi[idx2] > rsi[idx1]     # Higher low in RSI
                
                if price_ll and rsi_hl and current_rsi < 35:  # Near oversold
                    strength = min(100, 60 + (35 - current_rsi) * 2)
                    return DivergenceSignal(
                        divergence_type='regular_bullish',
                        indicator='RSI',
                        strength=strength,
                        price_swing1=low[idx1],
                        price_swing2=low[idx2],
                        indicator_swing1=rsi[idx1],
                        indicator_swing2=rsi[idx2],
                        timeframe='',
                        description=f"🟢 Regular Bullish Divergence: Price LL, RSI HL (RSI={current_rsi:.0f})"
                    )
        
        # === REGULAR BEARISH DIVERGENCE (Priority) ===
        # Price makes higher high, RSI makes lower high
        if len(recent_high_indices) >= 2:
            idx1, idx2 = recent_high_indices[-2], recent_high_indices[-1]
            if idx1 < len(price) and idx2 < len(price) and idx1 < len(rsi) and idx2 < len(rsi):
                price_hh = high[idx2] > high[idx1]  # Higher high in price
                rsi_lh = rsi[idx2] < rsi[idx1]       # Lower high in RSI
                
                if price_hh and rsi_lh and current_rsi > 65:  # Near overbought
                    strength = min(100, 60 + (current_rsi - 65) * 2)
                    return DivergenceSignal(
                        divergence_type='regular_bearish',
                        indicator='RSI',
                        strength=strength,
                        price_swing1=high[idx1],
                        price_swing2=high[idx2],
                        indicator_swing1=rsi[idx1],
                        indicator_swing2=rsi[idx2],
                        timeframe='',
                        description=f"🔴 Regular Bearish Divergence: Price HH, RSI LH (RSI={current_rsi:.0f})"
                    )
        
        # === HIDDEN DIVERGENCES (Lower priority) ===
        # Hidden Bullish: Price higher low, RSI lower low (continuation in uptrend)
        if len(recent_low_indices) >= 2:
            idx1, idx2 = recent_low_indices[-2], recent_low_indices[-1]
            if idx1 < len(price) and idx2 < len(price):
                price_hl = low[idx2] > low[idx1]
                rsi_ll = rsi[idx2] < rsi[idx1]
                
                if price_hl and rsi_ll and current_rsi < 45:
                    return DivergenceSignal(
                        divergence_type='hidden_bullish',
                        indicator='RSI',
                        strength=55,
                        price_swing1=low[idx1],
                        price_swing2=low[idx2],
                        indicator_swing1=rsi[idx1],
                        indicator_swing2=rsi[idx2],
                        timeframe='',
                        description=f"🟡 Hidden Bullish Divergence: Price HL, RSI LL"
                    )
        
        # Hidden Bearish: Price lower high, RSI higher high (continuation in downtrend)
        if len(recent_high_indices) >= 2:
            idx1, idx2 = recent_high_indices[-2], recent_high_indices[-1]
            if idx1 < len(price) and idx2 < len(price):
                price_lh = high[idx2] < high[idx1]
                rsi_hh = rsi[idx2] > rsi[idx1]
                
                if price_lh and rsi_hh and current_rsi > 55:
                    return DivergenceSignal(
                        divergence_type='hidden_bearish',
                        indicator='RSI',
                        strength=55,
                        price_swing1=high[idx1],
                        price_swing2=high[idx2],
                        indicator_swing1=rsi[idx1],
                        indicator_swing2=rsi[idx2],
                        timeframe='',
                        description=f"🟡 Hidden Bearish Divergence: Price LH, RSI HH"
                    )
        
        return None
        
    except Exception as e:
        logging.debug(f"RSI divergence detection error: {e}")
        return None


def detect_macd_divergence(
    df: pd.DataFrame,
    lookback: int = 30,
    swing_lookback: int = 5
) -> Optional[DivergenceSignal]:
    """
    Detect MACD divergences.
    
    Uses MACD histogram for divergence detection.
    """
    if df is None or len(df) < lookback + 26:
        return None
    
    try:
        # Calculate MACD if not present
        if 'macd_hist' not in df.columns:
            import pandas_ta as ta
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is None or len(macd_result.columns) < 3:
                return None
            df = df.copy()
            df['macd_hist'] = macd_result.iloc[:, 2]
        
        price = df['close'].iloc[-lookback:].values
        macd = df['macd_hist'].iloc[-lookback:].values
        low = df['low'].iloc[-lookback:].values
        high = df['high'].iloc[-lookback:].values
        
        # Find swing points
        swing_lows = _find_swing_lows(low, swing_lookback)
        swing_highs = _find_swing_highs(high, swing_lookback)
        
        if len(swing_lows) < 2 or len(swing_highs) < 2:
            return None
        
        # Regular Bullish: Price LL, MACD HL
        recent_low_indices = swing_lows[-2:]
        idx1, idx2 = recent_low_indices[-2], recent_low_indices[-1]
        
        if idx1 < len(macd) and idx2 < len(macd):
            price_ll = low[idx2] < low[idx1]
            macd_hl = macd[idx2] > macd[idx1]
            
            if price_ll and macd_hl and macd[-1] < 0:
                return DivergenceSignal(
                    divergence_type='regular_bullish',
                    indicator='MACD',
                    strength=65,
                    price_swing1=low[idx1],
                    price_swing2=low[idx2],
                    indicator_swing1=macd[idx1],
                    indicator_swing2=macd[idx2],
                    timeframe='',
                    description="🟢 MACD Bullish Divergence: Price LL, MACD HL"
                )
        
        # Regular Bearish: Price HH, MACD LH
        recent_high_indices = swing_highs[-2:]
        idx1, idx2 = recent_high_indices[-2], recent_high_indices[-1]
        
        if idx1 < len(macd) and idx2 < len(macd):
            price_hh = high[idx2] > high[idx1]
            macd_lh = macd[idx2] < macd[idx1]
            
            if price_hh and macd_lh and macd[-1] > 0:
                return DivergenceSignal(
                    divergence_type='regular_bearish',
                    indicator='MACD',
                    strength=65,
                    price_swing1=high[idx1],
                    price_swing2=high[idx2],
                    indicator_swing1=macd[idx1],
                    indicator_swing2=macd[idx2],
                    timeframe='',
                    description="🔴 MACD Bearish Divergence: Price HH, MACD LH"
                )
        
        return None
        
    except Exception as e:
        logging.debug(f"MACD divergence detection error: {e}")
        return None


# ============================================================================
# 2. CANDLESTICK REVERSAL PATTERNS
# ============================================================================

def detect_candlestick_patterns(
    df: pd.DataFrame,
    avg_volume_period: int = 20
) -> List[CandlestickPattern]:
    """
    Detect reversal candlestick patterns.
    
    Bullish: Hammer, Bullish Engulfing, Morning Star
    Bearish: Shooting Star, Bearish Engulfing, Evening Star
    """
    patterns = []
    
    if df is None or len(df) < avg_volume_period + 3:
        return patterns
    
    try:
        # Get recent candles
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        v = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        
        # Average volume for confirmation
        avg_vol = np.mean(v[-avg_volume_period:-1])
        current_vol = v[-1]
        volume_spike = current_vol > avg_vol * 1.5
        
        # Current and previous candles
        i = -1  # Current candle
        
        body = abs(c[i] - o[i])
        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - l[i]
        candle_range = h[i] - l[i]
        
        if candle_range == 0:
            return patterns
        
        is_bullish = c[i] > o[i]
        is_bearish = c[i] < o[i]
        
        # === HAMMER (Bullish) ===
        # Small body at top, long lower wick (>2x body), small upper wick
        if is_bullish and lower_wick > body * 2 and upper_wick < body * 0.5:
            strength = 70 if volume_spike else 55
            patterns.append(CandlestickPattern(
                pattern_name='Hammer',
                pattern_type='bullish_reversal',
                strength=strength,
                volume_confirmed=volume_spike,
                at_key_level=False,
                index=len(df) - 1,
                description="🔨 Hammer: Long lower wick, small body at top"
            ))
        
        # === INVERTED HAMMER / SHOOTING STAR (Bearish) ===
        # Small body at bottom, long upper wick, small lower wick
        if is_bearish and upper_wick > body * 2 and lower_wick < body * 0.5:
            strength = 70 if volume_spike else 55
            patterns.append(CandlestickPattern(
                pattern_name='Shooting Star',
                pattern_type='bearish_reversal',
                strength=strength,
                volume_confirmed=volume_spike,
                at_key_level=False,
                index=len(df) - 1,
                description="💫 Shooting Star: Long upper wick, small body at bottom"
            ))
        
        # === BULLISH ENGULFING ===
        if len(df) >= 2:
            prev_body = abs(c[-2] - o[-2])
            prev_bearish = c[-2] < o[-2]
            
            if prev_bearish and is_bullish:
                # Current bullish candle engulfs previous bearish
                if o[i] <= c[-2] and c[i] >= o[-2] and body > prev_body:
                    strength = 75 if volume_spike else 60
                    patterns.append(CandlestickPattern(
                        pattern_name='Bullish Engulfing',
                        pattern_type='bullish_reversal',
                        strength=strength,
                        volume_confirmed=volume_spike,
                        at_key_level=False,
                        index=len(df) - 1,
                        description="🟢 Bullish Engulfing: Bullish candle engulfs previous bearish"
                    ))
        
        # === BEARISH ENGULFING ===
        if len(df) >= 2:
            prev_body = abs(c[-2] - o[-2])
            prev_bullish = c[-2] > o[-2]
            
            if prev_bullish and is_bearish:
                # Current bearish candle engulfs previous bullish
                if o[i] >= c[-2] and c[i] <= o[-2] and body > prev_body:
                    strength = 75 if volume_spike else 60
                    patterns.append(CandlestickPattern(
                        pattern_name='Bearish Engulfing',
                        pattern_type='bearish_reversal',
                        strength=strength,
                        volume_confirmed=volume_spike,
                        at_key_level=False,
                        index=len(df) - 1,
                        description="🔴 Bearish Engulfing: Bearish candle engulfs previous bullish"
                    ))
        
        # === MORNING STAR (3-candle bullish reversal) ===
        if len(df) >= 3:
            # Day 1: Large bearish candle
            day1_bearish = c[-3] < o[-3] and abs(c[-3] - o[-3]) > candle_range * 0.5
            # Day 2: Small body (doji-like), gaps down
            day2_small = abs(c[-2] - o[-2]) < candle_range * 0.3
            day2_gap = max(o[-2], c[-2]) < c[-3]
            # Day 3: Large bullish candle closing above day 1 midpoint
            day3_bullish = is_bullish and c[i] > (o[-3] + c[-3]) / 2
            
            if day1_bearish and day2_small and day3_bullish:
                strength = 80 if volume_spike else 65
                patterns.append(CandlestickPattern(
                    pattern_name='Morning Star',
                    pattern_type='bullish_reversal',
                    strength=strength,
                    volume_confirmed=volume_spike,
                    at_key_level=False,
                    index=len(df) - 1,
                    description="⭐ Morning Star: 3-candle bullish reversal pattern"
                ))
        
        # === EVENING STAR (3-candle bearish reversal) ===
        if len(df) >= 3:
            # Day 1: Large bullish candle
            day1_bullish = c[-3] > o[-3] and abs(c[-3] - o[-3]) > candle_range * 0.5
            # Day 2: Small body, gaps up
            day2_small = abs(c[-2] - o[-2]) < candle_range * 0.3
            day2_gap = min(o[-2], c[-2]) > c[-3]
            # Day 3: Large bearish candle closing below day 1 midpoint
            day3_bearish = is_bearish and c[i] < (o[-3] + c[-3]) / 2
            
            if day1_bullish and day2_small and day3_bearish:
                strength = 80 if volume_spike else 65
                patterns.append(CandlestickPattern(
                    pattern_name='Evening Star',
                    pattern_type='bearish_reversal',
                    strength=strength,
                    volume_confirmed=volume_spike,
                    at_key_level=False,
                    index=len(df) - 1,
                    description="🌙 Evening Star: 3-candle bearish reversal pattern"
                ))
        
        # === DOJI (Indecision - needs context) ===
        if body < candle_range * 0.1:  # Very small body
            patterns.append(CandlestickPattern(
                pattern_name='Doji',
                pattern_type='indecision',
                strength=40,
                volume_confirmed=volume_spike,
                at_key_level=False,
                index=len(df) - 1,
                description="⚖️ Doji: Market indecision, potential reversal"
            ))
        
        return patterns
        
    except Exception as e:
        logging.debug(f"Candlestick pattern detection error: {e}")
        return patterns


# ============================================================================
# 3. CLASSICAL CHART PATTERNS
# ============================================================================

def detect_double_bottom(
    df: pd.DataFrame,
    tolerance_pct: float = 2.0,
    lookback: int = 50
) -> Optional[ChartPattern]:
    """
    Detect double bottom pattern (bullish reversal).
    
    Two lows at approximately the same level with a peak between them.
    """
    if df is None or len(df) < lookback:
        return None
    
    try:
        low = df['low'].iloc[-lookback:].values
        high = df['high'].iloc[-lookback:].values
        close = df['close'].iloc[-lookback:].values
        
        # Find swing lows
        swing_lows = _find_swing_lows(low, 5)
        
        if len(swing_lows) < 2:
            return None
        
        # Get two most recent lows
        low1_idx, low2_idx = swing_lows[-2], swing_lows[-1]
        low1_price = low[low1_idx]
        low2_price = low[low2_idx]
        
        # Check if lows are at similar level
        tolerance = low1_price * (tolerance_pct / 100)
        if abs(low1_price - low2_price) > tolerance:
            return None
        
        # Find the peak between the two lows (neckline)
        if low2_idx <= low1_idx:
            return None
        
        neckline_idx = low1_idx + np.argmax(high[low1_idx:low2_idx])
        neckline = high[neckline_idx]
        
        # Current price should be approaching or breaking neckline
        current_price = close[-1]
        
        # Pattern height for target calculation
        pattern_height = neckline - min(low1_price, low2_price)
        target = neckline + pattern_height
        invalidation = min(low1_price, low2_price) * 0.98
        
        # Calculate completion
        if current_price >= neckline:
            completion = 100
        else:
            completion = (current_price - low2_price) / (neckline - low2_price) * 100
        
        if completion > 70:  # Pattern is forming
            return ChartPattern(
                pattern_name='Double Bottom',
                pattern_type='bullish_reversal',
                completion_pct=min(100, completion),
                neckline=neckline,
                target=target,
                invalidation=invalidation,
                description=f"📈 Double Bottom forming: Neckline ${neckline:.2f}, Target ${target:.2f}"
            )
        
        return None
        
    except Exception as e:
        logging.debug(f"Double bottom detection error: {e}")
        return None


def detect_double_top(
    df: pd.DataFrame,
    tolerance_pct: float = 2.0,
    lookback: int = 50
) -> Optional[ChartPattern]:
    """
    Detect double top pattern (bearish reversal).
    
    Two highs at approximately the same level with a trough between them.
    """
    if df is None or len(df) < lookback:
        return None
    
    try:
        low = df['low'].iloc[-lookback:].values
        high = df['high'].iloc[-lookback:].values
        close = df['close'].iloc[-lookback:].values
        
        # Find swing highs
        swing_highs = _find_swing_highs(high, 5)
        
        if len(swing_highs) < 2:
            return None
        
        # Get two most recent highs
        high1_idx, high2_idx = swing_highs[-2], swing_highs[-1]
        high1_price = high[high1_idx]
        high2_price = high[high2_idx]
        
        # Check if highs are at similar level
        tolerance = high1_price * (tolerance_pct / 100)
        if abs(high1_price - high2_price) > tolerance:
            return None
        
        # Find the trough between the two highs (neckline)
        if high2_idx <= high1_idx:
            return None
        
        neckline_idx = high1_idx + np.argmin(low[high1_idx:high2_idx])
        neckline = low[neckline_idx]
        
        current_price = close[-1]
        
        # Pattern height for target calculation
        pattern_height = max(high1_price, high2_price) - neckline
        target = neckline - pattern_height
        invalidation = max(high1_price, high2_price) * 1.02
        
        # Calculate completion
        if current_price <= neckline:
            completion = 100
        else:
            completion = (high2_price - current_price) / (high2_price - neckline) * 100
        
        if completion > 70:
            return ChartPattern(
                pattern_name='Double Top',
                pattern_type='bearish_reversal',
                completion_pct=min(100, completion),
                neckline=neckline,
                target=target,
                invalidation=invalidation,
                description=f"📉 Double Top forming: Neckline ${neckline:.2f}, Target ${target:.2f}"
            )
        
        return None
        
    except Exception as e:
        logging.debug(f"Double top detection error: {e}")
        return None


def detect_head_shoulders(
    df: pd.DataFrame,
    lookback: int = 60
) -> Optional[ChartPattern]:
    """
    Detect Head and Shoulders pattern (bearish reversal).
    
    Left shoulder, head (higher), right shoulder at similar level to left.
    """
    if df is None or len(df) < lookback:
        return None
    
    try:
        high = df['high'].iloc[-lookback:].values
        low = df['low'].iloc[-lookback:].values
        close = df['close'].iloc[-lookback:].values
        
        # Find swing highs (need at least 3)
        swing_highs = _find_swing_highs(high, 5)
        
        if len(swing_highs) < 3:
            return None
        
        # Get three most recent highs
        ls_idx, head_idx, rs_idx = swing_highs[-3], swing_highs[-2], swing_highs[-1]
        
        left_shoulder = high[ls_idx]
        head = high[head_idx]
        right_shoulder = high[rs_idx]
        
        # Head must be highest
        if not (head > left_shoulder and head > right_shoulder):
            return None
        
        # Shoulders should be at similar level (within 5%)
        shoulder_tolerance = left_shoulder * 0.05
        if abs(left_shoulder - right_shoulder) > shoulder_tolerance:
            return None
        
        # Find neckline (lows between shoulders and head)
        neckline1_idx = ls_idx + np.argmin(low[ls_idx:head_idx])
        neckline2_idx = head_idx + np.argmin(low[head_idx:rs_idx])
        neckline = (low[neckline1_idx] + low[neckline2_idx]) / 2
        
        current_price = close[-1]
        
        # Pattern height for target
        pattern_height = head - neckline
        target = neckline - pattern_height
        invalidation = head * 1.02
        
        # Completion (price approaching neckline)
        if current_price <= neckline:
            completion = 100
        else:
            completion = (right_shoulder - current_price) / (right_shoulder - neckline) * 100
        
        if completion > 60:
            return ChartPattern(
                pattern_name='Head and Shoulders',
                pattern_type='bearish_reversal',
                completion_pct=min(100, completion),
                neckline=neckline,
                target=target,
                invalidation=invalidation,
                description=f"👤 Head & Shoulders: Neckline ${neckline:.2f}, Target ${target:.2f}"
            )
        
        return None
        
    except Exception as e:
        logging.debug(f"Head & Shoulders detection error: {e}")
        return None


# ============================================================================
# 4. MAIN REVERSAL DETECTION FUNCTION
# ============================================================================

def detect_early_reversal(
    df: pd.DataFrame,
    timeframe: str = '4h'
) -> Optional[ReversalSignal]:
    """
    Main function to detect early reversal signals.
    
    Combines all detection methods and requires confluence (≥2 signals).
    
    Args:
        df: OHLCV DataFrame
        timeframe: Timeframe string for logging
    
    Returns:
        ReversalSignal if reversal detected with confluence, None otherwise
    """
    if df is None or len(df) < 50:
        return None
    
    divergences = []
    candlestick_patterns = []
    chart_patterns = []
    
    # 1. Detect divergences (earliest signals)
    rsi_div = detect_rsi_divergence(df)
    if rsi_div:
        rsi_div.timeframe = timeframe
        divergences.append(rsi_div)
    
    macd_div = detect_macd_divergence(df)
    if macd_div:
        macd_div.timeframe = timeframe
        divergences.append(macd_div)
    
    # 2. Detect candlestick patterns
    candle_patterns = detect_candlestick_patterns(df)
    candlestick_patterns.extend(candle_patterns)
    
    # 3. Detect chart patterns
    double_bottom = detect_double_bottom(df)
    if double_bottom:
        chart_patterns.append(double_bottom)
    
    double_top = detect_double_top(df)
    if double_top:
        chart_patterns.append(double_top)
    
    head_shoulders = detect_head_shoulders(df)
    if head_shoulders:
        chart_patterns.append(head_shoulders)
    
    # 4. Check RSI extremes
    rsi_extreme = False
    if 'rsi' in df.columns and len(df) > 0:
        current_rsi = df['rsi'].iloc[-1]
        if not pd.isna(current_rsi):
            rsi_extreme = current_rsi < 30 or current_rsi > 70
    
    # 5. Check volume confirmation
    volume_confirmed = False
    if 'volume' in df.columns and len(df) >= 21:
        avg_vol = df['volume'].iloc[-20:-1].mean()
        current_vol = df['volume'].iloc[-1]
        if avg_vol > 0 and not pd.isna(current_vol):
            volume_confirmed = current_vol > avg_vol * 1.5
    
    # 6. Determine direction and count confluence
    bullish_signals = 0
    bearish_signals = 0
    
    for div in divergences:
        if 'bullish' in div.divergence_type:
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    for pattern in candlestick_patterns:
        if pattern.pattern_type == 'bullish_reversal':
            bullish_signals += 1
        elif pattern.pattern_type == 'bearish_reversal':
            bearish_signals += 1
    
    for pattern in chart_patterns:
        if pattern.pattern_type == 'bullish_reversal':
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    # Require confluence (≥2 signals)
    total_signals = max(bullish_signals, bearish_signals)
    
    if total_signals < 2:
        return None
    
    # Determine primary direction
    if bullish_signals > bearish_signals:
        direction = 'LONG'
        confluence_count = bullish_signals
        # Filter to only bullish signals
        divergences = [d for d in divergences if 'bullish' in d.divergence_type]
        candlestick_patterns = [p for p in candlestick_patterns if p.pattern_type == 'bullish_reversal']
        chart_patterns = [p for p in chart_patterns if p.pattern_type == 'bullish_reversal']
    else:
        direction = 'SHORT'
        confluence_count = bearish_signals
        # Filter to only bearish signals
        divergences = [d for d in divergences if 'bearish' in d.divergence_type]
        candlestick_patterns = [p for p in candlestick_patterns if p.pattern_type == 'bearish_reversal']
        chart_patterns = [p for p in chart_patterns if p.pattern_type == 'bearish_reversal']
    
    # Calculate confidence
    base_confidence = 50
    
    # Add for each confluence factor
    for div in divergences:
        base_confidence += 15 if 'regular' in div.divergence_type else 8
    
    for pattern in candlestick_patterns:
        base_confidence += 10 if pattern.volume_confirmed else 5
    
    for pattern in chart_patterns:
        base_confidence += 12
    
    if volume_confirmed:
        base_confidence += 8
    
    if rsi_extreme:
        base_confidence += 5
    
    confidence = min(95, base_confidence)
    
    # Build summary
    summary_parts = []
    if divergences:
        summary_parts.append(f"Divergence: {divergences[0].indicator}")
    if candlestick_patterns:
        summary_parts.append(f"Candle: {candlestick_patterns[0].pattern_name}")
    if chart_patterns:
        summary_parts.append(f"Pattern: {chart_patterns[0].pattern_name}")
    
    summary = f"🔄 Early {direction} Reversal ({timeframe}): " + ", ".join(summary_parts)
    
    return ReversalSignal(
        direction=direction,
        confidence=confidence,
        divergences=divergences,
        candlestick_patterns=candlestick_patterns,
        chart_patterns=chart_patterns,
        volume_confirmed=volume_confirmed,
        rsi_extreme=rsi_extreme,
        confluence_count=confluence_count,
        summary=summary
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_swing_lows(data: np.ndarray, lookback: int = 5) -> List[int]:
    """Find indices of swing lows."""
    swing_lows = []
    for i in range(lookback, len(data) - lookback):
        if all(data[i] <= data[i - j] for j in range(1, lookback + 1)) and \
           all(data[i] <= data[i + j] for j in range(1, lookback + 1)):
            swing_lows.append(i)
    return swing_lows


def _find_swing_highs(data: np.ndarray, lookback: int = 5) -> List[int]:
    """Find indices of swing highs."""
    swing_highs = []
    for i in range(lookback, len(data) - lookback):
        if all(data[i] >= data[i - j] for j in range(1, lookback + 1)) and \
           all(data[i] >= data[i + j] for j in range(1, lookback + 1)):
            swing_highs.append(i)
    return swing_highs


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def get_reversal_confluence_factors(reversal: ReversalSignal) -> List[str]:
    """
    Extract confluence factors from reversal signal for signal evaluation.
    
    Returns list of factor strings to add to signal confluence.
    """
    factors = []
    
    for div in reversal.divergences:
        if 'regular_bullish' in div.divergence_type:
            factors.append(f"RSI Bullish Divergence")
        elif 'regular_bearish' in div.divergence_type:
            factors.append(f"RSI Bearish Divergence")
        elif 'hidden_bullish' in div.divergence_type:
            factors.append(f"Hidden Bullish Divergence")
        elif 'hidden_bearish' in div.divergence_type:
            factors.append(f"Hidden Bearish Divergence")
    
    for pattern in reversal.candlestick_patterns:
        factors.append(pattern.pattern_name)
    
    for pattern in reversal.chart_patterns:
        factors.append(pattern.pattern_name)
    
    if reversal.volume_confirmed:
        factors.append("Volume Spike")
    
    if reversal.rsi_extreme:
        factors.append("RSI Extreme")
    
    return factors
