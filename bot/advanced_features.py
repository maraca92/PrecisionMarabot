# advanced_features.py - Grok Elite Signal Bot v27.12.13 - Advanced Trading Features
# -*- coding: utf-8 -*-
"""
Advanced trading features:
- Dynamic Take Profit based on volatility regime
- Multi-Timeframe (MTF) Confluence Weighting
- Integrated signal scoring enhancements

v27.12.13: Initial implementation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION IMPORTS
# ============================================================================

try:
    from bot.config import (
        DYNAMIC_TP_ENABLED, DYNAMIC_TP_HIGH_VOL, DYNAMIC_TP_MEDIUM_VOL, DYNAMIC_TP_LOW_VOL,
        MTF_WEIGHTING_ENABLED, MTF_WEIGHTS, MTF_MIN_ALIGNMENT_SCORE
    )
except ImportError:
    # Fallback defaults
    DYNAMIC_TP_ENABLED = True
    DYNAMIC_TP_HIGH_VOL = {'tp1_r': 2.0, 'tp2_r': 4.0}
    DYNAMIC_TP_MEDIUM_VOL = {'tp1_r': 1.75, 'tp2_r': 3.0}
    DYNAMIC_TP_LOW_VOL = {'tp1_r': 1.5, 'tp2_r': 2.5}
    MTF_WEIGHTING_ENABLED = True
    MTF_WEIGHTS = {'1d': 0.40, '4h': 0.35, '1h': 0.25}
    MTF_MIN_ALIGNMENT_SCORE = 0.5


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VolatilityRegime:
    """Volatility regime classification."""
    regime: str  # 'HIGH', 'MEDIUM', 'LOW'
    atr_pct: float  # ATR as percentage of price
    atr_percentile: float  # Current ATR vs historical (0-100)
    description: str


@dataclass
class DynamicTPResult:
    """Result of dynamic TP calculation."""
    tp1: float
    tp2: float
    tp1_r: float  # R-multiple
    tp2_r: float
    regime: str
    sl_distance: float


@dataclass
class MTFSignal:
    """Multi-timeframe signal data."""
    timeframe: str
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    strength: float  # 0-100
    trend: str  # 'BULLISH', 'BEARISH', 'RANGING'
    key_levels: List[float]


@dataclass
class MTFConfluence:
    """Multi-timeframe confluence result."""
    alignment_score: float  # 0-1
    primary_direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    signals: Dict[str, MTFSignal]
    confluence_strength: int  # Number of aligned timeframes
    weighted_confidence: float  # Weighted average confidence


# ============================================================================
# VOLATILITY REGIME DETECTION
# ============================================================================

def detect_volatility_regime(
    df: pd.DataFrame,
    atr_period: int = 14,
    lookback: int = 100
) -> VolatilityRegime:
    """
    Detect current volatility regime based on ATR.
    
    Args:
        df: OHLCV DataFrame
        atr_period: Period for ATR calculation
        lookback: Historical lookback for percentile
    
    Returns:
        VolatilityRegime with classification
    """
    if df is None or len(df) < atr_period + lookback:
        return VolatilityRegime(
            regime='MEDIUM',
            atr_pct=2.0,
            atr_percentile=50.0,
            description='Insufficient data - using default'
        )
    
    try:
        # Calculate ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        # Exponential moving average of TR
        atr = np.zeros(len(tr))
        atr[atr_period - 1] = np.mean(tr[:atr_period])
        
        multiplier = 2 / (atr_period + 1)
        for i in range(atr_period, len(tr)):
            atr[i] = (tr[i] - atr[i - 1]) * multiplier + atr[i - 1]
        
        current_atr = atr[-1]
        current_price = close[-1]
        atr_pct = (current_atr / current_price) * 100
        
        # Calculate percentile vs historical
        historical_atr = atr[-lookback:] if len(atr) >= lookback else atr
        atr_percentile = (np.sum(historical_atr < current_atr) / len(historical_atr)) * 100
        
        # Classify regime
        if atr_percentile >= 70:
            regime = 'HIGH'
            description = 'High volatility - wider TPs recommended'
        elif atr_percentile <= 30:
            regime = 'LOW'
            description = 'Low volatility - tighter TPs recommended'
        else:
            regime = 'MEDIUM'
            description = 'Normal volatility'
        
        return VolatilityRegime(
            regime=regime,
            atr_pct=atr_pct,
            atr_percentile=atr_percentile,
            description=description
        )
        
    except Exception as e:
        logging.warning(f"Volatility detection error: {e}")
        return VolatilityRegime(
            regime='MEDIUM',
            atr_pct=2.0,
            atr_percentile=50.0,
            description='Error - using default'
        )


# ============================================================================
# DYNAMIC TAKE PROFIT CALCULATION
# ============================================================================

def calculate_dynamic_tp(
    entry_price: float,
    sl_price: float,
    direction: str,
    volatility_regime: VolatilityRegime
) -> DynamicTPResult:
    """
    Calculate dynamic take profit levels based on volatility.
    
    Args:
        entry_price: Entry price
        sl_price: Stop loss price
        direction: 'LONG' or 'SHORT'
        volatility_regime: Current volatility classification
    
    Returns:
        DynamicTPResult with calculated TP levels
    """
    if not DYNAMIC_TP_ENABLED:
        # Use static 2R TPs if disabled
        sl_distance = abs(entry_price - sl_price)
        if direction.upper() == 'LONG':
            return DynamicTPResult(
                tp1=entry_price + sl_distance * 2,
                tp2=entry_price + sl_distance * 3,
                tp1_r=2.0,
                tp2_r=3.0,
                regime='STATIC',
                sl_distance=sl_distance
            )
        else:
            return DynamicTPResult(
                tp1=entry_price - sl_distance * 2,
                tp2=entry_price - sl_distance * 3,
                tp1_r=2.0,
                tp2_r=3.0,
                regime='STATIC',
                sl_distance=sl_distance
            )
    
    # Get TP multipliers based on regime
    if volatility_regime.regime == 'HIGH':
        tp_config = DYNAMIC_TP_HIGH_VOL
    elif volatility_regime.regime == 'LOW':
        tp_config = DYNAMIC_TP_LOW_VOL
    else:
        tp_config = DYNAMIC_TP_MEDIUM_VOL
    
    tp1_r = tp_config['tp1_r']
    tp2_r = tp_config['tp2_r']
    
    # Calculate SL distance (1R)
    sl_distance = abs(entry_price - sl_price)
    
    # Calculate TP levels
    if direction.upper() == 'LONG':
        tp1 = entry_price + (sl_distance * tp1_r)
        tp2 = entry_price + (sl_distance * tp2_r)
    else:  # SHORT
        tp1 = entry_price - (sl_distance * tp1_r)
        tp2 = entry_price - (sl_distance * tp2_r)
    
    return DynamicTPResult(
        tp1=tp1,
        tp2=tp2,
        tp1_r=tp1_r,
        tp2_r=tp2_r,
        regime=volatility_regime.regime,
        sl_distance=sl_distance
    )


# ============================================================================
# MULTI-TIMEFRAME CONFLUENCE
# ============================================================================

def analyze_timeframe(
    df: pd.DataFrame,
    timeframe: str
) -> MTFSignal:
    """
    Analyze a single timeframe for trend and signal.
    
    Args:
        df: OHLCV DataFrame for this timeframe
        timeframe: Timeframe identifier (e.g., '4h', '1d')
    
    Returns:
        MTFSignal with analysis results
    """
    if df is None or len(df) < 50:
        return MTFSignal(
            timeframe=timeframe,
            direction='NEUTRAL',
            strength=0,
            trend='RANGING',
            key_levels=[]
        )
    
    try:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Calculate EMAs for trend
        ema_20 = _calculate_ema(close, 20)
        ema_50 = _calculate_ema(close, 50)
        
        current_price = close[-1]
        
        # Determine trend
        if ema_20[-1] > ema_50[-1] and current_price > ema_20[-1]:
            trend = 'BULLISH'
            direction = 'LONG'
            
            # Strength based on price position relative to EMAs
            ema_spread = (ema_20[-1] - ema_50[-1]) / ema_50[-1] * 100
            strength = min(100, 50 + ema_spread * 10)
            
        elif ema_20[-1] < ema_50[-1] and current_price < ema_20[-1]:
            trend = 'BEARISH'
            direction = 'SHORT'
            
            ema_spread = (ema_50[-1] - ema_20[-1]) / ema_50[-1] * 100
            strength = min(100, 50 + ema_spread * 10)
            
        else:
            trend = 'RANGING'
            direction = 'NEUTRAL'
            strength = 30
        
        # Find key levels
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        key_levels = [recent_low, recent_high]
        
        return MTFSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            trend=trend,
            key_levels=key_levels
        )
        
    except Exception as e:
        logging.warning(f"MTF analysis error for {timeframe}: {e}")
        return MTFSignal(
            timeframe=timeframe,
            direction='NEUTRAL',
            strength=0,
            trend='RANGING',
            key_levels=[]
        )


def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    ema = np.zeros(len(data))
    ema[period - 1] = np.mean(data[:period])
    
    multiplier = 2 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_mtf_confluence(
    df_dict: Dict[str, pd.DataFrame]
) -> MTFConfluence:
    """
    Calculate multi-timeframe confluence score.
    
    Args:
        df_dict: Dictionary of DataFrames by timeframe {'1d': df, '4h': df, '1h': df}
    
    Returns:
        MTFConfluence with alignment score and signals
    """
    if not MTF_WEIGHTING_ENABLED:
        return MTFConfluence(
            alignment_score=1.0,
            primary_direction='NEUTRAL',
            signals={},
            confluence_strength=0,
            weighted_confidence=50
        )
    
    signals = {}
    
    # Analyze each timeframe
    for tf, df in df_dict.items():
        if tf in MTF_WEIGHTS:
            signals[tf] = analyze_timeframe(df, tf)
    
    if not signals:
        return MTFConfluence(
            alignment_score=0.5,
            primary_direction='NEUTRAL',
            signals={},
            confluence_strength=0,
            weighted_confidence=50
        )
    
    # Calculate weighted direction
    long_score = 0
    short_score = 0
    total_weight = 0
    
    for tf, signal in signals.items():
        weight = MTF_WEIGHTS.get(tf, 0.25)
        total_weight += weight
        
        if signal.direction == 'LONG':
            long_score += weight * (signal.strength / 100)
        elif signal.direction == 'SHORT':
            short_score += weight * (signal.strength / 100)
    
    # Normalize
    if total_weight > 0:
        long_score /= total_weight
        short_score /= total_weight
    
    # Determine primary direction
    if long_score > short_score + 0.2:
        primary_direction = 'LONG'
        alignment_score = long_score
    elif short_score > long_score + 0.2:
        primary_direction = 'SHORT'
        alignment_score = short_score
    else:
        primary_direction = 'NEUTRAL'
        alignment_score = 0.5
    
    # Count aligned timeframes
    confluence_strength = sum(
        1 for s in signals.values()
        if s.direction == primary_direction
    )
    
    # Weighted confidence
    weighted_confidence = sum(
        signals[tf].strength * MTF_WEIGHTS.get(tf, 0.25)
        for tf in signals
    ) / total_weight if total_weight > 0 else 50
    
    return MTFConfluence(
        alignment_score=alignment_score,
        primary_direction=primary_direction,
        signals=signals,
        confluence_strength=confluence_strength,
        weighted_confidence=weighted_confidence
    )


def is_mtf_aligned(
    mtf_confluence: MTFConfluence,
    signal_direction: str
) -> Tuple[bool, float]:
    """
    Check if signal direction aligns with MTF confluence.
    
    Args:
        mtf_confluence: The MTF confluence result
        signal_direction: Proposed signal direction ('LONG' or 'SHORT')
    
    Returns:
        Tuple of (is_aligned, adjustment)
        - is_aligned: True if signal matches MTF
        - adjustment: Score adjustment (-10 to +10)
    """
    if not MTF_WEIGHTING_ENABLED:
        return True, 0
    
    signal_dir = signal_direction.upper()
    
    if mtf_confluence.primary_direction == 'NEUTRAL':
        # Neutral MTF - slight penalty
        return True, -2
    
    if mtf_confluence.primary_direction == signal_dir:
        # Aligned - bonus based on confluence strength
        bonus = min(10, mtf_confluence.confluence_strength * 3)
        return True, bonus
    else:
        # Counter to MTF - penalty based on alignment score
        penalty = -int(mtf_confluence.alignment_score * 15)
        return False, penalty


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def enhance_signal_with_advanced_features(
    signal: Dict,
    df_dict: Dict[str, pd.DataFrame],
    entry_price: float,
    sl_price: float,
    direction: str
) -> Dict:
    """
    Enhance a signal with all advanced features.
    
    Args:
        signal: Original signal dictionary
        df_dict: DataFrames by timeframe
        entry_price: Entry price
        sl_price: Stop loss price
        direction: Trade direction
    
    Returns:
        Enhanced signal dictionary
    """
    enhanced = signal.copy()
    
    # Get the main timeframe DF for volatility
    main_df = df_dict.get('4h') or df_dict.get('1d') or list(df_dict.values())[0] if df_dict else None
    
    # 1. Volatility regime and dynamic TPs
    if main_df is not None and DYNAMIC_TP_ENABLED:
        vol_regime = detect_volatility_regime(main_df)
        dynamic_tp = calculate_dynamic_tp(entry_price, sl_price, direction, vol_regime)
        
        enhanced['volatility_regime'] = vol_regime.regime
        enhanced['dynamic_tp1'] = dynamic_tp.tp1
        enhanced['dynamic_tp2'] = dynamic_tp.tp2
        enhanced['tp_r_multiples'] = {'tp1': dynamic_tp.tp1_r, 'tp2': dynamic_tp.tp2_r}
    
    # 2. MTF confluence
    if MTF_WEIGHTING_ENABLED:
        mtf_confluence = calculate_mtf_confluence(df_dict)
        is_aligned, mtf_adjustment = is_mtf_aligned(mtf_confluence, direction)
        
        enhanced['mtf_alignment'] = mtf_confluence.alignment_score
        enhanced['mtf_direction'] = mtf_confluence.primary_direction
        enhanced['mtf_confluence_strength'] = mtf_confluence.confluence_strength
        enhanced['mtf_adjusted'] = mtf_adjustment
        enhanced['mtf_aligned'] = is_aligned
        
        # Adjust confidence if present
        if 'confidence' in enhanced:
            enhanced['confidence'] = max(0, min(100, enhanced['confidence'] + mtf_adjustment))
    
    return enhanced
