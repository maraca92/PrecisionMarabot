# dynamic_sl.py - Grok Elite Signal Bot v27.9.0 - Structure-Based Stop Loss
"""
Calculates stop loss based on market structure rather than arbitrary percentages.
Uses 15m swing highs/lows for precision placement.

v27.9.0: NEW MODULE - Structure-based SL for better risk management
"""

import logging
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Dict[str, list]:
    """
    Find swing high and swing low points in the data.
    
    Args:
        df: DataFrame with OHLCV
        lookback: Number of bars to look for swing points
    
    Returns:
        Dict with 'swing_highs' and 'swing_lows' lists
    """
    swing_highs = []
    swing_lows = []
    
    if len(df) < lookback * 2 + 1:
        return {'swing_highs': [], 'swing_lows': []}
    
    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append({
                'index': i,
                'price': df['high'].iloc[i],
                'date': df['date'].iloc[i] if 'date' in df.columns else None
            })
        
        # Check for swing low
        is_swing_low = True
        for j in range(i - lookback, i + lookback + 1):
            if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append({
                'index': i,
                'price': df['low'].iloc[i],
                'date': df['date'].iloc[i] if 'date' in df.columns else None
            })
    
    return {'swing_highs': swing_highs, 'swing_lows': swing_lows}


async def calculate_structure_based_sl(
    symbol: str,
    direction: str,
    entry_price: float,
    df_15m: pd.DataFrame,
    max_sl_pct: float = 2.5,
    buffer_pct: float = 0.1
) -> Optional[float]:
    """
    Calculate SL based on market structure (swing points).
    
    For Long: SL below recent swing low
    For Short: SL above recent swing high
    
    Args:
        symbol: Trading pair
        direction: 'Long' or 'Short'
        entry_price: Entry price for the trade
        df_15m: 15-minute DataFrame
        max_sl_pct: Maximum SL distance as percentage
        buffer_pct: Buffer below/above swing point
    
    Returns:
        Structure-based SL price or None
    """
    if df_15m is None or len(df_15m) < 30:
        return None
    
    try:
        # Find swing points
        swings = find_swing_points(df_15m, lookback=3)
        
        if direction == 'Long':
            # For longs, find swing lows below entry
            swing_lows = swings.get('swing_lows', [])
            
            # Filter to swing lows below entry
            valid_lows = [
                sw for sw in swing_lows
                if sw['price'] < entry_price
            ]
            
            if not valid_lows:
                return None
            
            # Get closest swing low to entry
            closest_low = max(valid_lows, key=lambda x: x['price'])
            
            # Place SL below swing low with buffer
            sl_price = closest_low['price'] * (1 - buffer_pct / 100)
            
            # Check max distance
            sl_dist_pct = abs(entry_price - sl_price) / entry_price * 100
            if sl_dist_pct > max_sl_pct:
                # Use max SL instead
                sl_price = entry_price * (1 - max_sl_pct / 100)
            
            logging.debug(f"{symbol}: Long structure SL at {sl_price:.4f} (swing low {closest_low['price']:.4f})")
            return sl_price
        
        else:  # Short
            # For shorts, find swing highs above entry
            swing_highs = swings.get('swing_highs', [])
            
            # Filter to swing highs above entry
            valid_highs = [
                sw for sw in swing_highs
                if sw['price'] > entry_price
            ]
            
            if not valid_highs:
                return None
            
            # Get closest swing high to entry
            closest_high = min(valid_highs, key=lambda x: x['price'])
            
            # Place SL above swing high with buffer
            sl_price = closest_high['price'] * (1 + buffer_pct / 100)
            
            # Check max distance
            sl_dist_pct = abs(sl_price - entry_price) / entry_price * 100
            if sl_dist_pct > max_sl_pct:
                # Use max SL instead
                sl_price = entry_price * (1 + max_sl_pct / 100)
            
            logging.debug(f"{symbol}: Short structure SL at {sl_price:.4f} (swing high {closest_high['price']:.4f})")
            return sl_price
    
    except Exception as e:
        logging.debug(f"{symbol}: Structure SL calculation error: {e}")
        return None


def calculate_sl_with_atr_buffer(
    direction: str,
    structure_level: float,
    atr: float,
    buffer_mult: float = 0.3
) -> float:
    """
    Add ATR-based buffer to structure level.
    
    Args:
        direction: 'Long' or 'Short'
        structure_level: The swing high/low price
        atr: Current ATR value
        buffer_mult: ATR multiplier for buffer
    
    Returns:
        SL price with buffer
    """
    buffer = atr * buffer_mult
    
    if direction == 'Long':
        return structure_level - buffer
    else:
        return structure_level + buffer


def validate_sl_distance(
    entry_price: float,
    sl_price: float,
    direction: str,
    min_pct: float = 0.5,
    max_pct: float = 3.0
) -> Tuple[bool, str]:
    """
    Validate that SL distance is reasonable.
    
    Args:
        entry_price: Entry price
        sl_price: Stop loss price
        direction: 'Long' or 'Short'
        min_pct: Minimum SL distance
        max_pct: Maximum SL distance
    
    Returns:
        Tuple of (is_valid, reason)
    """
    dist_pct = abs(entry_price - sl_price) / entry_price * 100
    
    if dist_pct < min_pct:
        return False, f"SL too tight: {dist_pct:.2f}% < {min_pct}%"
    
    if dist_pct > max_pct:
        return False, f"SL too wide: {dist_pct:.2f}% > {max_pct}%"
    
    # Direction check
    if direction == 'Long' and sl_price >= entry_price:
        return False, "Long SL must be below entry"
    
    if direction == 'Short' and sl_price <= entry_price:
        return False, "Short SL must be above entry"
    
    return True, f"Valid SL: {dist_pct:.2f}%"
