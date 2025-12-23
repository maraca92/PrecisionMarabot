# order_blocks.py - Grok Elite Signal Bot v27.10.1 - Order Block Detection
"""
Order Block (OB) detection and analysis.

v27.10.1 CRITICAL FIXES:
- min_strength parameter defaults to 0.5 (was hardcoded 2.0)
- Removed double filtering
- Added extensive logging
- Relaxed data requirements (30 candles instead of 50)
- Removed 'dead regime' skip
- Increased OB_DISTANCE_PCT to 15%
"""
import logging
from typing import Dict, List, Optional
import pandas as pd
import pandas_ta as ta
import numpy as np

# Safe config imports with relaxed fallbacks
try:
    from bot.config import (
        VOL_SURGE_MULTIPLIER, OB_OVERLAP_THRESHOLD, DAILY_ATR_MULT,
        OB_MIN_MOVE_PCT, OB_MIN_STRENGTH, OB_MAX_MITIGATION, OB_DISTANCE_PCT,
        BASE_CONFIDENCE_THRESHOLD
    )
except ImportError:
    VOL_SURGE_MULTIPLIER = 3.0
    OB_OVERLAP_THRESHOLD = 0.3
    DAILY_ATR_MULT = 1.5
    OB_MIN_MOVE_PCT = 0.5
    OB_MIN_STRENGTH = 2.0
    OB_MAX_MITIGATION = 0.6
    OB_DISTANCE_PCT = 5.0
    BASE_CONFIDENCE_THRESHOLD = 60

try:
    from bot.data_fetcher import fetch_ohlcv
except ImportError:
    async def fetch_ohlcv(*args, **kwargs):
        return pd.DataFrame()

try:
    from bot.utils import zones_overlap
except ImportError:
    def zones_overlap(l1, h1, l2, h2):
        overlap = min(h1, h2) - max(l1, l2)
        range1 = h1 - l1
        return max(0, overlap / range1) if range1 > 0 else 0

# v27.10.1: Very relaxed defaults
DEFAULT_MIN_STRENGTH = 0.5  # Return ALL OBs, let caller filter
RELAXED_OB_DISTANCE_PCT = 15.0  # Allow OBs up to 15% away


# ============================================================================
# ORDER BLOCK MITIGATION TRACKING
# ============================================================================
def track_ob_mitigation(ob: Dict, df_since_formation: pd.DataFrame) -> float:
    """Calculate how much an order block has been mitigated."""
    ob_mid = (ob['low'] + ob['high']) / 2
    ob_range = ob['high'] - ob['low']
    
    if ob_range == 0:
        return 1.0
    
    if len(df_since_formation) == 0:
        return 0.0
    
    touches = df_since_formation[
        (df_since_formation['low'] <= ob['high']) &
        (df_since_formation['high'] >= ob['low'])
    ]
    
    if len(touches) == 0:
        return 0.0
    
    total_vol_in_ob = touches['volume'].sum()
    vol_mean = df_since_formation['volume'].rolling(5, min_periods=1).mean()
    formation_vol = vol_mean.iloc[0] * 5 if len(vol_mean) > 0 else 1
    
    if formation_vol == 0:
        mitigation_score = 0.5
    else:
        mitigation_score = min(total_vol_in_ob / formation_vol, 1.0)
    
    if ob_range > 0:
        deepest_penetration = max(
            (touches['high'].max() - ob_mid) / ob_range,
            (ob_mid - touches['low'].min()) / ob_range
        )
        if deepest_penetration > 0.7:
            mitigation_score += 0.3
    
    return min(mitigation_score, 1.0)


# ============================================================================
# UNMITIGATED ORDER BLOCK DETECTION - v27.10.1 FIXED
# ============================================================================
async def find_unmitigated_order_blocks(
    df: pd.DataFrame,
    lookback: int = 100,
    atr_mult: float = 1.5,
    tf: str = None,
    symbol: str = None,
    min_strength: float = None  # v27.10.1: Caller specifies threshold
) -> Dict[str, List[Dict]]:
    """
    Find unmitigated order blocks.
    
    v27.10.1 FIX:
    - min_strength defaults to 0.5 (very low) to return ALL OBs
    - Caller filters to their desired threshold
    - Added logging for debugging
    """
    if len(df) < 20:  # v27.10.1: Relaxed from 30
        logging.debug(f"OB detection: Insufficient data ({len(df)} candles)")
        return {'bullish': [], 'bearish': []}
    
    # v27.10.1: Use very low threshold if not specified
    if min_strength is None:
        min_strength = DEFAULT_MIN_STRENGTH
    
    ltf_mult = 1 if tf in ['15m', '1h'] else 2 if tf == '4h' else 3
    dyn_lookback = min(lookback * ltf_mult, len(df) - 5)
    df_local = df.tail(dyn_lookback).copy()
    
    # Calculate indicators
    df_local['atr'] = ta.atr(df_local['high'], df_local['low'], df_local['close'], 14)
    df_local['direction'] = np.where(df_local['close'] > df_local['open'], 1, -1)
    
    # Swing detection with smaller window for more swings
    df_local['swing_high'] = df_local['high'].rolling(7, center=True, min_periods=1).max() == df_local['high']
    df_local['swing_low'] = df_local['low'].rolling(7, center=True, min_periods=1).min() == df_local['low']
    
    vol_sma = df_local['volume'].rolling(20, min_periods=1).mean()
    df_local['volume_sma'] = vol_sma
    
    obs = {'bullish': [], 'bearish': []}
    
    # Count swings for debugging
    swing_highs = df_local['swing_high'].sum()
    swing_lows = df_local['swing_low'].sum()
    logging.debug(f"OB detection {symbol} {tf}: {swing_highs} swing highs, {swing_lows} swing lows")
    
    # BEARISH ORDER BLOCKS (from swing highs)
    bearish_candidates = 0
    for i in range(5, len(df_local) - 3):
        if df_local['swing_high'].iloc[i]:
            bearish_candidates += 1
            ob_high = df_local['high'].iloc[i]
            ob_low = max(df_local['open'].iloc[i], df_local['close'].iloc[i])
            
            atr_val = df_local['atr'].iloc[i]
            if pd.isna(atr_val) or atr_val == 0:
                atr_val = (ob_high - ob_low) * 2  # Fallback
            
            future_lows = df_local['low'].iloc[i+1:min(i+10, len(df_local))]
            if len(future_lows) == 0:
                continue
            
            # v27.10.1: Relaxed move requirements
            move_down = future_lows.min() < ob_low - (atr_val * 1.0)  # Was 1.5
            move_pct = abs((ob_low - future_lows.min()) / ob_low) if ob_low > 0 else 0
            
            if move_down and move_pct > 0.003:  # 0.3% minimum move
                df_since = df_local.iloc[i+1:]
                mitigation = track_ob_mitigation({'low': ob_low, 'high': ob_high}, df_since)
                
                if mitigation < 0.7:  # v27.10.1: Relaxed from 0.6
                    zone_type = 'Breaker' if any(df_local['low'].iloc[i+1:min(i+6, len(df_local))] < ob_low) else 'OB'
                    
                    vol_sma_val = df_local['volume_sma'].iloc[i]
                    current_vol = df_local['volume'].iloc[i]
                    
                    if pd.isna(vol_sma_val) or vol_sma_val == 0:
                        strength = 1.5
                    else:
                        vol_ratio = current_vol / vol_sma_val
                        strength = 1.0 + min(vol_ratio, 2.5)
                    
                    adjusted_strength = strength * (1 - mitigation * 0.4)
                    
                    # v27.10.1: Use caller's threshold, not global
                    if adjusted_strength >= min_strength:
                        obs['bearish'].append({
                            'low': ob_low,
                            'high': ob_high,
                            'type': zone_type,
                            'strength': adjusted_strength,
                            'index': i,
                            'mitigation': mitigation
                        })
    
    # BULLISH ORDER BLOCKS (from swing lows)
    bullish_candidates = 0
    for i in range(5, len(df_local) - 3):
        if df_local['swing_low'].iloc[i]:
            bullish_candidates += 1
            ob_low = df_local['low'].iloc[i]
            ob_high = min(df_local['open'].iloc[i], df_local['close'].iloc[i])
            
            atr_val = df_local['atr'].iloc[i]
            if pd.isna(atr_val) or atr_val == 0:
                atr_val = (ob_high - ob_low) * 2  # Fallback
            
            future_highs = df_local['high'].iloc[i+1:min(i+10, len(df_local))]
            if len(future_highs) == 0:
                continue
            
            # v27.10.1: Relaxed move requirements
            move_up = future_highs.max() > ob_high + (atr_val * 1.0)  # Was 1.5
            move_pct = abs((future_highs.max() - ob_high) / ob_high) if ob_high > 0 else 0
            
            if move_up and move_pct > 0.003:  # 0.3% minimum move
                df_since = df_local.iloc[i+1:]
                mitigation = track_ob_mitigation({'low': ob_low, 'high': ob_high}, df_since)
                
                if mitigation < 0.7:  # v27.10.1: Relaxed from 0.6
                    zone_type = 'Breaker' if any(df_local['high'].iloc[i+1:min(i+6, len(df_local))] > ob_high) else 'OB'
                    
                    vol_sma_val = df_local['volume_sma'].iloc[i]
                    current_vol = df_local['volume'].iloc[i]
                    
                    if pd.isna(vol_sma_val) or vol_sma_val == 0:
                        strength = 1.5
                    else:
                        vol_ratio = current_vol / vol_sma_val
                        strength = 1.0 + min(vol_ratio, 2.5)
                    
                    adjusted_strength = strength * (1 - mitigation * 0.4)
                    
                    # v27.10.1: Use caller's threshold, not global
                    if adjusted_strength >= min_strength:
                        obs['bullish'].append({
                            'low': ob_low,
                            'high': ob_high,
                            'type': zone_type,
                            'strength': adjusted_strength,
                            'index': i,
                            'mitigation': mitigation
                        })
    
    # Log results
    logging.info(f"OB detection {symbol} {tf}: {len(obs['bullish'])} bullish, {len(obs['bearish'])} bearish (min_str={min_strength})")
    
    # Sort by strength
    for key in obs:
        obs[key] = sorted(obs[key], key=lambda z: z['strength'], reverse=True)[:15]
    
    return obs


# ============================================================================
# MULTI-TIMEFRAME CONFLUENCE
# ============================================================================
async def calculate_mtf_confluence(symbol: str, price: float, direction: str) -> float:
    """Calculate multi-timeframe confluence score."""
    score = 0.0
    timeframes = ['1h', '4h', '1d']
    weights = [1, 2, 3]
    
    for tf, weight in zip(timeframes, weights):
        try:
            df = await fetch_ohlcv(symbol, tf, 100)
            if len(df) == 0:
                continue
            
            try:
                from bot.indicators import add_institutional_indicators
                df = add_institutional_indicators(df)
            except:
                pass
            
            if 'ema200' in df.columns and len(df) >= 10:
                ema_current = df['ema200'].iloc[-1]
                ema_prev = df['ema200'].iloc[-10]
                if pd.notna(ema_current) and pd.notna(ema_prev) and ema_prev != 0:
                    ema_slope = (ema_current - ema_prev) / ema_prev
                    if (direction == 'Long' and ema_slope > 0) or (direction == 'Short' and ema_slope < 0):
                        score += weight * 2
            
            obs = await find_unmitigated_order_blocks(df, tf=tf, min_strength=0.5)
            relevant_obs = obs['bullish'] if direction == 'Long' else obs['bearish']
            
            if relevant_obs:
                closest_ob = min(relevant_obs, key=lambda x: abs((x['low']+x['high'])/2 - price))
                dist_pct = abs((closest_ob['low']+closest_ob['high'])/2 - price) / price * 100
                
                if dist_pct < 5:
                    score += weight * 3 * (5 - dist_pct) / 5
        
        except Exception as e:
            logging.debug(f"MTF confluence error for {symbol} {tf}: {e}")
            continue
    
    return min(score / 30, 1.0)


# ============================================================================
# PREMIUM ZONE DETECTION - v27.10.1 COMPLETELY REWRITTEN
# ============================================================================
async def find_next_premium_zones(
    df: pd.DataFrame,
    current_price: float,
    tf: str,
    symbol: str = None,
    oi_data: Optional[Dict[str, float]] = None,
    trend: str = None,
    whale_data: Optional[Dict] = None,  # Backwards compatibility
    order_book: Optional[Dict] = None,
    min_strength: float = None
) -> List[Dict]:
    """
    Find premium/discount zones with order blocks.
    
    v27.10.1 COMPLETE REWRITE:
    - Removed double filtering
    - Removed 'dead regime' skip
    - Added extensive logging
    - Relaxed all thresholds
    - Works in ALL market conditions
    """
    if len(df) < 20:
        logging.debug(f"Premium zones: Insufficient data for {symbol} {tf}")
        return []
    
    # v27.10.1: Default to very low threshold
    if min_strength is None:
        min_strength = 0.5
    
    # Get ALL order blocks with low threshold
    obs = await find_unmitigated_order_blocks(
        df, lookback=150, tf=tf, symbol=symbol, min_strength=min_strength
    )
    
    total_obs = len(obs.get('bullish', [])) + len(obs.get('bearish', []))
    logging.info(f"Premium zones {symbol} {tf}: Found {total_obs} raw OBs")
    
    if total_obs == 0:
        return []
    
    zones = []
    
    # Process bullish OBs (Long entries in discount - BELOW price)
    for ob in obs.get('bullish', []):
        mid = (ob['low'] + ob['high']) / 2
        dist_pct = abs(current_price - mid) / current_price * 100
        
        # v27.10.1: Skip if too close (<0.1%) or too far (>15%)
        if dist_pct < 0.1 or dist_pct > RELAXED_OB_DISTANCE_PCT:
            continue
        
        # Long zones should be BELOW current price (discount)
        if mid > current_price:
            continue
        
        # Build confluence
        confluence = f"{ob['type']} ({tf})"
        confluence += " +Discount"
        
        # Calculate confidence
        base_conf = 50 + ob['strength'] * 10
        if tf == '1d':
            base_conf += 8
        elif tf == '4h':
            base_conf += 5
        
        # Distance bonus (closer = better)
        if dist_pct < 2:
            base_conf += 8
        elif dist_pct < 5:
            base_conf += 5
        elif dist_pct < 10:
            base_conf += 2
        
        # OI factor
        if oi_data:
            oi_change = oi_data.get('oi_change_pct', 0)
            if abs(oi_change) > 3:
                confluence += f" +OI({oi_change:+.1f}%)"
                base_conf += 3
        
        confidence = min(int(base_conf), 95)
        
        # v27.10.1: Very relaxed minimum
        if confidence < 50:
            continue
        
        zones.append({
            'direction': 'Long',
            'zone_low': ob['low'],
            'zone_high': ob['high'],
            'strength': ob['strength'],
            'type': ob['type'],
            'confluence': confluence,
            'prob': confidence,
            'confidence': confidence,
            'mitigation': ob.get('mitigation', 0),
            'dist_pct': dist_pct
        })
    
    # Process bearish OBs (Short entries in premium - ABOVE price)
    for ob in obs.get('bearish', []):
        mid = (ob['low'] + ob['high']) / 2
        dist_pct = abs(current_price - mid) / current_price * 100
        
        # v27.10.1: Skip if too close (<0.1%) or too far (>15%)
        if dist_pct < 0.1 or dist_pct > RELAXED_OB_DISTANCE_PCT:
            continue
        
        # Short zones should be ABOVE current price (premium)
        if mid < current_price:
            continue
        
        # Build confluence
        confluence = f"{ob['type']} ({tf})"
        confluence += " +Premium"
        
        # Calculate confidence
        base_conf = 50 + ob['strength'] * 10
        if tf == '1d':
            base_conf += 8
        elif tf == '4h':
            base_conf += 5
        
        # Distance bonus
        if dist_pct < 2:
            base_conf += 8
        elif dist_pct < 5:
            base_conf += 5
        elif dist_pct < 10:
            base_conf += 2
        
        # OI factor
        if oi_data:
            oi_change = oi_data.get('oi_change_pct', 0)
            if abs(oi_change) > 3:
                confluence += f" +OI({oi_change:+.1f}%)"
                base_conf += 3
        
        confidence = min(int(base_conf), 95)
        
        # v27.10.1: Very relaxed minimum
        if confidence < 50:
            continue
        
        zones.append({
            'direction': 'Short',
            'zone_low': ob['low'],
            'zone_high': ob['high'],
            'strength': ob['strength'],
            'type': ob['type'],
            'confluence': confluence,
            'prob': confidence,
            'confidence': confidence,
            'mitigation': ob.get('mitigation', 0),
            'dist_pct': dist_pct
        })
    
    # Sort by confidence and return top zones
    zones = sorted(zones, key=lambda z: z['prob'], reverse=True)
    
    logging.info(f"Premium zones {symbol} {tf}: Returning {len(zones)} zones (from {total_obs} OBs)")
    
    return zones[:10]
