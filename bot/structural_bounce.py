# structural_bounce.py - Grok Elite Signal Bot v27.12.10 - Structural Bounces
# -*- coding: utf-8 -*-
"""
Detect structural bounce opportunities at psychological levels.

v27.12.10 UPDATES:
1. Added ROADMAP_MAX_DISTANCE_PCT import (7% max distance)
2. Updated find_nearest_psychological_level() to use 7% instead of 15%
3. Added zone distance check in detection loop
4. All v27.10.1 features maintained (dynamic ATR TPs, max 2 zones)

v27.10.1 IMPROVEMENTS:
1. Dynamic TP calculations using ATR (2x and 4x for TP1/TP2) instead of fixed %
2. Integration with Grok opinion layer
3. Max 2 structural zones (quality focused, was 8)
4. Better psychological level granularity
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from bot.config import (
    SYMBOLS,
    # v27.10.1 TP improvements
    STRUCTURAL_TP1_ATR_MULT, STRUCTURAL_TP2_ATR_MULT,
    # Existing structural config
    RELAXED_MAX_ZONES_STRUCTURAL, RELAXED_MIN_OB_STRENGTH,
    RELAXED_MIN_CONFIDENCE
)

# v27.12.10: Import max distance config
try:
    from bot.config import ROADMAP_MAX_DISTANCE_PCT
except ImportError:
    ROADMAP_MAX_DISTANCE_PCT = 7.0  # Default to 7% if not in config

from bot.indicators import add_institutional_indicators
from bot.order_blocks import find_unmitigated_order_blocks


# ============================================================================
# PSYCHOLOGICAL LEVELS (Extended for v27.10.1)
# ============================================================================
PSYCHOLOGICAL_LEVELS = {
    'BTC/USDT': [100000, 95000, 90000, 85000, 80000, 75000, 70000, 65000, 60000, 
                 55000, 50000, 45000, 40000, 35000, 30000, 25000, 20000],
    'ETH/USDT': [5000, 4500, 4000, 3800, 3500, 3200, 3000, 2800, 2500, 2200, 
                 2000, 1800, 1500, 1200, 1000, 800],
    'SOL/USDT': [300, 280, 250, 220, 200, 180, 150, 130, 120, 100, 80, 60, 50, 40],
    'BNB/USDT': [800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200],
    'XRP/USDT': [3.0, 2.8, 2.5, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.75, 0.50, 0.30],
    'ADA/USDT': [2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.75, 0.50, 0.40, 0.30, 0.20],
    'AVAX/USDT': [120, 100, 90, 80, 70, 60, 50, 40, 35, 30, 25, 20, 15]
}


# ============================================================================
# v27.10.1: IMPROVED TP CALCULATION FOR STRUCTURAL BOUNCES
# ============================================================================
def calculate_structural_tp(
    entry_mid: float,
    direction: str,
    atr: float
) -> Tuple[float, float]:
    """
    Calculate TP1 and TP2 for structural bounces.
    v27.10.1: Uses ATR multipliers (2x and 4x) instead of fixed percentages.
    
    Args:
        entry_mid: Entry zone midpoint
        direction: 'Long' or 'Short'
        atr: Current ATR value
    
    Returns:
        (tp1, tp2) prices
    """
    # Structural bounces are more conservative
    tp1_mult = STRUCTURAL_TP1_ATR_MULT  # 2.0x ATR
    tp2_mult = STRUCTURAL_TP2_ATR_MULT  # 4.0x ATR
    
    if direction == 'Long':
        tp1 = entry_mid + (atr * tp1_mult)
        tp2 = entry_mid + (atr * tp2_mult)
    else:  # Short
        tp1 = entry_mid - (atr * tp1_mult)
        tp2 = entry_mid - (atr * tp2_mult)
    
    logging.debug(f"Structural TP: ATR={atr:.2f}, TP1={tp1:.4f} ({tp1_mult}x), TP2={tp2:.4f} ({tp2_mult}x)")
    
    return tp1, tp2


# ============================================================================
# FIND NEAREST PSYCHOLOGICAL LEVEL - v27.12.10 Updated
# ============================================================================
def find_nearest_psychological_level(symbol: str, price: float) -> Optional[float]:
    """
    Find the nearest psychological level for a symbol.
    v27.12.10: Only returns level if within ROADMAP_MAX_DISTANCE_PCT (7%) of price.
    
    Args:
        symbol: Trading pair
        price: Current price
    
    Returns:
        Nearest psychological level or None if too far
    """
    levels = PSYCHOLOGICAL_LEVELS.get(symbol, [])
    if not levels:
        return None
    
    # Find closest level
    closest = min(levels, key=lambda x: abs(x - price))
    
    # v27.12.10: Only return if within ROADMAP_MAX_DISTANCE_PCT of current price
    dist_pct = abs(price - closest) / price * 100
    if dist_pct <= ROADMAP_MAX_DISTANCE_PCT:
        return closest
    
    return None


# ============================================================================
# DETECT STRUCTURAL BOUNCES - v27.12.10 (With 7% Distance Filter)
# ============================================================================
async def detect_structural_bounces_batch(
    data_cache: Dict,
    prices: Dict,
    btc_trend: str
) -> List[Dict]:
    """
    Detect structural bounce opportunities at psychological levels.
    v27.12.10: Only zones within ROADMAP_MAX_DISTANCE_PCT (7%) of current price.
    v27.10.1: Improved TP calculations using ATR, max 2 zones.
    
    Args:
        data_cache: Dict of symbol -> timeframe -> DataFrame
        prices: Dict of current prices
        btc_trend: 'bullish' or 'bearish'
    
    Returns:
        List of structural zone dicts
    """
    
    structural_zones = []
    
    for symbol in SYMBOLS:
        try:
            price = prices.get(symbol)
            if price is None:
                continue
            
            # v27.12.10: Find nearest psychological level (now respects 7% limit)
            psych_level = find_nearest_psychological_level(symbol, price)
            if psych_level is None:
                logging.debug(f"{symbol}: No psychological level within {ROADMAP_MAX_DISTANCE_PCT}%")
                continue
            
            dist_to_level = abs(price - psych_level) / price * 100
            
            # Only consider if close to level (within 8% for entry)
            if dist_to_level > 8.0:
                continue
            
            # Get data
            df_1d = data_cache.get(symbol, {}).get('1d')
            df_4h = data_cache.get(symbol, {}).get('4h', pd.DataFrame())
            
            if df_1d is None or len(df_1d) < 50:
                continue
            
            # Find order blocks near psychological level
            obs = await find_unmitigated_order_blocks(df_1d, lookback=100, tf='1d')
            
            atr = df_1d['atr'].iloc[-1] if 'atr' in df_1d.columns else price * 0.02
            
            # Check for bullish bounce at support
            if price < psych_level * 1.08:  # Below or near level
                for ob in obs.get('bullish', []):
                    if ob['strength'] < RELAXED_MIN_OB_STRENGTH:
                        continue
                    
                    ob_mid = (ob['low'] + ob['high']) / 2
                    
                    # OB should be near psychological level
                    ob_dist = abs(ob_mid - psych_level) / psych_level * 100
                    if ob_dist > 5.0:
                        continue
                    
                    # v27.12.10: Check zone distance from current price
                    zone_dist_pct = abs(price - ob_mid) / price * 100
                    if zone_dist_pct > ROADMAP_MAX_DISTANCE_PCT:
                        logging.debug(f"{symbol}: Long OB zone too far ({zone_dist_pct:.1f}% > {ROADMAP_MAX_DISTANCE_PCT}%)")
                        continue
                    
                    # Build zone
                    entry_low = max(ob['low'], psych_level * 0.98)
                    entry_high = min(ob['high'], psych_level * 1.02)
                    entry_mid = (entry_low + entry_high) / 2
                    
                    sl = entry_low - (atr * 0.5)
                    
                    # v27.10.1: Calculate dynamic TPs
                    tp1, tp2 = calculate_structural_tp(
                        entry_mid=entry_mid,
                        direction='Long',
                        atr=atr
                    )
                    
                    # Determine if counter-trend
                    is_counter_trend = False
                    if btc_trend and 'bear' in btc_trend.lower():
                        is_counter_trend = True
                    
                    # Build confluence
                    confluence_factors = [f"Psych ${psych_level:,}", f"OB str{ob['strength']:.1f}"]
                    
                    # Check volume
                    if len(df_1d) >= 20:
                        vol_ratio = df_1d['volume'].iloc[-1] / df_1d['volume'].rolling(20).mean().iloc[-1]
                        if vol_ratio > 1.3:
                            confluence_factors.append("Vol surge")
                    
                    # Check EMA alignment
                    if 'ema_20' in df_1d.columns and 'ema_50' in df_1d.columns:
                        if df_1d['ema_20'].iloc[-1] > df_1d['ema_50'].iloc[-1]:
                            confluence_factors.append("EMA align")
                    
                    confluence_str = " +".join(confluence_factors)
                    
                    # Calculate confidence
                    base_conf = 55
                    base_conf += int(ob['strength'] * 5)
                    base_conf += len(confluence_factors) * 3
                    if not is_counter_trend:
                        base_conf += 5
                    
                    confidence = min(85, base_conf)
                    
                    if confidence < RELAXED_MIN_CONFIDENCE:
                        continue
                    
                    zone = {
                        'symbol': symbol,
                        'type': 'structural_bounce',
                        'direction': 'Long',
                        'zone_low': entry_low,
                        'zone_high': entry_high,
                        'entry_low': entry_low,
                        'entry_high': entry_high,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'leverage': 3,
                        'confidence': confidence,
                        'ob_strength': ob['strength'],
                        'confluence': confluence_str,
                        'psychological_level': psych_level,
                        'is_counter_trend': is_counter_trend,
                        'created_at': datetime.now(timezone.utc),
                        'timeframe': '1d',
                        'converted': False,
                        'alert_count': 0,
                        'dist_pct': zone_dist_pct  # v27.12.10: Store distance
                    }
                    
                    structural_zones.append(zone)
                    break  # One per symbol
            
            # Check for bearish bounce at resistance
            elif price > psych_level * 0.92:  # Above or near level
                for ob in obs.get('bearish', []):
                    if ob['strength'] < RELAXED_MIN_OB_STRENGTH:
                        continue
                    
                    ob_mid = (ob['low'] + ob['high']) / 2
                    
                    # OB should be near psychological level
                    ob_dist = abs(ob_mid - psych_level) / psych_level * 100
                    if ob_dist > 5.0:
                        continue
                    
                    # v27.12.10: Check zone distance from current price
                    zone_dist_pct = abs(price - ob_mid) / price * 100
                    if zone_dist_pct > ROADMAP_MAX_DISTANCE_PCT:
                        logging.debug(f"{symbol}: Short OB zone too far ({zone_dist_pct:.1f}% > {ROADMAP_MAX_DISTANCE_PCT}%)")
                        continue
                    
                    # Build zone
                    entry_low = max(ob['low'], psych_level * 0.98)
                    entry_high = min(ob['high'], psych_level * 1.02)
                    entry_mid = (entry_low + entry_high) / 2
                    
                    sl = entry_high + (atr * 0.5)
                    
                    # v27.10.1: Calculate dynamic TPs
                    tp1, tp2 = calculate_structural_tp(
                        entry_mid=entry_mid,
                        direction='Short',
                        atr=atr
                    )
                    
                    # Determine if counter-trend
                    is_counter_trend = False
                    if btc_trend and 'bull' in btc_trend.lower():
                        is_counter_trend = True
                    
                    # Build confluence
                    confluence_factors = [f"Psych ${psych_level:,}", f"OB str{ob['strength']:.1f}"]
                    
                    # Check volume
                    if len(df_1d) >= 20:
                        vol_ratio = df_1d['volume'].iloc[-1] / df_1d['volume'].rolling(20).mean().iloc[-1]
                        if vol_ratio > 1.3:
                            confluence_factors.append("Vol surge")
                    
                    # Check EMA alignment
                    if 'ema_20' in df_1d.columns and 'ema_50' in df_1d.columns:
                        if df_1d['ema_20'].iloc[-1] < df_1d['ema_50'].iloc[-1]:
                            confluence_factors.append("EMA align")
                    
                    confluence_str = " +".join(confluence_factors)
                    
                    # Calculate confidence
                    base_conf = 55
                    base_conf += int(ob['strength'] * 5)
                    base_conf += len(confluence_factors) * 3
                    if not is_counter_trend:
                        base_conf += 5
                    
                    confidence = min(85, base_conf)
                    
                    if confidence < RELAXED_MIN_CONFIDENCE:
                        continue
                    
                    zone = {
                        'symbol': symbol,
                        'type': 'structural_bounce',
                        'direction': 'Short',
                        'zone_low': entry_low,
                        'zone_high': entry_high,
                        'entry_low': entry_low,
                        'entry_high': entry_high,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'leverage': 3,
                        'confidence': confidence,
                        'ob_strength': ob['strength'],
                        'confluence': confluence_str,
                        'psychological_level': psych_level,
                        'is_counter_trend': is_counter_trend,
                        'created_at': datetime.now(timezone.utc),
                        'timeframe': '1d',
                        'converted': False,
                        'alert_count': 0,
                        'dist_pct': zone_dist_pct  # v27.12.10: Store distance
                    }
                    
                    structural_zones.append(zone)
                    break  # One per symbol
            
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logging.error(f"{symbol}: Structural bounce detection error: {e}")
            continue
    
    # Sort by confidence and return top zones (v27.12.10: max 2)
    structural_zones = sorted(structural_zones, key=lambda z: z['confidence'], reverse=True)
    structural_zones = structural_zones[:RELAXED_MAX_ZONES_STRUCTURAL]
    
    logging.info(f"Structural bounce detection: {len(structural_zones)} zones (max {RELAXED_MAX_ZONES_STRUCTURAL}, within {ROADMAP_MAX_DISTANCE_PCT}%)")
    
    return structural_zones


# ============================================================================
# VALIDATE STRUCTURAL ZONE - v27.12.10 Updated
# ============================================================================
def validate_structural_zone(symbol: str, zone: Dict, current_price: float) -> Tuple[bool, Dict]:
    """
    Validate a structural bounce zone before conversion.
    v27.12.10: Uses ROADMAP_MAX_DISTANCE_PCT for distance threshold.
    
    Args:
        symbol: Trading pair
        zone: Zone dict with entry, SL, TP, confidence, etc.
        current_price: Current market price
    
    Returns:
        (is_valid, validation_dict)
    """
    
    checks = {}
    
    # 1. Distance check (v27.12.10: use ROADMAP_MAX_DISTANCE_PCT instead of 8.0)
    zone_mid = (zone['zone_low'] + zone['zone_high']) / 2
    dist_pct = abs(current_price - zone_mid) / current_price * 100
    
    checks['distance'] = {
        'passed': dist_pct <= ROADMAP_MAX_DISTANCE_PCT,
        'value': dist_pct,
        'threshold': ROADMAP_MAX_DISTANCE_PCT,
        'label': f'Distance {dist_pct:.1f}%'
    }
    
    # 2. Confidence check
    checks['confidence'] = {
        'passed': zone['confidence'] >= RELAXED_MIN_CONFIDENCE,
        'value': zone['confidence'],
        'threshold': RELAXED_MIN_CONFIDENCE,
        'label': f'Confidence {zone["confidence"]}%'
    }
    
    # 3. OB strength check
    checks['ob_strength'] = {
        'passed': zone.get('ob_strength', 0) >= RELAXED_MIN_OB_STRENGTH,
        'value': zone.get('ob_strength', 0),
        'threshold': RELAXED_MIN_OB_STRENGTH,
        'label': f'OB {zone.get("ob_strength", 0):.1f}'
    }
    
    # Need at least 2 of 3 checks to pass
    passed = sum(1 for c in checks.values() if c['passed'])
    
    return passed >= 2, {
        'valid': passed >= 2,
        'checks': checks,
        'passed_count': passed
    }
