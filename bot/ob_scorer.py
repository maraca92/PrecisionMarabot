# ob_scorer.py - Grok Elite Signal Bot v27.9.0 - Order Block Quality Scoring
"""
Scores order blocks by quality metrics to select only the best setups.
Ensures minimum OB strength of 2.0 for high win rate.

v27.9.0: NEW MODULE - Quality scoring for OB selection
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def score_order_block(
    ob: Dict,
    current_price: float,
    df: pd.DataFrame,
    direction: str
) -> float:
    """
    Score an order block on multiple quality metrics.
    
    Score components (0-100 total):
    - Strength: 0-25 points (OB strength 1.5-3.5 mapped to 0-25)
    - Freshness: 0-20 points (more recent = better)
    - Distance: 0-20 points (closer to price = better)
    - Volume: 0-15 points (volume at OB formation)
    - HTF Alignment: 0-10 points (direction matches HTF trend)
    - Mitigation: 0-10 points (less mitigation = better)
    
    Args:
        ob: Order block dict
        current_price: Current market price
        df: DataFrame with OHLCV
        direction: 'Long' or 'Short'
    
    Returns:
        Quality score 0-100
    """
    score = 0.0
    
    # ========================================================================
    # 1. STRENGTH SCORE (0-25)
    # Minimum 2.0 required, 2.0-3.5 mapped to 0-25
    # ========================================================================
    strength = ob.get('strength', 0)
    
    if strength < 2.0:
        # Below minimum - heavily penalize
        score -= 20
    else:
        # Map 2.0-3.5 to 0-25 points
        strength_score = min(25, (strength - 2.0) / 1.5 * 25)
        score += strength_score
    
    # ========================================================================
    # 2. FRESHNESS SCORE (0-20)
    # More recent OBs are generally better (less tested)
    # ========================================================================
    ob_index = ob.get('index', 0)
    if df is not None and len(df) > 0:
        total_bars = len(df)
        recency = ob_index / total_bars  # 0 = oldest, 1 = newest
        freshness_score = recency * 20
        score += freshness_score
    
    # ========================================================================
    # 3. DISTANCE SCORE (0-20)
    # Closer to current price = more immediately actionable
    # ========================================================================
    ob_mid = (ob.get('low', 0) + ob.get('high', 0)) / 2
    if current_price > 0 and ob_mid > 0:
        dist_pct = abs(current_price - ob_mid) / current_price * 100
        
        if dist_pct < 1.0:
            distance_score = 20
        elif dist_pct < 2.0:
            distance_score = 15
        elif dist_pct < 3.0:
            distance_score = 10
        elif dist_pct < 5.0:
            distance_score = 5
        else:
            distance_score = 0
        
        score += distance_score
    
    # ========================================================================
    # 4. VOLUME SCORE (0-15)
    # OBs formed with high volume are stronger
    # ========================================================================
    if df is not None and 'volume' in df.columns:
        vol_mean = df['volume'].mean()
        if vol_mean > 0:
            ob_index_safe = min(ob_index, len(df) - 1) if ob_index >= 0 else 0
            ob_vol = df['volume'].iloc[ob_index_safe] if ob_index_safe < len(df) else vol_mean
            vol_ratio = ob_vol / vol_mean
            
            if vol_ratio > 2.0:
                volume_score = 15
            elif vol_ratio > 1.5:
                volume_score = 10
            elif vol_ratio > 1.0:
                volume_score = 5
            else:
                volume_score = 0
            
            score += volume_score
    
    # ========================================================================
    # 5. HTF ALIGNMENT SCORE (0-10)
    # Direction should match higher timeframe trend
    # ========================================================================
    if df is not None and 'ema200' in df.columns and len(df) > 0:
        ema200 = df['ema200'].iloc[-1]
        if pd.notna(ema200) and ema200 > 0:
            if direction == 'Long' and current_price > ema200:
                score += 10  # Long aligned with uptrend
            elif direction == 'Short' and current_price < ema200:
                score += 10  # Short aligned with downtrend
            elif direction == 'Long' and current_price < ema200:
                score += 3  # Counter-trend long (riskier)
            elif direction == 'Short' and current_price > ema200:
                score += 3  # Counter-trend short (riskier)
    
    # ========================================================================
    # 6. MITIGATION SCORE (0-10)
    # Less mitigation = cleaner OB = better
    # ========================================================================
    mitigation = ob.get('mitigation', 0)
    mitigation_score = (1 - mitigation) * 10
    score += mitigation_score
    
    return max(0, min(100, score))


def select_best_ob(
    zones: List[Dict],
    current_price: float,
    df: pd.DataFrame,
    min_score: float = 50.0,
    max_zones: int = 3
) -> List[Dict]:
    """
    Score all zones and return the best ones.
    
    Args:
        zones: List of zone/OB dicts
        current_price: Current market price
        df: DataFrame with OHLCV
        min_score: Minimum quality score to include
        max_zones: Maximum zones to return
    
    Returns:
        Filtered and sorted list of best zones
    """
    scored_zones = []
    
    for zone in zones:
        direction = zone.get('direction', 'Long')
        
        # Create OB-like dict from zone
        ob_data = {
            'low': zone.get('zone_low', zone.get('low', 0)),
            'high': zone.get('zone_high', zone.get('high', 0)),
            'strength': zone.get('strength', zone.get('ob_strength', 1.5)),
            'index': zone.get('index', 0),
            'mitigation': zone.get('mitigation', 0)
        }
        
        score = score_order_block(ob_data, current_price, df, direction)
        
        if score >= min_score:
            zone_copy = zone.copy()
            zone_copy['quality_score'] = score
            scored_zones.append(zone_copy)
            
            logging.debug(f"OB scored: {direction} at {ob_data['low']:.2f}-{ob_data['high']:.2f} -> {score:.1f}")
        else:
            logging.debug(f"OB rejected (score {score:.1f} < {min_score}): {direction}")
    
    # Sort by score descending
    scored_zones = sorted(scored_zones, key=lambda z: z['quality_score'], reverse=True)
    
    # Return top N
    return scored_zones[:max_zones]


def get_ob_quality_label(score: float) -> str:
    """Get human-readable quality label from score."""
    if score >= 80:
        return "ELITE"
    elif score >= 65:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    elif score >= 35:
        return "LOW"
    else:
        return "POOR"


def filter_by_strength(zones: List[Dict], min_strength: float = 2.0) -> List[Dict]:
    """
    Filter zones by minimum OB strength.
    
    Args:
        zones: List of zone dicts
        min_strength: Minimum strength required
    
    Returns:
        Filtered list
    """
    return [
        z for z in zones
        if z.get('strength', z.get('ob_strength', 0)) >= min_strength
    ]
