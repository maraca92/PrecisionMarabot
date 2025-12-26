# liquidity_map.py - Grok Elite Signal Bot v27.12.13 - Liquidity Heatmap Analysis
# -*- coding: utf-8 -*-
"""
Liquidity Heatmap Analysis Module

Identifies where stop losses are likely clustered for high-probability reversal zones.
Key concepts:
- Swing highs/lows are natural stop loss locations
- Round numbers attract stops (psychological levels)
- Recent highs/lows have more stops than old ones
- Volume at levels indicates significance

v27.12.13: Initial implementation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

# Minimum significance for a liquidity cluster
MIN_CLUSTER_SIGNIFICANCE = 3.0

# Weight factors for different liquidity sources
LIQUIDITY_WEIGHTS = {
    'swing_high': 2.0,       # Strong resistance = stop losses above
    'swing_low': 2.0,        # Strong support = stop losses below
    'round_number': 1.5,     # Psychological levels
    'recent_high': 1.8,      # Recent highs have fresh stops
    'recent_low': 1.8,       # Recent lows have fresh stops
    'volume_node': 1.3,      # High volume areas
    'gap': 1.5,              # Price gaps attract fills
}

# Round number intervals by price range
ROUND_NUMBER_INTERVALS = {
    'btc': 1000,      # $87,000, $88,000, etc.
    'eth': 50,        # $2,900, $2,950, etc.
    'default_high': 10,   # For prices > $100
    'default_mid': 1,     # For prices $10-$100
    'default_low': 0.1,   # For prices < $10
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LiquidityCluster:
    """Represents a cluster of liquidity (stop losses)."""
    price_level: float
    price_low: float
    price_high: float
    significance: float
    cluster_type: str  # 'stops_above' or 'stops_below'
    sources: List[str]
    recency_score: float  # 0-1, higher = more recent
    volume_at_level: float
    distance_pct: float  # Distance from current price


@dataclass
class LiquidityMap:
    """Full liquidity map for a symbol."""
    symbol: str
    current_price: float
    clusters_above: List[LiquidityCluster]
    clusters_below: List[LiquidityCluster]
    nearest_resistance: Optional[float]
    nearest_support: Optional[float]
    liquidity_bias: str  # 'above', 'below', 'balanced'
    generated_at: datetime


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """
    Find swing highs and swing lows in price data.
    
    These are key liquidity zones because traders place stops just beyond them.
    """
    if len(df) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(lookback, len(df) - lookback):
        # Check for swing high (local maximum)
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            recency = (i - lookback) / (len(df) - lookback * 2)  # 0 to 1
            swing_highs.append({
                'price': highs[i],
                'index': i,
                'recency': recency,
                'volume': df['volume'].iloc[i] if 'volume' in df.columns else 0
            })
        
        # Check for swing low (local minimum)
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            recency = (i - lookback) / (len(df) - lookback * 2)
            swing_lows.append({
                'price': lows[i],
                'index': i,
                'recency': recency,
                'volume': df['volume'].iloc[i] if 'volume' in df.columns else 0
            })
    
    return swing_highs, swing_lows


def find_round_numbers(price: float, symbol: str, range_pct: float = 5.0) -> List[Dict]:
    """
    Find psychological round number levels near current price.
    
    These attract liquidity because retail traders set stops at round numbers.
    """
    # Determine interval based on symbol/price
    if 'BTC' in symbol.upper():
        interval = ROUND_NUMBER_INTERVALS['btc']
    elif 'ETH' in symbol.upper():
        interval = ROUND_NUMBER_INTERVALS['eth']
    elif price > 100:
        interval = ROUND_NUMBER_INTERVALS['default_high']
    elif price > 10:
        interval = ROUND_NUMBER_INTERVALS['default_mid']
    else:
        interval = ROUND_NUMBER_INTERVALS['default_low']
    
    # Calculate range
    range_val = price * (range_pct / 100)
    low_bound = price - range_val
    high_bound = price + range_val
    
    # Find round numbers in range
    round_numbers = []
    level = (low_bound // interval) * interval
    
    while level <= high_bound:
        if low_bound <= level <= high_bound and level > 0:
            distance_pct = abs(level - price) / price * 100
            round_numbers.append({
                'price': level,
                'significance': LIQUIDITY_WEIGHTS['round_number'],
                'distance_pct': distance_pct
            })
        level += interval
    
    return round_numbers


def find_volume_nodes(df: pd.DataFrame, num_bins: int = 20) -> List[Dict]:
    """
    Find high volume price levels (volume profile nodes).
    
    High volume areas indicate significant liquidity.
    """
    if len(df) < 10 or 'volume' not in df.columns:
        return []
    
    # Create volume profile
    price_range = df['high'].max() - df['low'].min()
    if price_range <= 0:
        return []
    
    bin_size = price_range / num_bins
    
    # Calculate volume at each price level
    volume_profile = {}
    for _, row in df.iterrows():
        mid_price = (row['high'] + row['low']) / 2
        bin_idx = int((mid_price - df['low'].min()) / bin_size)
        bin_price = df['low'].min() + (bin_idx + 0.5) * bin_size
        
        if bin_price not in volume_profile:
            volume_profile[bin_price] = 0
        volume_profile[bin_price] += row['volume']
    
    if not volume_profile:
        return []
    
    # Find high volume nodes (above average)
    avg_volume = np.mean(list(volume_profile.values()))
    
    nodes = []
    for price, volume in volume_profile.items():
        if volume > avg_volume * 1.5:  # 50% above average
            nodes.append({
                'price': price,
                'volume': volume,
                'significance': (volume / avg_volume) * LIQUIDITY_WEIGHTS['volume_node']
            })
    
    return sorted(nodes, key=lambda x: x['volume'], reverse=True)[:5]


def cluster_liquidity_levels(
    levels: List[Dict],
    current_price: float,
    tolerance_pct: float = 0.5
) -> List[LiquidityCluster]:
    """
    Cluster nearby liquidity levels together.
    
    Levels within tolerance_pct are merged into single clusters.
    """
    if not levels:
        return []
    
    # Sort by price
    sorted_levels = sorted(levels, key=lambda x: x['price'])
    
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        prev_price = current_cluster[-1]['price']
        tolerance = prev_price * (tolerance_pct / 100)
        
        if abs(level['price'] - prev_price) <= tolerance:
            current_cluster.append(level)
        else:
            # Finalize current cluster
            if current_cluster:
                cluster = _create_cluster(current_cluster, current_price)
                if cluster.significance >= MIN_CLUSTER_SIGNIFICANCE:
                    clusters.append(cluster)
            current_cluster = [level]
    
    # Don't forget the last cluster
    if current_cluster:
        cluster = _create_cluster(current_cluster, current_price)
        if cluster.significance >= MIN_CLUSTER_SIGNIFICANCE:
            clusters.append(cluster)
    
    return clusters


def _create_cluster(levels: List[Dict], current_price: float) -> LiquidityCluster:
    """Create a LiquidityCluster from a group of levels."""
    prices = [l['price'] for l in levels]
    avg_price = np.mean(prices)
    
    # Determine if stops are above or below this level
    cluster_type = 'stops_above' if avg_price > current_price else 'stops_below'
    
    # Calculate significance (sum of all level significances)
    significance = sum(l.get('significance', 1.0) for l in levels)
    
    # Calculate recency (average)
    recency_scores = [l.get('recency', 0.5) for l in levels]
    avg_recency = np.mean(recency_scores)
    
    # Get sources
    sources = list(set(l.get('source', 'unknown') for l in levels))
    
    # Total volume
    total_volume = sum(l.get('volume', 0) for l in levels)
    
    # Distance from current price
    distance_pct = abs(avg_price - current_price) / current_price * 100
    
    return LiquidityCluster(
        price_level=avg_price,
        price_low=min(prices),
        price_high=max(prices),
        significance=significance,
        cluster_type=cluster_type,
        sources=sources,
        recency_score=avg_recency,
        volume_at_level=total_volume,
        distance_pct=distance_pct
    )


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def generate_liquidity_map(
    symbol: str,
    df: pd.DataFrame,
    current_price: float,
    max_distance_pct: float = 7.0
) -> Optional[LiquidityMap]:
    """
    Generate a complete liquidity map for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        df: OHLCV DataFrame
        current_price: Current market price
        max_distance_pct: Maximum distance to consider
    
    Returns:
        LiquidityMap with clusters above and below current price
    """
    if df is None or len(df) < 20:
        return None
    
    try:
        all_levels = []
        
        # 1. Find swing points
        swing_highs, swing_lows = find_swing_points(df, lookback=5)
        
        for sh in swing_highs:
            all_levels.append({
                'price': sh['price'],
                'significance': LIQUIDITY_WEIGHTS['swing_high'] * (0.5 + sh['recency'] * 0.5),
                'recency': sh['recency'],
                'volume': sh['volume'],
                'source': 'swing_high'
            })
        
        for sl in swing_lows:
            all_levels.append({
                'price': sl['price'],
                'significance': LIQUIDITY_WEIGHTS['swing_low'] * (0.5 + sl['recency'] * 0.5),
                'recency': sl['recency'],
                'volume': sl['volume'],
                'source': 'swing_low'
            })
        
        # 2. Find round numbers
        round_nums = find_round_numbers(current_price, symbol, range_pct=max_distance_pct)
        for rn in round_nums:
            all_levels.append({
                'price': rn['price'],
                'significance': rn['significance'],
                'recency': 0.5,  # Neutral recency for static levels
                'volume': 0,
                'source': 'round_number'
            })
        
        # 3. Find volume nodes
        volume_nodes = find_volume_nodes(df)
        for vn in volume_nodes:
            all_levels.append({
                'price': vn['price'],
                'significance': vn['significance'],
                'recency': 0.7,  # Recent volume is more relevant
                'volume': vn['volume'],
                'source': 'volume_node'
            })
        
        # 4. Recent highs/lows (last 20 candles)
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        
        all_levels.append({
            'price': recent_high,
            'significance': LIQUIDITY_WEIGHTS['recent_high'],
            'recency': 0.9,
            'volume': 0,
            'source': 'recent_high'
        })
        
        all_levels.append({
            'price': recent_low,
            'significance': LIQUIDITY_WEIGHTS['recent_low'],
            'recency': 0.9,
            'volume': 0,
            'source': 'recent_low'
        })
        
        # Filter by distance
        filtered_levels = [
            l for l in all_levels
            if abs(l['price'] - current_price) / current_price * 100 <= max_distance_pct
        ]
        
        # Cluster the levels
        all_clusters = cluster_liquidity_levels(filtered_levels, current_price)
        
        # Separate above/below
        clusters_above = sorted(
            [c for c in all_clusters if c.price_level > current_price],
            key=lambda x: x.distance_pct
        )
        clusters_below = sorted(
            [c for c in all_clusters if c.price_level < current_price],
            key=lambda x: x.distance_pct
        )
        
        # Calculate bias
        sig_above = sum(c.significance for c in clusters_above)
        sig_below = sum(c.significance for c in clusters_below)
        
        if sig_above > sig_below * 1.3:
            liquidity_bias = 'above'
        elif sig_below > sig_above * 1.3:
            liquidity_bias = 'below'
        else:
            liquidity_bias = 'balanced'
        
        return LiquidityMap(
            symbol=symbol,
            current_price=current_price,
            clusters_above=clusters_above[:5],  # Top 5
            clusters_below=clusters_below[:5],
            nearest_resistance=clusters_above[0].price_level if clusters_above else None,
            nearest_support=clusters_below[0].price_level if clusters_below else None,
            liquidity_bias=liquidity_bias,
            generated_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logging.error(f"Liquidity map generation failed for {symbol}: {e}")
        return None


def get_liquidity_signal(
    liquidity_map: LiquidityMap,
    direction: str
) -> Dict:
    """
    Get a trading signal based on liquidity analysis.
    
    Args:
        liquidity_map: The generated liquidity map
        direction: 'LONG' or 'SHORT'
    
    Returns:
        Dict with signal adjustments
    """
    if not liquidity_map:
        return {'adjustment': 0, 'reason': 'No liquidity data'}
    
    adjustment = 0
    reasons = []
    
    if direction.upper() == 'LONG':
        # For longs, we want liquidity above (stop hunts to sweep)
        if liquidity_map.liquidity_bias == 'above':
            adjustment += 5
            reasons.append("Liquidity above to target")
        elif liquidity_map.liquidity_bias == 'below':
            adjustment -= 3
            reasons.append("Liquidity below (may sweep first)")
        
        # Check if near support cluster
        if liquidity_map.clusters_below:
            nearest = liquidity_map.clusters_below[0]
            if nearest.distance_pct < 1.5:
                adjustment += 5
                reasons.append(f"Near support cluster ({nearest.distance_pct:.1f}%)")
    
    else:  # SHORT
        # For shorts, we want liquidity below (stop hunts to sweep)
        if liquidity_map.liquidity_bias == 'below':
            adjustment += 5
            reasons.append("Liquidity below to target")
        elif liquidity_map.liquidity_bias == 'above':
            adjustment -= 3
            reasons.append("Liquidity above (may sweep first)")
        
        # Check if near resistance cluster
        if liquidity_map.clusters_above:
            nearest = liquidity_map.clusters_above[0]
            if nearest.distance_pct < 1.5:
                adjustment += 5
                reasons.append(f"Near resistance cluster ({nearest.distance_pct:.1f}%)")
    
    return {
        'adjustment': adjustment,
        'reason': ' | '.join(reasons) if reasons else 'Neutral liquidity',
        'bias': liquidity_map.liquidity_bias,
        'nearest_support': liquidity_map.nearest_support,
        'nearest_resistance': liquidity_map.nearest_resistance
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_liquidity_report(liquidity_map: LiquidityMap) -> str:
    """Format a liquidity map as a readable report."""
    if not liquidity_map:
        return "No liquidity data available"
    
    lines = [
        f"📊 **Liquidity Map: {liquidity_map.symbol}**",
        f"Current Price: ${liquidity_map.current_price:,.2f}",
        f"Bias: {liquidity_map.liquidity_bias.upper()}",
        ""
    ]
    
    if liquidity_map.clusters_above:
        lines.append("**Resistance Clusters (stops above):**")
        for c in liquidity_map.clusters_above[:3]:
            lines.append(f"  • ${c.price_level:,.2f} ({c.distance_pct:.1f}% away) - Sig: {c.significance:.1f}")
    
    if liquidity_map.clusters_below:
        lines.append("")
        lines.append("**Support Clusters (stops below):**")
        for c in liquidity_map.clusters_below[:3]:
            lines.append(f"  • ${c.price_level:,.2f} ({c.distance_pct:.1f}% away) - Sig: {c.significance:.1f}")
    
    return "\n".join(lines)
