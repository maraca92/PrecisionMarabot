# ote.py - Grok Elite Signal Bot v27.8.3 - Optimal Trade Entry (OTE)
"""
OTE (Optimal Trade Entry) - 62-79% Fibonacci retracement zones.
ICT concept: The sweet spot for entries after a swing move.

v27.8.3: New module for OTE confluence detection
"""
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

# ============================================================================
# SWING HIGH/LOW DETECTION
# ============================================================================

def find_swing_high_low(df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
    """
    Find most recent significant swing high and low.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Bars to look back for swing detection
    
    Returns:
        Dict with swing_high, swing_low, range, or None
    """
    if len(df) < lookback:
        return None
    
    try:
        # Get recent data
        recent = df.tail(lookback * 2).copy()
        
        # Find swing high (highest high in lookback period)
        swing_high = recent['high'].max()
        swing_high_idx = recent['high'].idxmax()
        
        # Find swing low (lowest low in lookback period)
        swing_low = recent['low'].min()
        swing_low_idx = recent['low'].idxmin()
        
        # Calculate range
        range_size = swing_high - swing_low
        
        if range_size <= 0:
            return None
        
        # Determine current market structure
        current_price = df['close'].iloc[-1]
        
        # Is this an upswing or downswing?
        if swing_high_idx > swing_low_idx:
            structure = 'upswing'  # Low formed first, then high
        else:
            structure = 'downswing'  # High formed first, then low
        
        return {
            'swing_high': float(swing_high),
            'swing_low': float(swing_low),
            'range': float(range_size),
            'structure': structure,
            'current_price': float(current_price)
        }
        
    except Exception as e:
        logging.debug(f"Swing detection error: {e}")
        return None

# ============================================================================
# OTE ZONE CALCULATION
# ============================================================================

def calculate_ote_zones(swing_high: float, swing_low: float, 
                        structure: str) -> Dict:
    """
    Calculate 62-79% OTE retracement zones.
    
    Args:
        swing_high: Recent swing high
        swing_low: Recent swing low
        structure: 'upswing' or 'downswing'
    
    Returns:
        Dict with bullish and bearish OTE zones
    """
    range_size = swing_high - swing_low
    
    if range_size <= 0:
        return {'bullish': None, 'bearish': None}
    
    # Bullish OTE: Retracement from high down to 62-79% (looking to buy dip)
    # After upswing, price retraces down to OTE zone
    bull_ote_low = swing_low + (range_size * 0.62)   # 62% from low
    bull_ote_high = swing_low + (range_size * 0.79)  # 79% from low
    
    # Bearish OTE: Retracement from low up to 62-79% (looking to sell rally)
    # After downswing, price retraces up to OTE zone
    bear_ote_low = swing_high - (range_size * 0.79)  # 79% from high (lower bound)
    bear_ote_high = swing_high - (range_size * 0.62) # 62% from high (upper bound)
    
    return {
        'bullish': {
            'ote_low': float(bull_ote_low),
            'ote_high': float(bull_ote_high),
            'ote_mid': float((bull_ote_low + bull_ote_high) / 2),
            'fib_62': float(bull_ote_low),
            'fib_79': float(bull_ote_high)
        },
        'bearish': {
            'ote_low': float(bear_ote_low),
            'ote_high': float(bear_ote_high),
            'ote_mid': float((bear_ote_low + bear_ote_high) / 2),
            'fib_62': float(bear_ote_high),
            'fib_79': float(bear_ote_low)
        }
    }

# ============================================================================
# OTE CONFLUENCE DETECTION
# ============================================================================

def check_ote_confluence(
    price: float,
    zone: Dict,
    ote_zones: Dict,
    direction: str
) -> Optional[Dict]:
    """
    Check if price/zone is in OTE and provides confluence.
    
    Args:
        price: Current market price
        zone: Order block zone dict
        ote_zones: OTE zones from calculate_ote_zones()
        direction: 'Long' or 'Short'
    
    Returns:
        Dict with OTE confluence info or None
    """
    if not ote_zones:
        return None
    
    ote_zone = ote_zones.get('bullish' if direction == 'Long' else 'bearish')
    
    if not ote_zone:
        return None
    
    ote_low = ote_zone['ote_low']
    ote_high = ote_zone['ote_high']
    
    # Check if current price is in OTE zone
    price_in_ote = ote_low <= price <= ote_high
    
    # Check if order block overlaps with OTE zone
    zone_low = zone.get('zone_low', zone.get('low', 0))
    zone_high = zone.get('zone_high', zone.get('high', 0))
    zone_mid = (zone_low + zone_high) / 2
    
    # OB is in OTE if its midpoint is in the OTE zone
    ob_in_ote = ote_low <= zone_mid <= ote_high
    
    # Calculate how deep into OTE we are (for confidence scaling)
    ote_range = ote_high - ote_low
    if ote_range > 0:
        # Distance from optimal (70.5% is the sweet spot)
        optimal = ote_low + (ote_range * 0.5)  # Middle of 62-79% range
        distance_from_optimal = abs(price - optimal) / ote_range
        
        # Closer to optimal = higher confidence boost
        if distance_from_optimal < 0.2:  # Within 20% of optimal
            confidence_boost = 7
        elif distance_from_optimal < 0.4:  # Within 40% of optimal
            confidence_boost = 5
        else:
            confidence_boost = 3
    else:
        confidence_boost = 5
    
    if price_in_ote or ob_in_ote:
        # Calculate exact Fib level for display
        fib_level = ((price - ote_zone['fib_62']) / (ote_zone['fib_79'] - ote_zone['fib_62']) * 17 + 62)
        fib_level = max(62, min(79, fib_level))  # Clamp to 62-79
        
        return {
            'in_ote': price_in_ote,
            'ob_in_ote': ob_in_ote,
            'ote_low': ote_low,
            'ote_high': ote_high,
            'fib_level': fib_level,
            'confluence': f"+OTE({fib_level:.0f}%)",
            'confidence_boost': confidence_boost
        }
    
    return None

# ============================================================================
# MAIN OTE ANALYSIS FUNCTION
# ============================================================================

def analyze_ote(df: pd.DataFrame, price: float, zones: list, 
                direction: str, lookback: int = 50) -> Tuple[bool, Optional[Dict], str]:
    """
    Perform complete OTE analysis for a symbol.
    
    Args:
        df: DataFrame with OHLCV data
        price: Current market price
        zones: List of order block zones
        direction: 'Long' or 'Short'
        lookback: Bars to look back for swing detection
    
    Returns:
        Tuple of (has_ote_confluence, ote_data, log_message)
    """
    try:
        # Find swing high/low
        swings = find_swing_high_low(df, lookback)
        
        if not swings:
            return False, None, "No clear swing structure"
        
        # Calculate OTE zones
        ote_zones = calculate_ote_zones(
            swings['swing_high'],
            swings['swing_low'],
            swings['structure']
        )
        
        if not ote_zones:
            return False, None, "OTE calculation failed"
        
        # Check for OTE confluence
        best_confluence = None
        for zone in zones:
            confluence = check_ote_confluence(price, zone, ote_zones, direction)
            
            if confluence:
                # Use the best confluence found
                if not best_confluence or confluence['confidence_boost'] > best_confluence['confidence_boost']:
                    best_confluence = confluence
        
        if best_confluence:
            ote_zone = ote_zones.get('bullish' if direction == 'Long' else 'bearish')
            log_msg = (
                f"IN OTE ZONE: {best_confluence['fib_level']:.0f}% Fib "
                f"(${ote_zone['ote_low']:.2f}-${ote_zone['ote_high']:.2f}) "
                f"+{best_confluence['confidence_boost']}% conf"
            )
            return True, best_confluence, log_msg
        else:
            ote_zone = ote_zones.get('bullish' if direction == 'Long' else 'bearish')
            distance = abs(price - ote_zone['ote_mid']) / price * 100
            log_msg = f"Outside OTE ({distance:.1f}% away from optimal)"
            return False, None, log_msg
        
    except Exception as e:
        logging.error(f"OTE analysis error: {e}")
        return False, None, f"OTE error: {str(e)}"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_ote_info(ote_data: Optional[Dict]) -> str:
    """Format OTE data for display in signals"""
    if not ote_data:
        return ""
    
    return f"OTE: {ote_data['fib_level']:.0f}% Fib (${ote_data['ote_low']:.2f}-${ote_data['ote_high']:.2f})"

def get_ote_visual(df: pd.DataFrame, lookback: int = 50) -> Optional[str]:
    """
    Get visual representation of OTE zones for debugging.
    
    Returns:
        String with OTE zone information
    """
    swings = find_swing_high_low(df, lookback)
    
    if not swings:
        return None
    
    ote_zones = calculate_ote_zones(
        swings['swing_high'],
        swings['swing_low'],
        swings['structure']
    )
    
    current_price = df['close'].iloc[-1]
    
    bull_zone = ote_zones['bullish']
    bear_zone = ote_zones['bearish']
    
    visual = f"""
OTE Analysis:
Swing High: ${swings['swing_high']:.2f}
Swing Low: ${swings['swing_low']:.2f}
Current: ${current_price:.2f}

Bullish OTE (62-79%): ${bull_zone['ote_low']:.2f} - ${bull_zone['ote_high']:.2f}
Bearish OTE (62-79%): ${bear_zone['ote_low']:.2f} - ${bear_zone['ote_high']:.2f}
    """
    
    return visual
