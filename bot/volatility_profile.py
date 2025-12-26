# volatility_profile.py - Grok Elite Signal Bot v27.12.1 - Volatility & Beta Analysis
# -*- coding: utf-8 -*-
"""
v27.12.1: NEW MODULE - Volatility Profile for Claude Context

Provides:
1. Beta to BTC calculation (correlation-based)
2. ATR% (daily volatility measure)
3. Annualized volatility
4. Volatility regime classification
5. Formatted context string for Claude

This helps Claude make smarter decisions based on:
- High beta = favor trend continuation
- Low volatility = require extreme confluence
- High volatility = wider TPs, avoid counter-trend
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timezone


# ============================================================================
# VOLATILITY THRESHOLDS (tunable)
# ============================================================================
HIGH_ATR_PCT = 6.0          # Daily ATR% above this = high volatility
LOW_ATR_PCT = 3.0           # Daily ATR% below this = low/choppy
HIGH_ANN_VOL = 70.0         # Annualized vol above this = high
LOW_ANN_VOL = 35.0          # Annualized vol below this = low
HIGH_BETA = 1.5             # Beta above this = amplifies BTC moves
LOW_BETA = 0.8              # Beta below this = decoupled from BTC
LOOKBACK_DAYS = 20          # Days for rolling calculations


# ============================================================================
# MAIN CALCULATION FUNCTION
# ============================================================================
def calculate_volatility_profile(
    symbol: str,
    df_symbol: pd.DataFrame,
    df_btc: pd.DataFrame,
    lookback: int = LOOKBACK_DAYS
) -> Dict:
    """
    Calculate comprehensive volatility metrics for a symbol.
    
    Args:
        symbol: Trading pair (e.g., 'SOL/USDT')
        df_symbol: Daily OHLCV DataFrame for the symbol
        df_btc: Daily OHLCV DataFrame for BTC/USDT
        lookback: Number of days for rolling calculations
    
    Returns:
        Dict with beta, atr_pct, ann_vol_pct, regime, trend_bias, guidance
    """
    result = {
        'symbol': symbol,
        'beta': 1.0,
        'atr_pct': 4.0,
        'ann_vol_pct': 50.0,
        'regime': 'MEDIUM',
        'trend_bias': 'NEUTRAL',
        'confidence_adjustment': 0,
        'guidance': '',
        'calculated_at': datetime.now(timezone.utc).isoformat()
    }
    
    try:
        # Validate inputs
        if df_symbol is None or len(df_symbol) < lookback:
            logging.debug(f"{symbol}: Insufficient data for volatility calc ({len(df_symbol) if df_symbol is not None else 0} rows)")
            return result
        
        # ====================================================================
        # 1. BETA TO BTC (correlation * volatility ratio)
        # ====================================================================
        if df_btc is not None and len(df_btc) >= lookback:
            try:
                # Calculate returns
                symbol_returns = df_symbol['close'].pct_change().dropna().tail(lookback)
                btc_returns = df_btc['close'].pct_change().dropna().tail(lookback)
                
                # Align lengths
                min_len = min(len(symbol_returns), len(btc_returns))
                if min_len >= 10:  # Need at least 10 data points
                    symbol_returns = symbol_returns.tail(min_len)
                    btc_returns = btc_returns.tail(min_len)
                    
                    # Calculate correlation
                    correlation = symbol_returns.corr(btc_returns)
                    
                    # Calculate standard deviations
                    symbol_std = symbol_returns.std()
                    btc_std = btc_returns.std()
                    
                    # Beta = correlation * (symbol_vol / btc_vol)
                    if btc_std > 0 and not np.isnan(correlation):
                        beta = correlation * (symbol_std / btc_std)
                        result['beta'] = round(float(beta), 2)
                    
            except Exception as e:
                logging.debug(f"{symbol}: Beta calculation error: {e}")
        
        # ====================================================================
        # 2. ATR PERCENTAGE (current daily volatility)
        # ====================================================================
        try:
            if 'atr' in df_symbol.columns:
                atr = df_symbol['atr'].iloc[-1]
            else:
                # Calculate ATR manually if not present
                high = df_symbol['high'].tail(14)
                low = df_symbol['low'].tail(14)
                close = df_symbol['close'].tail(14)
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.mean()
            
            current_price = df_symbol['close'].iloc[-1]
            if current_price > 0 and not np.isnan(atr):
                atr_pct = (atr / current_price) * 100
                result['atr_pct'] = round(float(atr_pct), 2)
                
        except Exception as e:
            logging.debug(f"{symbol}: ATR calculation error: {e}")
        
        # ====================================================================
        # 3. ANNUALIZED VOLATILITY (historical characteristic)
        # ====================================================================
        try:
            daily_returns = df_symbol['close'].pct_change().dropna().tail(lookback)
            
            if len(daily_returns) >= 10:
                daily_vol = daily_returns.std()
                # Annualize: daily_vol * sqrt(252 trading days)
                ann_vol_pct = daily_vol * np.sqrt(252) * 100
                result['ann_vol_pct'] = round(float(ann_vol_pct), 1)
                
        except Exception as e:
            logging.debug(f"{symbol}: Annualized vol calculation error: {e}")
        
        # ====================================================================
        # 4. VOLATILITY REGIME CLASSIFICATION
        # ====================================================================
        atr_pct = result['atr_pct']
        ann_vol = result['ann_vol_pct']
        beta = result['beta']
        
        if atr_pct > HIGH_ATR_PCT or ann_vol > HIGH_ANN_VOL:
            result['regime'] = 'HIGH'
        elif atr_pct < LOW_ATR_PCT or ann_vol < LOW_ANN_VOL:
            result['regime'] = 'LOW'
        else:
            result['regime'] = 'MEDIUM'
        
        # ====================================================================
        # 5. TREND BIAS (based on beta + regime)
        # ====================================================================
        if beta > HIGH_BETA and result['regime'] == 'HIGH':
            result['trend_bias'] = 'STRONG_TREND'
            result['confidence_adjustment'] = 15
        elif beta > HIGH_BETA:
            result['trend_bias'] = 'TREND_FAVORED'
            result['confidence_adjustment'] = 10
        elif beta < LOW_BETA:
            result['trend_bias'] = 'INDEPENDENT'
            result['confidence_adjustment'] = 0
        elif result['regime'] == 'LOW':
            result['trend_bias'] = 'CHOPPY'
            result['confidence_adjustment'] = -10
        else:
            result['trend_bias'] = 'NEUTRAL'
            result['confidence_adjustment'] = 0
        
        # ====================================================================
        # 6. GENERATE GUIDANCE TEXT
        # ====================================================================
        result['guidance'] = _generate_guidance(result)
        
        logging.debug(f"{symbol}: Vol profile - Beta:{beta:.2f} ATR:{atr_pct:.1f}% Ann:{ann_vol:.0f}% Regime:{result['regime']}")
        
    except Exception as e:
        logging.error(f"{symbol}: Volatility profile error: {e}")
    
    return result


def _generate_guidance(vol_data: Dict) -> str:
    """Generate trading guidance based on volatility profile."""
    regime = vol_data.get('regime', 'MEDIUM')
    beta = vol_data.get('beta', 1.0)
    
    if regime == 'HIGH':
        return (
            "HIGH VOLATILITY: Favor trend continuation. "
            "Widen TP targets 20-30%. "
            "Penalize counter-trend by -15% confidence. "
            "Use structure-based SL with buffer."
        )
    elif regime == 'LOW':
        return (
            "LOW VOLATILITY: Require extreme confluence (5+ factors). "
            "Tighten TP targets. "
            "Skip marginal setups - high fakeout probability. "
            "Prefer waiting for breakout."
        )
    else:
        if beta > HIGH_BETA:
            return (
                "MEDIUM VOL + HIGH BETA: Standard rules but favor trend. "
                "+10% confidence for trend continuation setups."
            )
        else:
            return (
                "MEDIUM VOLATILITY: Apply standard confluence requirements. "
                "Both trend and reversal setups viable with proper confluence."
            )


# ============================================================================
# CONTEXT BUILDER FOR CLAUDE
# ============================================================================
def build_volatility_context(symbol: str, vol_data: Dict) -> str:
    """
    Build formatted volatility context string for Claude prompt.
    
    Args:
        symbol: Trading pair
        vol_data: Dict from calculate_volatility_profile()
    
    Returns:
        Formatted string to include in Claude context
    """
    if not vol_data:
        return ""
    
    beta = vol_data.get('beta', 1.0)
    atr_pct = vol_data.get('atr_pct', 4.0)
    ann_vol = vol_data.get('ann_vol_pct', 50.0)
    regime = vol_data.get('regime', 'MEDIUM')
    trend_bias = vol_data.get('trend_bias', 'NEUTRAL')
    guidance = vol_data.get('guidance', '')
    
    symbol_short = symbol.replace('/USDT', '')
    
    # Beta interpretation
    if beta >= HIGH_BETA:
        beta_note = "HIGH - Amplifies BTC moves significantly"
    elif beta <= LOW_BETA:
        beta_note = "LOW - Decoupled from BTC, independent moves"
    else:
        beta_note = "MODERATE - Normal BTC correlation"
    
    # ATR interpretation
    if atr_pct > HIGH_ATR_PCT:
        atr_note = "HIGH - Large daily swings"
    elif atr_pct < LOW_ATR_PCT:
        atr_note = "LOW - Compressed/choppy"
    else:
        atr_note = "NORMAL"
    
    context = f"""
## Volatility Profile: {symbol_short}

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Beta to BTC | {beta:.2f} | {beta_note} |
| Daily ATR | {atr_pct:.1f}% | {atr_note} |
| Annualized Vol | {ann_vol:.0f}% | {regime} regime |
| Trend Bias | {trend_bias} | |

**{guidance}**

**CONFIDENCE ADJUSTMENT RULES:**
- Setup MATCHES volatility profile (e.g., trend continuation in HIGH vol): +10-15%
- Setup MISMATCHES profile (e.g., reversal in HIGH vol without extreme confluence): -15-20%
- LOW volatility regime without 5+ confluence factors: SKIP the setup
- Always reference volatility in your reasoning

"""
    return context


# ============================================================================
# ASYNC WRAPPER FOR MAIN.PY INTEGRATION
# ============================================================================
async def get_volatility_profile(
    symbol: str,
    data_cache: Dict,
    btc_data: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Async wrapper to get volatility profile from cached data.
    
    Args:
        symbol: Trading pair
        data_cache: Dict with symbol -> {timeframe -> DataFrame}
        btc_data: Optional pre-fetched BTC DataFrame
    
    Returns:
        Volatility profile dict
    """
    try:
        # Get symbol's 1d data from cache
        symbol_data = data_cache.get(symbol, {})
        df_symbol = symbol_data.get('1d')
        
        # Get BTC data from cache or use provided
        if btc_data is None:
            btc_cache = data_cache.get('BTC/USDT', {})
            df_btc = btc_cache.get('1d')
        else:
            df_btc = btc_data
        
        # If symbol IS BTC, beta is always 1.0
        if 'BTC' in symbol:
            df_btc = df_symbol  # Use self for BTC
        
        return calculate_volatility_profile(symbol, df_symbol, df_btc)
        
    except Exception as e:
        logging.error(f"{symbol}: get_volatility_profile error: {e}")
        return {
            'symbol': symbol,
            'beta': 1.0,
            'atr_pct': 4.0,
            'ann_vol_pct': 50.0,
            'regime': 'MEDIUM',
            'trend_bias': 'NEUTRAL',
            'confidence_adjustment': 0,
            'guidance': 'Volatility data unavailable - use standard rules.'
        }


# ============================================================================
# HELPER FOR QUICK REGIME CHECK
# ============================================================================
def should_skip_low_vol_setup(vol_data: Dict, confluence_count: int) -> bool:
    """
    Quick check if a setup should be skipped due to low volatility.
    
    Args:
        vol_data: Volatility profile dict
        confluence_count: Number of confluence factors
    
    Returns:
        True if setup should be skipped
    """
    if vol_data.get('regime') == 'LOW' and confluence_count < 5:
        return True
    return False


def get_volatility_confidence_adjustment(
    vol_data: Dict,
    direction: str,
    btc_trend: str,
    is_counter_trend: bool
) -> int:
    """
    Calculate confidence adjustment based on volatility profile and trade direction.
    
    Args:
        vol_data: Volatility profile dict
        direction: 'Long' or 'Short'
        btc_trend: 'Uptrend', 'Downtrend', or 'Sideways'
        is_counter_trend: Whether trade is against BTC trend
    
    Returns:
        Confidence adjustment value (can be negative)
    """
    regime = vol_data.get('regime', 'MEDIUM')
    beta = vol_data.get('beta', 1.0)
    base_adjustment = vol_data.get('confidence_adjustment', 0)
    
    adjustment = base_adjustment
    
    # High volatility + counter-trend = penalty
    if regime == 'HIGH' and is_counter_trend:
        adjustment -= 15
    
    # High volatility + high beta + trend continuation = bonus
    if regime == 'HIGH' and beta > HIGH_BETA and not is_counter_trend:
        adjustment += 10
    
    # Low volatility = generally reduce confidence
    if regime == 'LOW':
        adjustment -= 10
    
    # Cap adjustment
    return max(-25, min(25, adjustment))
