# trading.py - Grok Elite Signal Bot v27.11.0 - Trading Logic
# -*- coding: utf-8 -*-
"""
Core trading logic with Phase 2 additions:
- Signal grading (A/B/C/D/F)
- Structure break detection (BOS/CHoCH)
- Counter-trend TP1-only strategy
- Psychology confluence

v27.11.0: FIXES
- Added None safety checks for SL, TP1, TP2, entry_price
- Prevents "'>=' not supported between 'float' and 'NoneType'" errors
- Safe fallbacks for missing trade fields

v27.9.5: PHASE 2 - Balanced grading, structure breaks, psychology
"""
import logging
import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import aiofiles
import pandas as pd
import numpy as np

from bot.config import (
    RISK_PER_TRADE_PCT, MAX_DRAWDOWN_PCT, SLIPPAGE_PCT, ENTRY_SLIPPAGE_PCT,
    FEE_PCT, SIMULATED_CAPITAL, PAPER_TRADING, MAX_CONCURRENT_TRADES,
    CHAT_ID, COUNTER_TREND_TP1_ONLY,
    # v27.9.5: Phase 2 config
    SIGNAL_GRADING_ENABLED, GRADE_A_THRESHOLD, GRADE_B_THRESHOLD, 
    GRADE_C_THRESHOLD, GRADE_D_THRESHOLD, EXECUTABLE_GRADES, GRADE_SIZE_MULT,
    STRUCTURE_DETECTION_ENABLED, STRUCTURE_SWING_LOOKBACK,
    PSYCHOLOGY_ENABLED, FG_EXTREME_FEAR, FG_EXTREME_GREED, 
    LS_CROWDED_LONG, LS_CROWDED_SHORT
)
from bot.models import HISTORICAL_DATA, load_stats, save_stats_async
from bot.utils import get_clean_symbol, format_price, send_throttled, extract_factors_from_reason

# Try importing optional modules
try:
    from bot.data_fetcher import exchange
except ImportError:
    exchange = None


# ============================================================================
# v27.12.15: PROPER LEVERAGED PNL CALCULATION
# ============================================================================

def calculate_leveraged_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    leverage: int = 3,
    position_pct: float = 1.0,
    fee_pct: float = 0.06
) -> dict:
    """
    Calculate proper leveraged PnL for perpetual futures.
    
    Rules:
    - LONG: Profit % = ((exit - entry) / entry) × leverage × position_pct
    - SHORT: Profit % = ((entry - exit) / entry) × leverage × position_pct
    - Fees applied on entry and exit (2x fee_pct)
    
    Args:
        direction: 'Long' or 'Short'
        entry_price: Entry price
        exit_price: Exit price  
        leverage: Leverage multiplier
        position_pct: Position size as decimal (1.0 = 100%, 0.5 = 50%)
        fee_pct: Fee percentage per transaction
    
    Returns:
        Dict with gross_pnl_pct, fees_pct, net_pnl_pct
    """
    # Calculate raw price move percentage
    if direction.lower() == 'long':
        raw_move_pct = (exit_price - entry_price) / entry_price * 100
    else:
        raw_move_pct = (entry_price - exit_price) / entry_price * 100
    
    # Apply leverage and position size
    leveraged_pnl_pct = raw_move_pct * leverage * position_pct
    
    # Fees on entry + exit (as % of position value, scaled by leverage)
    # For leveraged position, fee is on notional value
    total_fees_pct = fee_pct * 2 * position_pct * leverage
    
    # Net PnL
    net_pnl_pct = leveraged_pnl_pct - total_fees_pct
    
    return {
        'raw_move_pct': raw_move_pct,
        'leveraged_pnl_pct': leveraged_pnl_pct,
        'fees_pct': total_fees_pct,
        'net_pnl_pct': net_pnl_pct,
        'direction': direction,
        'leverage': leverage,
        'position_pct': position_pct
    }


def format_partial_close_msg(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    leverage: int = 3,
    partial_pct: float = 0.5
) -> str:
    """
    Format message for partial close at TP1 with proper math.
    
    Shows:
    - Realized PnL on closed portion
    - Remaining position status
    - Risk profile after partial close
    """
    pnl = calculate_leveraged_pnl(
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        leverage=leverage,
        position_pct=partial_pct
    )
    
    clean_sym = symbol.replace('/USDT', '')
    closed_pct = int(partial_pct * 100)
    remaining_pct = int((1 - partial_pct) * 100)
    
    msg = f"**TP1 PARTIAL EXIT** {clean_sym}\n\n"
    msg += f"Closed {closed_pct}% → **{pnl['net_pnl_pct']:+.2f}%** realized\n"
    msg += f"Remaining {remaining_pct}% at breakeven SL (0R risk)\n"
    msg += f"Targeting TP2\n\n"
    msg += f"_Entry: {format_price(entry_price)} | Exit: {format_price(exit_price)} | {leverage}x_"
    
    return msg


# ============================================================================
# FILE PATHS (with fallback)
# ============================================================================
try:
    from bot.config import (
        MAJOR_SYMBOLS, L1_ALT_SYMBOLS, TRADE_LOG_FILE, FACTOR_PERFORMANCE_FILE,
        SYMBOLS, ZONE_LOOKBACK, TRAILING_STOP_ENABLED, TRAILING_STOP_ATR_MULT,
        TRAILING_ACTIVATION_PCT, TRADE_TIMEOUT_HOURS, PROTECT_AFTER_HOURS
    )
except ImportError:
    MAJOR_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
    L1_ALT_SYMBOLS = ['SOL/USDT', 'AVAX/USDT', 'ADA/USDT']
    TRADE_LOG_FILE = 'data/trades_log.csv'
    FACTOR_PERFORMANCE_FILE = 'data/factor_performance.json'
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT']
    ZONE_LOOKBACK = 100
    TRAILING_STOP_ENABLED = True
    TRAILING_STOP_ATR_MULT = 1.5
    TRAILING_ACTIVATION_PCT = 1.5
    TRADE_TIMEOUT_HOURS = 48
    PROTECT_AFTER_HOURS = 4


# ============================================================================
# v27.9.5: SIGNAL GRADING SYSTEM (BALANCED)
# ============================================================================

def grade_signal(
    ob_score: int = 50,
    confluence_count: int = 3,
    trend_aligned: bool = True,
    rr_ratio: float = 2.0,
    entry_distance_pct: float = 2.0,
    volatility_safe: bool = True,
    is_counter_trend: bool = False,
    claude_confidence: int = 65,
    momentum_aligned: bool = False,
    structure_aligned: bool = False,
    grok_agrees: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Grade a signal A/B/C/D/F.
    
    v27.9.5: Balanced scoring with improved normalization.
    Each factor contributes up to ~15-25 points to prevent any single factor
    from dominating the score.
    """
    score = 0
    factors = []
    
    # 1. Order Block Quality (0-20 points) - Most important
    if ob_score >= 80:
        score += 20
        factors.append("Strong OB")
    elif ob_score >= 60:
        score += 15
        factors.append("Good OB")
    elif ob_score >= 40:
        score += 10
    else:
        score += 5
    
    # 2. Confluence Count (0-20 points) - Critical for high WR
    confluence_points = min(confluence_count * 4, 20)
    score += confluence_points
    if confluence_count >= 5:
        factors.append(f"{confluence_count}+ confluence")
    elif confluence_count >= 4:
        factors.append(f"{confluence_count} confluence")
    
    # 3. Trend Alignment (0-15 points)
    if trend_aligned:
        score += 15
        factors.append("Trend aligned")
    elif is_counter_trend:
        score += 5
        factors.append("Counter-trend")
    
    # 4. Risk/Reward (0-15 points)
    if rr_ratio >= 3.0:
        score += 15
        factors.append(f"RR {rr_ratio:.1f}")
    elif rr_ratio >= 2.0:
        score += 12
        factors.append(f"RR {rr_ratio:.1f}")
    elif rr_ratio >= 1.5:
        score += 8
    else:
        score += 4
    
    # 5. Entry Distance (0-10 points)
    if entry_distance_pct <= 1.0:
        score += 10
        factors.append("Tight entry")
    elif entry_distance_pct <= 2.0:
        score += 8
    elif entry_distance_pct <= 3.0:
        score += 5
    else:
        score += 2
    
    # 6. Volatility Safety (0-5 points)
    if volatility_safe:
        score += 5
    
    # 7. Claude Confidence (0-10 points)
    if claude_confidence >= 85:
        score += 10
        factors.append("High AI conf")
    elif claude_confidence >= 75:
        score += 7
    elif claude_confidence >= 65:
        score += 5
    else:
        score += 2
    
    # 8. Momentum Alignment (0-5 points)
    if momentum_aligned:
        score += 5
        factors.append("Momentum")
    
    # 9. Structure Alignment (0-5 points) - BOS/CHoCH
    if structure_aligned:
        score += 5
        factors.append("Structure")
    
    # 10. Grok Opinion (bonus/penalty)
    if grok_agrees is True:
        score += 3
        factors.append("Grok agrees")
    elif grok_agrees is False:
        score -= 3
    
    # Determine grade
    if score >= GRADE_A_THRESHOLD:
        grade = 'A'
    elif score >= GRADE_B_THRESHOLD:
        grade = 'B'
    elif score >= GRADE_C_THRESHOLD:
        grade = 'C'
    elif score >= GRADE_D_THRESHOLD:
        grade = 'D'
    else:
        grade = 'F'
    
    # Get size multiplier
    size_mult = GRADE_SIZE_MULT.get(grade, 0.5)
    
    # Check if executable
    executable = grade in EXECUTABLE_GRADES
    
    return {
        'grade': grade,
        'score': score,
        'size_mult': size_mult,
        'executable': executable,
        'factors': factors,
        'breakdown': {
            'ob': ob_score,
            'confluence': confluence_count,
            'trend': trend_aligned,
            'rr': rr_ratio,
            'distance': entry_distance_pct,
            'volatility': volatility_safe,
            'counter_trend': is_counter_trend,
            'ai_confidence': claude_confidence,
            'momentum': momentum_aligned,
            'structure': structure_aligned,
            'grok': grok_agrees
        }
    }


# ============================================================================
# v27.9.5: STRUCTURE BREAK DETECTION (BOS/CHoCH)
# ============================================================================

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Dict[str, List[Dict]]:
    """Find swing highs and lows in price data."""
    if len(df) < lookback * 2:
        return {'swing_highs': [], 'swing_lows': []}
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        high_val = df['high'].iloc[i]
        is_swing_high = all(df['high'].iloc[i-j] <= high_val for j in range(1, lookback+1)) and \
                        all(df['high'].iloc[i+j] <= high_val for j in range(1, lookback+1))
        
        if is_swing_high:
            swing_highs.append({
                'index': i,
                'price': high_val,
                'date': df['date'].iloc[i] if 'date' in df.columns else i
            })
        
        # Check for swing low
        low_val = df['low'].iloc[i]
        is_swing_low = all(df['low'].iloc[i-j] >= low_val for j in range(1, lookback+1)) and \
                       all(df['low'].iloc[i+j] >= low_val for j in range(1, lookback+1))
        
        if is_swing_low:
            swing_lows.append({
                'index': i,
                'price': low_val,
                'date': df['date'].iloc[i] if 'date' in df.columns else i
            })
    
    return {'swing_highs': swing_highs, 'swing_lows': swing_lows}


def detect_structure_break(df: pd.DataFrame, lookback: int = None) -> Optional[Dict]:
    """
    Detect BOS (Break of Structure) or CHoCH (Change of Character).
    
    BOS: Price breaks previous swing in trend direction (continuation)
    CHoCH: Price breaks previous swing against trend (reversal signal)
    """
    if lookback is None:
        lookback = STRUCTURE_SWING_LOOKBACK
    
    if len(df) < 50:
        return None
    
    swings = find_swing_points(df, lookback=lookback)
    
    if len(swings['swing_highs']) < 2 or len(swings['swing_lows']) < 2:
        return None
    
    current_price = df['close'].iloc[-1]
    
    # Get recent swing points
    recent_highs = swings['swing_highs'][-3:]
    recent_lows = swings['swing_lows'][-3:]
    
    # Determine trend from swing structure
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        higher_highs = recent_highs[-1]['price'] > recent_highs[-2]['price']
        higher_lows = recent_lows[-1]['price'] > recent_lows[-2]['price']
        lower_highs = recent_highs[-1]['price'] < recent_highs[-2]['price']
        lower_lows = recent_lows[-1]['price'] < recent_lows[-2]['price']
        
        # Bullish structure
        if higher_highs and higher_lows:
            trend = 'bullish'
        # Bearish structure
        elif lower_highs and lower_lows:
            trend = 'bearish'
        else:
            trend = 'ranging'
    else:
        trend = 'unknown'
    
    # Check for structure break
    last_swing_high = recent_highs[-1]['price'] if recent_highs else None
    last_swing_low = recent_lows[-1]['price'] if recent_lows else None
    
    result = {
        'trend': trend,
        'current_price': current_price,
        'last_swing_high': last_swing_high,
        'last_swing_low': last_swing_low,
        'break_type': None,
        'signal': None,
        'strength': 0
    }
    
    # Detect breaks
    if last_swing_high and current_price > last_swing_high:
        if trend == 'bearish':
            result['break_type'] = 'CHoCH'
            result['signal'] = 'bullish'
            result['strength'] = 2
        else:
            result['break_type'] = 'BOS'
            result['signal'] = 'bullish'
            result['strength'] = 1
    
    elif last_swing_low and current_price < last_swing_low:
        if trend == 'bullish':
            result['break_type'] = 'CHoCH'
            result['signal'] = 'bearish'
            result['strength'] = 2
        else:
            result['break_type'] = 'BOS'
            result['signal'] = 'bearish'
            result['strength'] = 1
    
    return result


def is_structure_aligned(structure_break: Optional[Dict], direction: str) -> bool:
    """Check if structure break aligns with trade direction."""
    if not structure_break or not structure_break.get('signal'):
        return False
    
    signal = structure_break['signal'].lower()
    dir_lower = direction.lower()
    
    return (dir_lower == 'long' and signal == 'bullish') or \
           (dir_lower == 'short' and signal == 'bearish')


# ============================================================================
# v27.9.5: PSYCHOLOGY ANALYSIS
# ============================================================================

def analyze_psychology(fear_greed: Optional[Dict], long_short: Optional[Dict]) -> Dict:
    """
    Analyze market psychology for confluence.
    
    Returns signal boost/penalty based on crowd positioning.
    """
    result = {
        'signal': 'neutral',
        'boost': 0,
        'reasons': []
    }
    
    # Fear & Greed Index
    if fear_greed:
        fg_value = fear_greed.get('value', 50)
        
        if fg_value <= FG_EXTREME_FEAR:
            result['signal'] = 'long'
            result['boost'] += 3
            result['reasons'].append(f"Extreme fear ({fg_value})")
        elif fg_value >= FG_EXTREME_GREED:
            result['signal'] = 'short'
            result['boost'] += 3
            result['reasons'].append(f"Extreme greed ({fg_value})")
    
    # Long/Short Ratio
    if long_short:
        ratio = long_short.get('ratio', 1.0)
        
        if ratio >= LS_CROWDED_LONG:
            if result['signal'] == 'short':
                result['boost'] += 2
            elif result['signal'] == 'neutral':
                result['signal'] = 'short'
                result['boost'] += 2
            result['reasons'].append(f"Crowded long ({ratio:.2f})")
        
        elif ratio <= LS_CROWDED_SHORT:
            if result['signal'] == 'long':
                result['boost'] += 2
            elif result['signal'] == 'neutral':
                result['signal'] = 'long'
                result['boost'] += 2
            result['reasons'].append(f"Crowded short ({ratio:.2f})")
    
    return result


async def get_fear_greed() -> Optional[Dict]:
    """Fetch Fear & Greed Index."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1")
            data = r.json()
            if data.get('data'):
                return {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification']
                }
    except Exception as e:
        logging.debug(f"Fear/Greed fetch failed: {e}")
    return None


async def get_long_short_ratio(symbol: str) -> Optional[Dict]:
    """Fetch Long/Short ratio from Bybit."""
    if exchange is None:
        return None
    
    try:
        # Use Bybit API for L/S ratio
        clean = symbol.replace('/USDT', '').replace('/', '')
        
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={clean}USDT&period=1h&limit=1"
            r = await client.get(url)
            data = r.json()
            
            if data.get('result', {}).get('list'):
                item = data['result']['list'][0]
                buy_ratio = float(item.get('buyRatio', 0.5))
                sell_ratio = float(item.get('sellRatio', 0.5))
                
                if sell_ratio > 0:
                    ratio = buy_ratio / sell_ratio
                else:
                    ratio = 1.0
                
                return {
                    'ratio': ratio,
                    'buy_pct': buy_ratio * 100,
                    'sell_pct': sell_ratio * 100
                }
    except Exception as e:
        logging.debug(f"L/S ratio fetch failed for {symbol}: {e}")
    
    return None


async def get_psychology_boost(symbol: str, direction: str) -> Tuple[int, str]:
    """
    Get psychology-based confidence boost.
    
    Returns (boost: int, reason: str)
    """
    if not PSYCHOLOGY_ENABLED:
        return 0, ""
    
    fg = await get_fear_greed()
    ls = await get_long_short_ratio(symbol)
    psych = analyze_psychology(fg, ls)
    
    if psych['signal'] == 'neutral':
        return 0, ""
    
    dir_lower = direction.lower()
    if (dir_lower == 'long' and psych['signal'] == 'long') or \
       (dir_lower == 'short' and psych['signal'] == 'short'):
        return psych['boost'], f"+Psychology({','.join(psych['reasons'][:2])})"
    return -2, f"-Psychology({psych['signal']})"


# ============================================================================
# FACTOR TRACKER
# ============================================================================

class FactorTracker:
    def __init__(self):
        self.data = self._load()
    
    def _load(self) -> Dict[str, Dict]:
        if os.path.exists(FACTOR_PERFORMANCE_FILE):
            try:
                with open(FACTOR_PERFORMANCE_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    async def _save(self):
        try:
            async with aiofiles.open(FACTOR_PERFORMANCE_FILE, 'w') as f:
                await f.write(json.dumps(self.data, indent=2))
        except Exception as e:
            logging.error(f"Failed to save factor performance: {e}")
    
    async def record_trade(self, factors: List[str], result: str, pnl_pct: float):
        for factor in factors:
            if factor not in self.data:
                self.data[factor] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': 0, 'avg_pnl': 0.0, 'win_rate': 0.0}
            
            self.data[factor]['trades'] += 1
            self.data[factor]['total_pnl'] += pnl_pct
            
            if result == 'WIN':
                self.data[factor]['wins'] += 1
            else:
                self.data[factor]['losses'] += 1
            
            total = self.data[factor]['trades']
            self.data[factor]['avg_pnl'] = self.data[factor]['total_pnl'] / total
            self.data[factor]['win_rate'] = self.data[factor]['wins'] / total * 100
        
        await self._save()
    
    def get_factor_stats(self, factor: str) -> Optional[Dict]:
        return self.data.get(factor)
    
    def get_best_factors(self, min_trades: int = 5) -> List[Tuple[str, float]]:
        qualified = [(f, d['win_rate']) for f, d in self.data.items() if d['trades'] >= min_trades]
        return sorted(qualified, key=lambda x: x[1], reverse=True)
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get all factor statistics."""
        return self.data


factor_tracker = FactorTracker()


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

def get_risk_scaling_factor(stats: Dict) -> float:
    current_dd = max(0, -stats.get('pnl', 0))
    
    if current_dd < 1:
        return 1.0
    elif current_dd < 2:
        return 0.75
    elif current_dd < 3:
        return 0.50
    else:
        return 0.0


def calculate_expected_value(trade: Dict, historical_data: Dict) -> float:
    entry_low = trade.get('entry_low', 0)
    entry_high = trade.get('entry_high', 0)
    entry_mid = (entry_low + entry_high) / 2
    
    if entry_mid == 0:
        return 0
    
    prob_hit_tp1 = historical_data.get('tp1_hit_rate', 0.60)
    prob_hit_tp2 = historical_data.get('tp2_hit_rate', 0.35)
    prob_hit_sl = 1 - prob_hit_tp1
    
    tp1 = trade.get('tp1', entry_mid)
    tp2 = trade.get('tp2', entry_mid)
    sl = trade.get('sl', entry_mid)
    
    if trade.get('direction') == 'Long':
        tp1_gain = (tp1 - entry_mid) * (1 - SLIPPAGE_PCT * 2)
        tp2_gain = (tp2 - entry_mid) * (1 - SLIPPAGE_PCT * 2)
        sl_loss = (entry_mid - sl) * (1 + SLIPPAGE_PCT * 2)
    else:
        tp1_gain = (entry_mid - tp1) * (1 - SLIPPAGE_PCT * 2)
        tp2_gain = (entry_mid - tp2) * (1 - SLIPPAGE_PCT * 2)
        sl_loss = (sl - entry_mid) * (1 + SLIPPAGE_PCT * 2)
    
    expected_gain = prob_hit_tp1 * tp1_gain * 0.5 + prob_hit_tp2 * tp2_gain * 0.5
    expected_loss = prob_hit_sl * sl_loss
    ev = expected_gain - expected_loss
    
    return ev / abs(sl_loss) if sl_loss != 0 else 0


def check_portfolio_correlation(symbol: str, open_trades: Dict, protected_trades: Dict) -> bool:
    active_symbols = [get_clean_symbol(k) for trades_dict in [open_trades, protected_trades] 
                      for k, t in trades_dict.items() if t.get('active')]
    
    if not active_symbols:
        return True
    
    if symbol in MAJOR_SYMBOLS and any(s in MAJOR_SYMBOLS for s in active_symbols):
        return False
    
    if symbol in L1_ALT_SYMBOLS:
        l1_count = sum(1 for s in active_symbols if s in L1_ALT_SYMBOLS)
        if l1_count >= 2:
            return False
    
    if symbol in active_symbols:
        return False
    
    return True


async def check_sufficient_liquidity(symbol: str) -> bool:
    """Check if there's sufficient liquidity for the trade."""
    if exchange is None:
        return True
    
    try:
        book = await exchange.fetch_order_book(symbol, limit=20)
        
        if not book.get('bids') or not book.get('asks'):
            return False
        
        best_bid = book['bids'][0][0]
        best_ask = book['asks'][0][0]
        spread_pct = (best_ask - best_bid) / best_bid * 100
        
        if spread_pct > 0.05:
            return False
        
        total_bid_vol = sum(amt for _, amt in book['bids'][:10])
        total_ask_vol = sum(amt for _, amt in book['asks'][:10])
        
        min_depth_usd = 5 * 60000 if 'SOL' in symbol or 'ETH' in symbol else 10 * 60000
        
        if total_bid_vol * best_bid < min_depth_usd or total_ask_vol * best_ask < min_depth_usd:
            return False
        
        return True
        
    except Exception as e:
        logging.warning(f"Liquidity check failed for {symbol}: {e}")
        return True


def is_in_no_trade_zone(price: float, symbol: str, df: pd.DataFrame) -> bool:
    """Check if price is near psychological levels or weekly close."""
    if price % 100 < 5 or price % 100 > 95:
        return True
    
    if price % 1000 < 50 or price % 1000 > 950:
        return True
    
    return False


def calculate_dynamic_threshold(df: pd.DataFrame, base: float = 60, stats: Dict = None) -> float:
    """Dynamically adjust confidence threshold based on market conditions."""
    if len(df) == 0 or 'atr' not in df.columns:
        return base
    
    atr_percentile = df['atr'].rank(pct=True).iloc[-1]
    vol_percentile = df['volume'].rank(pct=True).iloc[-1]
    
    adjustments = 0
    
    if atr_percentile > 0.8:
        adjustments += 10
    
    if vol_percentile < 0.3:
        adjustments += 8
    
    return min(base + adjustments, 95)


def calculate_trailing_stop(trade: Dict, current_price: float, atr: float) -> Optional[float]:
    """Calculate trailing stop level."""
    if not TRAILING_STOP_ENABLED or not trade.get('tp1_exited'):
        return None
    
    entry_price = trade.get('entry_price', 0)
    if entry_price == 0:
        return None
    
    # v27.11.0: Safe fallback for trailing_sl
    current_trail = trade.get('trailing_sl') or trade.get('sl')
    if current_trail is None:
        return None
    
    direction = trade.get('direction', 'Long')
    
    if direction == 'Long':
        profit_pct = (current_price - entry_price) / entry_price * 100
    else:
        profit_pct = (entry_price - current_price) / entry_price * 100
    
    if profit_pct < TRAILING_ACTIVATION_PCT:
        return None
    
    trail_distance = atr * TRAILING_STOP_ATR_MULT
    
    if direction == 'Long':
        new_trail = current_price - trail_distance
        if new_trail > current_trail:
            return new_trail
    else:
        new_trail = current_price + trail_distance
        if new_trail < current_trail:
            return new_trail
    
    return None


# ============================================================================
# MAIN TRADE PROCESSING - v27.11.0 with None Safety
# ============================================================================

async def process_trade(
    trades: Dict[str, Any], 
    to_delete: List[str], 
    now: datetime, 
    current_capital: float, 
    prices: Dict[str, Optional[float]], 
    updated_keys: List[str], 
    is_protected: bool = False, 
    stats_lock: asyncio.Lock = None, 
    stats: Dict = None, 
    atr_values: Dict[str, float] = None
):
    """
    Process active trades - check entries, TPs, SLs.
    
    v27.11.0: Added comprehensive None safety checks to prevent
    "'>=' not supported between 'float' and 'NoneType'" errors.
    """
    from bot.models import save_trades_async, save_protected_async
    
    risk_scale = get_risk_scaling_factor(stats) if stats else 1.0
    if risk_scale == 0:
        return
    
    if atr_values is None:
        atr_values = {}
    
    for trade_key, trade in list(trades.items()):
        try:
            clean_symbol = get_clean_symbol(trade_key)
            
            # v27.11.0: Validate required trade fields
            sl = trade.get('sl')
            tp1 = trade.get('tp1')
            tp2 = trade.get('tp2')
            
            if sl is None or tp1 is None:
                logging.warning(f"Trade {trade_key} missing SL or TP1, skipping")
                continue
            
            # Timeout check
            if 'last_check' in trade and trade['last_check']:
                try:
                    last_check = trade['last_check']
                    if isinstance(last_check, str):
                        last_check = datetime.fromisoformat(last_check)
                    time_diff = now - last_check if isinstance(last_check, datetime) else timedelta(0)
                    if time_diff > timedelta(hours=TRADE_TIMEOUT_HOURS):
                        await send_throttled(CHAT_ID, f"**TIMEOUT** {clean_symbol.replace('/USDT','')}", parse_mode='Markdown')
                        to_delete.append(trade_key)
                        continue
                except Exception as e:
                    logging.debug(f"Timeout check error for {trade_key}: {e}")
            
            price = prices.get(clean_symbol)
            if price is None:
                trade['last_check'] = now
                updated_keys.append(trade_key)
                continue
            
            trade['last_check'] = now
            updated_keys.append(trade_key)
            
            # Entry activation
            if not trade.get('active', False):
                entry_low = trade.get('entry_low')
                entry_high = trade.get('entry_high')
                
                if entry_low is None or entry_high is None:
                    logging.warning(f"Trade {trade_key} missing entry zone, skipping")
                    continue
                
                direction = trade.get('direction', 'Long')
                
                if direction == 'Long':
                    extended_high = entry_high * (1 + ENTRY_SLIPPAGE_PCT)
                    in_zone = entry_low <= price <= extended_high
                else:
                    extended_low = entry_low * (1 - ENTRY_SLIPPAGE_PCT)
                    in_zone = extended_low <= price <= entry_high
                
                if in_zone:
                    trade['active'] = True
                    slippage = SLIPPAGE_PCT * price
                    trade['entry_price'] = price + slippage if direction == 'Long' else price - slippage
                    
                    if 'entry_time' not in trade:
                        trade['entry_time'] = now
                    
                    risk_amount = current_capital * RISK_PER_TRADE_PCT / 100
                    risk_distance = abs(price - sl)
                    trade['position_size'] = risk_amount / risk_distance if risk_distance > 0 else 0
                    
                    # Apply grade-based position sizing
                    grade_mult = trade.get('size_mult', 1.0)
                    trade['position_size'] *= grade_mult
                    
                    reason = trade.get('reason', '')
                    trade['factors'] = extract_factors_from_reason(reason)
                    
                    ev_r = calculate_expected_value(trade, HISTORICAL_DATA)
                    
                    tag = '(*roadmap*)' if trade.get('type') == 'roadmap' else ('(*protected*)' if is_protected else '')
                    grade_display = trade.get('grade_display', '')
                    
                    await send_throttled(
                        CHAT_ID,
                        f"**ENTRY ACTIVATED** {tag} {grade_display}\n\n"
                        f"**{clean_symbol.replace('/USDT','')} {direction}** @ {format_price(price)}\n"
                        f"*SL* {format_price(sl)} | *TP1* {format_price(tp1)} | *TP2* {format_price(tp2) if tp2 else 'N/A'} | {trade.get('leverage', 3)}x\n"
                        f"EV: {ev_r:.2f}R",
                        parse_mode='Markdown'
                    )
            
            # Active trade management
            if trade.get('active'):
                entry_price = trade.get('entry_price')
                if entry_price is None:
                    logging.warning(f"Active trade {trade_key} has no entry_price, skipping")
                    continue
                
                size = trade.get('position_size', 1)
                direction = trade.get('direction', 'Long')
                
                # v27.11.0: Safe SL retrieval with None check
                current_sl = trade.get('trailing_sl') or trade.get('sl')
                if current_sl is None:
                    logging.warning(f"Trade {trade_key} has no SL defined, skipping")
                    continue
                
                # v27.9.0: Check if this is counter-trend trade with TP1-only strategy
                use_tp2 = trade.get('use_tp2', True)
                
                # Update trailing stop
                if trade.get('tp1_exited') and clean_symbol in atr_values:
                    atr = atr_values[clean_symbol]
                    new_trail = calculate_trailing_stop(trade, price, atr)
                    if new_trail is not None:
                        trade['trailing_sl'] = new_trail
                        current_sl = new_trail
                
                # TP1 check - v27.11.0: use validated tp1 variable
                hit_tp1 = (direction == 'Long' and price >= tp1) or \
                          (direction == 'Short' and price <= tp1)
                
                if hit_tp1 and current_sl == sl and not trade.get('tp1_exited'):
                    # v27.9.0: For counter-trend, close FULL position at TP1
                    if COUNTER_TREND_TP1_ONLY and not use_tp2:
                        # Close entire position
                        # v27.12.15: Proper leveraged PnL calculation
                        leverage = trade.get('leverage', 3)
                        pnl_result = calculate_leveraged_pnl(
                            direction=direction,
                            entry_price=entry_price,
                            exit_price=price,
                            leverage=leverage,
                            position_pct=1.0,  # Full close
                            fee_pct=FEE_PCT
                        )
                        
                        diff = (price - entry_price) if direction == 'Long' else (entry_price - price)
                        tp1_pnl_usdt = diff * size
                        fee_usdt = FEE_PCT * (entry_price * size)
                        net_tp1_pnl_usdt = tp1_pnl_usdt - fee_usdt
                        
                        # Use leveraged PnL for display
                        net_tp1_pnl_pct = pnl_result['net_pnl_pct']
                        
                        trade['position_size'] = 0
                        trade['tp1_exited'] = True
                        trade['tp1_pnl'] = net_tp1_pnl_pct
                        trade['processed'] = True
                        trade['hit_tp'] = True
                        
                        if stats_lock and stats:
                            async with stats_lock:
                                stats['capital'] += net_tp1_pnl_usdt
                                stats['pnl'] = (stats['capital'] - SIMULATED_CAPITAL) / SIMULATED_CAPITAL * 100
                                stats['tp1_hits'] = stats.get('tp1_hits', 0) + 1
                                stats['wins'] = stats.get('wins', 0) + 1
                                await save_stats_async(stats)
                        
                        await send_throttled(
                            CHAT_ID,
                            f"**TP1 FULL EXIT** {clean_symbol.replace('/USDT','')} *(Counter-Trend)*\n\n"
                            f"Closed 100% @ {format_price(price)} → **{net_tp1_pnl_pct:+.2f}%** realized\n"
                            f"_Entry: {format_price(entry_price)} | {leverage}x leverage_",
                            parse_mode='Markdown'
                        )
                        
                        to_delete.append(trade_key)
                        continue
                    
                    else:
                        # Standard partial exit (trend-following)
                        # v27.12.15: Proper leveraged PnL calculation
                        leverage = trade.get('leverage', 3)
                        pnl_result = calculate_leveraged_pnl(
                            direction=direction,
                            entry_price=entry_price,
                            exit_price=price,
                            leverage=leverage,
                            position_pct=0.5,  # 50% partial close
                            fee_pct=FEE_PCT
                        )
                        
                        size_to_close = size * 0.5
                        diff = (price - entry_price) if direction == 'Long' else (entry_price - price)
                        tp1_pnl_usdt = diff * size_to_close
                        fee_usdt = FEE_PCT * (entry_price * size_to_close)
                        net_tp1_pnl_usdt = tp1_pnl_usdt - fee_usdt
                        
                        # Use leveraged PnL for display
                        net_tp1_pnl_pct = pnl_result['net_pnl_pct']
                        
                        trade['position_size'] = size * 0.5
                        trade['trailing_sl'] = entry_price
                        trade['tp1_exited'] = True
                        trade['tp1_pnl'] = net_tp1_pnl_pct
                        
                        if stats_lock and stats:
                            async with stats_lock:
                                stats['capital'] += net_tp1_pnl_usdt
                                stats['pnl'] = (stats['capital'] - SIMULATED_CAPITAL) / SIMULATED_CAPITAL * 100
                                stats['tp1_hits'] = stats.get('tp1_hits', 0) + 1
                                await save_stats_async(stats)
                        
                        # v27.12.15: Improved partial close message
                        msg = format_partial_close_msg(
                            symbol=trade_key,
                            direction=direction,
                            entry_price=entry_price,
                            exit_price=price,
                            leverage=leverage,
                            partial_pct=0.5
                        )
                        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
                
                # TP2/SL/Trailing check - v27.11.0: Safe None checks
                hit_tp = False
                if use_tp2 and tp2 is not None:
                    hit_tp = (price >= tp2 if direction == 'Long' else price <= tp2)
                
                # v27.11.0: current_sl already validated above, safe to compare
                hit_sl = (price <= current_sl if direction == 'Long' else price >= current_sl)
                
                hit_trail = False
                trailing_sl = trade.get('trailing_sl')
                if trade.get('tp1_exited') and trailing_sl is not None:
                    if direction == 'Long':
                        hit_trail = price <= trailing_sl
                    else:
                        hit_trail = price >= trailing_sl
                
                if hit_tp or hit_sl or hit_trail:
                    if trade.get('processed'):
                        continue
                    
                    trade['processed'] = True
                    trade['hit_tp'] = hit_tp
                    
                    current_size = trade.get('position_size', size)
                    diff = (price - entry_price) if direction == 'Long' else (entry_price - price)
                    pnl_usdt = diff * current_size
                    fee_usdt = FEE_PCT * (entry_price * current_size)
                    net_pnl_usdt = pnl_usdt - fee_usdt
                    
                    if trade.get('tp1_exited'):
                        tp1_pnl_pct = trade.get('tp1_pnl', 0)
                        tp1_pnl_usdt = current_capital * (tp1_pnl_pct / 100) if current_capital > 0 else 0
                        net_pnl_usdt += tp1_pnl_usdt
                    
                    net_pnl_pct = net_pnl_usdt / current_capital * 100 if current_capital > 0 else 0
                    
                    if hit_tp:
                        result = "WIN"
                        exit_type = "TP2"
                    elif hit_trail:
                        result = "WIN" if net_pnl_pct > 0 else "LOSS"
                        exit_type = "TRAIL"
                    else:
                        result = "LOSS"
                        exit_type = "SL"
                    
                    if stats:
                        if net_pnl_pct > stats.get('best_trade', 0):
                            stats['best_trade'] = net_pnl_pct
                        if net_pnl_pct < stats.get('worst_trade', 0):
                            stats['worst_trade'] = net_pnl_pct
                    
                    tag = '(*roadmap*)' if trade.get('type') == 'roadmap' else ('(*protected*)' if is_protected else '')
                    
                    msg = f"**{result}** {clean_symbol.replace('/USDT','')} {tag}"
                    if exit_type == "TRAIL":
                        msg += " *(trailing stop)*"
                    msg += f"\n\n{net_pnl_pct:+.2f}% @ {trade.get('leverage', 3)}x **[fees adj]**"
                    
                    await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
                    
                    factors = trade.get('factors', [])
                    if factors:
                        await factor_tracker.record_trade(factors, result, net_pnl_pct)
                    
                    if stats_lock and stats:
                        async with stats_lock:
                            delta_capital = stats['capital'] * (net_pnl_pct / 100)
                            stats['capital'] += delta_capital
                            stats['pnl'] = (stats['capital'] - SIMULATED_CAPITAL) / SIMULATED_CAPITAL * 100
                            stats['wins' if result == 'WIN' else 'losses'] = stats.get('wins' if result == 'WIN' else 'losses', 0) + 1
                            await save_stats_async(stats)
                    
                    # Log trade
                    trade_log = {
                        'timestamp': now.isoformat(),
                        'symbol': clean_symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'exit_type': exit_type,
                        'result': result,
                        'pnl_pct': net_pnl_pct,
                        'grade': trade.get('grade', 'N/A'),
                        'from_roadmap': trade.get('from_roadmap', False)
                    }
                    
                    try:
                        file_exists = Path(TRADE_LOG_FILE).exists()
                        async with aiofiles.open(TRADE_LOG_FILE, 'a', newline='') as f:
                            if not file_exists:
                                await f.write(','.join(trade_log.keys()) + '\n')
                            await f.write(','.join(str(v) for v in trade_log.values()) + '\n')
                    except Exception as e:
                        logging.debug(f"Failed to log trade: {e}")
                    
                    to_delete.append(trade_key)
        
        except Exception as e:
            logging.error(f"Error processing trade {trade_key}: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            continue


async def get_atr_values(symbols: List[str], data_cache: Dict) -> Dict[str, float]:
    """Get ATR values for all symbols from cache."""
    atr_values = {}
    
    for symbol in symbols:
        # Clean the symbol key
        clean = get_clean_symbol(symbol)
        df = data_cache.get(clean, {}).get('1d')
        if df is not None and len(df) > 0 and 'atr' in df.columns:
            atr_val = df['atr'].iloc[-1]
            if not pd.isna(atr_val):
                atr_values[clean] = atr_val
    
    return atr_values
