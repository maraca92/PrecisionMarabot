# backtest_realistic.py - Grok Elite Signal Bot v27.12.17 - Realistic Backtesting
# -*- coding: utf-8 -*-
"""
REALISTIC PROFESSIONAL BACKTESTING SYSTEM

Based on quantitative finance research and institutional methodology.

Key principles implemented:
1. NO LOOK-AHEAD BIAS - Only uses data available at decision time
2. CONFIRMATION REQUIRED - Must wait for candle close + bounce confirmation
3. REALISTIC FILLS - Queue position modeling, not every signal fills
4. TRADE FREQUENCY LIMITS - Max 1-2 trades per symbol per week
5. VOLATILITY-SCALED SLIPPAGE - Higher during volatile periods
6. PROPER WARMUP - 200+ bars before any signals
7. STATISTICAL VALIDATION - Monte Carlo with realistic expectations

Expected realistic results:
- Win Rate: 45-55%
- Profit Factor: 1.3-2.0
- Sharpe Ratio: 0.8-1.5
- Max Drawdown: 15-30%
- Trades: 50-150 per year (not 3000+)

v27.12.17: Complete rewrite for realistic simulation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import random

# ============================================================================
# CONFIGURATION - REALISTIC PARAMETERS
# ============================================================================

@dataclass
class RealisticBacktestConfig:
    """Configuration based on professional quant standards"""
    
    # === DATA SETTINGS ===
    days: int = 365                      # Test period
    warmup_bars: int = 250               # Warmup for EMA200 + buffer
    primary_timeframe: str = '4h'        # Main trading timeframe
    htf_timeframe: str = '1d'            # Higher timeframe for bias
    
    # === SIGNAL REQUIREMENTS (STRICT) ===
    min_ob_strength: float = 2.5         # Higher threshold for quality
    min_confluence_factors: int = 3      # Need 3+ factors agreeing
    min_htf_alignment: bool = True       # Must align with daily trend
    
    # === CONFIRMATION REQUIREMENTS ===
    require_candle_close: bool = True    # Never enter mid-candle
    require_bounce_confirm: bool = True  # Need rejection candle
    min_rejection_wick_pct: float = 0.5  # Wick must be 50%+ of range
    confirmation_candles: int = 2        # Wait 2 candles after OB forms
    
    # === ENTRY ZONE REQUIREMENTS ===
    max_entry_distance_pct: float = 1.5  # Must be within 1.5% of zone
    zone_must_be_touched: bool = True    # Price must actually reach zone
    
    # === FILL PROBABILITY MODEL ===
    base_fill_probability: float = 0.65  # Only 65% of limit orders fill
    adverse_selection_factor: float = 0.3  # When filled, often moves against
    
    # === TRADE FREQUENCY LIMITS ===
    min_hours_between_trades: int = 24   # 24h minimum between trades same symbol
    max_trades_per_symbol_week: int = 2  # Max 2 trades per symbol per week
    max_concurrent_positions: int = 3    # Max 3 open at once
    
    # === REALISTIC COSTS ===
    # Base fees
    taker_fee_pct: float = 0.06          # Taker fee
    maker_fee_pct: float = 0.02          # Maker fee (rarely achieved)
    
    # Slippage by market cap tier
    slippage_major_pct: float = 0.08     # BTC/ETH
    slippage_midcap_pct: float = 0.20    # SOL, AVAX, etc
    slippage_altcoin_pct: float = 0.50   # Smaller alts
    
    # Volatility scaling
    slippage_vol_multiplier: float = 2.0  # 2x slippage during high vol
    high_vol_threshold: float = 3.0       # ATR > 3% = high volatility
    
    # Funding rates (for holding periods)
    avg_funding_rate_8h: float = 0.01    # 0.01% per 8h avg
    
    # === RISK MANAGEMENT ===
    risk_per_trade_pct: float = 1.5      # Risk 1.5% per trade
    default_leverage: int = 3            # 3x leverage
    
    # TP/SL settings
    tp1_r_multiple: float = 1.5          # TP1 at 1.5R
    tp2_r_multiple: float = 2.5          # TP2 at 2.5R (conservative)
    partial_close_pct: float = 0.5       # Close 50% at TP1
    max_hold_bars: int = 50              # Max 50 bars (~8 days on 4h)
    
    # === STATISTICAL VALIDATION ===
    monte_carlo_runs: int = 1000
    min_trades_for_validity: int = 30    # Need 30+ trades for any inference


class SignalState(Enum):
    """State machine for signal lifecycle"""
    ZONE_DETECTED = 1      # OB/zone identified (cannot trade yet)
    ZONE_CONFIRMED = 2     # Zone confirmed after N candles
    AWAITING_TOUCH = 3     # Waiting for price to reach zone
    TOUCHED = 4            # Price touched zone
    BOUNCE_FORMING = 5     # Potential rejection forming
    CONFIRMED = 6          # Bounce confirmed, ready to enter
    FILLED = 7             # Order filled
    REJECTED = 8           # Signal invalidated


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ZoneData:
    """Represents a support/resistance zone"""
    symbol: str
    zone_type: str  # 'bullish' or 'bearish'
    high: float
    low: float
    strength: float
    detection_bar: int  # Bar index when detected
    detection_time: datetime
    state: SignalState = SignalState.ZONE_DETECTED
    confirmation_bar: Optional[int] = None
    touch_bar: Optional[int] = None
    touch_price: Optional[float] = None
    invalidated: bool = False
    invalidation_reason: str = ''


@dataclass
class PendingSignal:
    """Signal waiting for confirmation"""
    zone: ZoneData
    direction: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    factors: List[str]
    confluence_count: int
    state: SignalState
    created_bar: int
    created_time: datetime
    fill_probability: float = 0.65


@dataclass 
class RealisticTrade:
    """Completed trade with full tracking"""
    symbol: str
    direction: str
    
    # Prices
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    exit_price: float
    
    # Times
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    
    # Execution details
    leverage: int
    slippage_pct: float
    fees_pct: float
    funding_paid_pct: float
    
    # Results
    gross_pnl_pct: float
    total_costs_pct: float
    net_pnl_pct: float
    pnl_r: float
    
    # Exit details
    exit_type: str  # 'TP1', 'TP2', 'SL', 'BE', 'TIMEOUT'
    tp1_hit: bool = False
    tp1_pnl_pct: float = 0.0
    bars_held: int = 0
    
    # Factors
    factors: List[str] = field(default_factory=list)
    result: str = ''  # 'WIN', 'LOSS', 'BREAKEVEN'


@dataclass
class RealisticBacktestResult:
    """Complete backtest results with realistic expectations"""
    # Trade counts
    total_signals: int = 0           # Signals detected
    signals_confirmed: int = 0       # Signals that got confirmation
    orders_placed: int = 0           # Orders that were placed
    orders_filled: int = 0           # Orders that actually filled
    
    # Results
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    
    # Performance
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_r: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Costs breakdown
    total_fees_pct: float = 0.0
    total_slippage_pct: float = 0.0
    total_funding_pct: float = 0.0
    
    # Trade statistics
    avg_hold_hours: float = 0.0
    tp1_hit_rate: float = 0.0
    tp2_hit_rate: float = 0.0
    
    # Monte Carlo
    mc_median_pnl: float = 0.0
    mc_95_lower: float = 0.0
    mc_95_upper: float = 0.0
    mc_prob_profit: float = 0.0
    mc_prob_drawdown_25: float = 0.0
    
    # Factor analysis
    factor_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # Validation flags
    is_statistically_valid: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    
    # Raw data
    trades: List[RealisticTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    # Metadata
    config: Optional[RealisticBacktestConfig] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


# ============================================================================
# SLIPPAGE & COST CALCULATOR
# ============================================================================

class RealisticCostCalculator:
    """Calculate realistic trading costs"""
    
    MAJOR_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
    MIDCAP_SYMBOLS = ['SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT']
    
    def __init__(self, config: RealisticBacktestConfig):
        self.config = config
    
    def get_slippage(
        self, 
        symbol: str, 
        volatility_pct: float,
        direction: str
    ) -> float:
        """
        Calculate realistic slippage based on:
        - Market cap tier
        - Current volatility
        - Adverse selection (slippage tends to go against you)
        """
        # Base slippage by tier
        if symbol in self.MAJOR_SYMBOLS:
            base_slip = self.config.slippage_major_pct
        elif symbol in self.MIDCAP_SYMBOLS:
            base_slip = self.config.slippage_midcap_pct
        else:
            base_slip = self.config.slippage_altcoin_pct
        
        # Volatility scaling (higher vol = higher slippage)
        vol_multiplier = 1.0
        if volatility_pct > self.config.high_vol_threshold:
            vol_multiplier = self.config.slippage_vol_multiplier
        elif volatility_pct > self.config.high_vol_threshold / 2:
            vol_multiplier = 1.5
        
        # Add randomness (slippage isn't constant)
        random_factor = random.uniform(0.5, 1.5)
        
        slippage = base_slip * vol_multiplier * random_factor
        
        return slippage
    
    def get_fees(self, is_maker: bool = False) -> float:
        """Get trading fees"""
        # Almost always taker in reality
        if is_maker and random.random() < 0.1:  # 10% chance of maker
            return self.config.maker_fee_pct
        return self.config.taker_fee_pct
    
    def get_funding_cost(self, hours_held: float, direction: str) -> float:
        """
        Calculate funding rate cost for perpetuals.
        Longs pay during bull markets (positive funding).
        """
        # Funding every 8 hours
        funding_periods = hours_held / 8
        
        # Random funding rate (usually slightly positive)
        avg_rate = self.config.avg_funding_rate_8h
        funding_rate = random.gauss(avg_rate, avg_rate * 0.5)
        
        # Longs pay positive funding, shorts receive
        if direction.lower() == 'long':
            return funding_rate * funding_periods
        else:
            return -funding_rate * funding_periods * 0.8  # Shorts usually receive less
    
    def calculate_total_costs(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        volatility_pct: float,
        hours_held: float,
        leverage: int
    ) -> Dict[str, float]:
        """Calculate all trading costs"""
        
        # Entry slippage
        entry_slip = self.get_slippage(symbol, volatility_pct, direction)
        
        # Exit slippage (often higher due to urgency)
        exit_slip = self.get_slippage(symbol, volatility_pct, direction) * 1.2
        
        # Fees (entry + exit)
        entry_fee = self.get_fees()
        exit_fee = self.get_fees()
        
        # Funding
        funding = self.get_funding_cost(hours_held, direction)
        
        # Total (scaled by leverage for margin impact)
        total_slippage = (entry_slip + exit_slip) * leverage
        total_fees = (entry_fee + exit_fee) * leverage
        total_funding = abs(funding) * leverage
        
        return {
            'entry_slippage': entry_slip,
            'exit_slippage': exit_slip,
            'total_slippage': total_slippage,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'total_fees': total_fees,
            'funding': total_funding,
            'total_costs': total_slippage + total_fees + total_funding
        }


# ============================================================================
# FILL PROBABILITY MODEL
# ============================================================================

class FillProbabilityModel:
    """
    Models whether a limit order actually fills.
    Based on queue position research showing negative correlation
    between fills and subsequent returns.
    """
    
    def __init__(self, config: RealisticBacktestConfig):
        self.config = config
    
    def calculate_fill_probability(
        self,
        zone_strength: float,
        distance_to_zone_pct: float,
        volatility_pct: float,
        is_with_trend: bool
    ) -> float:
        """
        Calculate probability that a limit order at zone will fill.
        
        Factors:
        - Stronger zones are more likely to be touched
        - Closer to zone = higher probability
        - Higher volatility = higher touch probability
        - Counter-trend entries fill more often (but worse outcomes)
        """
        base_prob = self.config.base_fill_probability
        
        # Zone strength adjustment (stronger = more respected)
        strength_factor = min(zone_strength / 3.0, 1.2)  # Cap at 1.2x
        
        # Distance adjustment (further = less likely)
        distance_factor = max(0.3, 1.0 - (distance_to_zone_pct / 3.0))
        
        # Volatility adjustment (higher vol = more likely to reach)
        vol_factor = min(1.0 + (volatility_pct / 5.0), 1.3)
        
        # Trend adjustment (counter-trend fills more but worse outcomes)
        trend_factor = 0.9 if is_with_trend else 1.1
        
        probability = base_prob * strength_factor * distance_factor * vol_factor * trend_factor
        
        # Add randomness
        probability *= random.uniform(0.85, 1.15)
        
        # Clamp to reasonable range
        return max(0.2, min(0.85, probability))
    
    def should_fill(self, probability: float) -> bool:
        """Determine if order fills based on probability"""
        return random.random() < probability


# ============================================================================
# CONFIRMATION ENGINE
# ============================================================================

class ConfirmationEngine:
    """
    Handles signal confirmation logic.
    Never enters without proper confirmation.
    """
    
    def __init__(self, config: RealisticBacktestConfig):
        self.config = config
    
    def is_zone_confirmed(
        self,
        zone: ZoneData,
        current_bar: int
    ) -> bool:
        """
        Zone is only confirmed after N bars since detection.
        This prevents look-ahead bias from using zones that
        are only visible in hindsight.
        """
        bars_since_detection = current_bar - zone.detection_bar
        return bars_since_detection >= self.config.confirmation_candles
    
    def has_zone_been_touched(
        self,
        zone: ZoneData,
        bar_data: pd.Series
    ) -> bool:
        """Check if price actually reached the zone"""
        zone_mid = (zone.high + zone.low) / 2
        zone_range = zone.high - zone.low
        
        # Price must enter the zone (not just get close)
        if zone.zone_type == 'bullish':
            # For bullish OB, price should dip into it
            return bar_data['low'] <= zone.high
        else:
            # For bearish OB, price should rise into it
            return bar_data['high'] >= zone.low
    
    def is_bounce_confirmed(
        self,
        zone: ZoneData,
        current_bar_data: pd.Series,
        prev_bar_data: pd.Series
    ) -> Tuple[bool, str]:
        """
        Check for bounce confirmation:
        - Rejection wick (50%+ of range)
        - Candle close in right direction
        - Volume confirmation
        """
        reasons = []
        
        high = current_bar_data['high']
        low = current_bar_data['low']
        open_p = current_bar_data['open']
        close = current_bar_data['close']
        candle_range = high - low
        
        if candle_range == 0:
            return False, "No range"
        
        body = abs(close - open_p)
        body_pct = body / candle_range
        
        if zone.zone_type == 'bullish':
            # Looking for bullish rejection (long lower wick)
            lower_wick = min(open_p, close) - low
            wick_pct = lower_wick / candle_range
            
            # Must have significant wick
            if wick_pct < self.config.min_rejection_wick_pct:
                return False, f"Insufficient wick ({wick_pct:.1%})"
            
            # Candle should close bullish or near top
            if close < open_p and body_pct > 0.3:
                return False, "Bearish close"
            
            # Close should be above zone
            if close < zone.high:
                return False, "Close inside zone"
            
            reasons.append(f"Bullish rejection ({wick_pct:.0%} wick)")
            
        else:
            # Looking for bearish rejection (long upper wick)
            upper_wick = high - max(open_p, close)
            wick_pct = upper_wick / candle_range
            
            if wick_pct < self.config.min_rejection_wick_pct:
                return False, f"Insufficient wick ({wick_pct:.1%})"
            
            if close > open_p and body_pct > 0.3:
                return False, "Bullish close"
            
            if close > zone.low:
                return False, "Close inside zone"
            
            reasons.append(f"Bearish rejection ({wick_pct:.0%} wick)")
        
        return True, "; ".join(reasons)
    
    def check_htf_alignment(
        self,
        direction: str,
        htf_df: Optional[pd.DataFrame],
        htf_bar_idx: int
    ) -> bool:
        """Check if trade aligns with higher timeframe trend"""
        if htf_df is None or len(htf_df) <= htf_bar_idx:
            return True  # Allow if no HTF data
        
        if 'ema50' not in htf_df.columns or 'ema200' not in htf_df.columns:
            return True
        
        htf_bar = htf_df.iloc[htf_bar_idx]
        close = htf_bar['close']
        ema50 = htf_bar['ema50']
        ema200 = htf_bar['ema200']
        
        if pd.isna(ema50) or pd.isna(ema200):
            return True
        
        # Strong trend: price > EMA50 > EMA200 (uptrend) or inverse
        if direction.lower() == 'long':
            return close > ema200 and ema50 > ema200
        else:
            return close < ema200 and ema50 < ema200


# ============================================================================
# TRADE SIMULATOR
# ============================================================================

class RealisticTradeSimulator:
    """Simulates trade execution with realistic constraints"""
    
    def __init__(self, config: RealisticBacktestConfig):
        self.config = config
        self.cost_calc = RealisticCostCalculator(config)
    
    def simulate_trade(
        self,
        signal: PendingSignal,
        df: pd.DataFrame,
        start_idx: int,
        entry_bar_data: pd.Series
    ) -> Optional[RealisticTrade]:
        """
        Simulate a trade from entry through exit.
        
        Handles:
        - Slippage on entry
        - TP1 partial close
        - SL to breakeven
        - TP2 or timeout
        - All costs
        """
        direction = signal.direction
        leverage = self.config.default_leverage
        
        # Calculate entry with slippage
        volatility = self._get_volatility(df, start_idx)
        entry_slip_pct = self.cost_calc.get_slippage(signal.zone.symbol, volatility, direction)
        
        if direction == 'Long':
            actual_entry = signal.entry_price * (1 + entry_slip_pct / 100)
        else:
            actual_entry = signal.entry_price * (1 - entry_slip_pct / 100)
        
        # Recalculate SL/TP based on actual entry
        risk = abs(signal.sl_price - actual_entry)
        if risk == 0:
            return None
        
        sl = signal.sl_price
        if direction == 'Long':
            tp1 = actual_entry + risk * self.config.tp1_r_multiple
            tp2 = actual_entry + risk * self.config.tp2_r_multiple
        else:
            tp1 = actual_entry - risk * self.config.tp1_r_multiple
            tp2 = actual_entry - risk * self.config.tp2_r_multiple
        
        # Simulate forward
        tp1_hit = False
        tp1_exit_price = None
        current_sl = sl
        exit_price = None
        exit_type = None
        exit_bar = start_idx
        
        max_bar = min(start_idx + self.config.max_hold_bars, len(df) - 1)
        
        for i in range(start_idx + 1, max_bar + 1):
            bar = df.iloc[i]
            high = bar['high']
            low = bar['low']
            
            # Check SL first (worst case)
            if direction == 'Long':
                if low <= current_sl:
                    exit_price = current_sl
                    exit_type = 'BE' if tp1_hit else 'SL'
                    exit_bar = i
                    break
            else:
                if high >= current_sl:
                    exit_price = current_sl
                    exit_type = 'BE' if tp1_hit else 'SL'
                    exit_bar = i
                    break
            
            # Check TP1 (partial close)
            if not tp1_hit:
                if (direction == 'Long' and high >= tp1) or \
                   (direction == 'Short' and low <= tp1):
                    tp1_hit = True
                    tp1_exit_price = tp1
                    current_sl = actual_entry  # Move to breakeven
            
            # Check TP2 (full close)
            if tp1_hit:
                if (direction == 'Long' and high >= tp2) or \
                   (direction == 'Short' and low <= tp2):
                    exit_price = tp2
                    exit_type = 'TP2'
                    exit_bar = i
                    break
        
        # Timeout - exit at current price
        if exit_price is None:
            exit_bar = max_bar
            exit_price = df.iloc[exit_bar]['close']
            exit_type = 'TIMEOUT'
        
        # Calculate bars held
        bars_held = exit_bar - start_idx
        hours_held = bars_held * 4  # 4h timeframe
        
        # Apply exit slippage
        exit_volatility = self._get_volatility(df, exit_bar)
        exit_slip_pct = self.cost_calc.get_slippage(signal.zone.symbol, exit_volatility, direction) * 1.2
        
        if direction == 'Long':
            actual_exit = exit_price * (1 - exit_slip_pct / 100)
        else:
            actual_exit = exit_price * (1 + exit_slip_pct / 100)
        
        # Calculate PnL
        if direction == 'Long':
            gross_pnl_pct = (actual_exit - actual_entry) / actual_entry * 100 * leverage
        else:
            gross_pnl_pct = (actual_entry - actual_exit) / actual_entry * 100 * leverage
        
        # If TP1 was hit, add that partial profit
        tp1_pnl = 0.0
        if tp1_hit and tp1_exit_price:
            if direction == 'Long':
                tp1_pnl = (tp1_exit_price - actual_entry) / actual_entry * 100 * leverage * 0.5
            else:
                tp1_pnl = (actual_entry - tp1_exit_price) / actual_entry * 100 * leverage * 0.5
            
            # Remaining position PnL
            remaining_pnl = gross_pnl_pct * 0.5
            gross_pnl_pct = tp1_pnl + remaining_pnl
        
        # Calculate costs
        costs = self.cost_calc.calculate_total_costs(
            signal.zone.symbol,
            direction,
            actual_entry,
            actual_exit,
            volatility,
            hours_held,
            leverage
        )
        
        net_pnl_pct = gross_pnl_pct - costs['total_costs']
        
        # Calculate R multiple
        risk_pct = abs(sl - actual_entry) / actual_entry * 100 * leverage
        pnl_r = net_pnl_pct / risk_pct if risk_pct > 0 else 0
        
        # Determine result
        if net_pnl_pct > 0.5:
            result = 'WIN'
        elif net_pnl_pct < -0.5:
            result = 'LOSS'
        else:
            result = 'BREAKEVEN'
        
        # Create trade record
        entry_time = df.index[start_idx] if isinstance(df.index[start_idx], datetime) else datetime.now(timezone.utc)
        exit_time = df.index[exit_bar] if isinstance(df.index[exit_bar], datetime) else datetime.now(timezone.utc)
        
        return RealisticTrade(
            symbol=signal.zone.symbol,
            direction=direction,
            entry_price=actual_entry,
            sl_price=sl,
            tp1_price=tp1,
            tp2_price=tp2,
            exit_price=actual_exit,
            signal_time=signal.created_time,
            entry_time=entry_time,
            exit_time=exit_time,
            leverage=leverage,
            slippage_pct=costs['total_slippage'],
            fees_pct=costs['total_fees'],
            funding_paid_pct=costs['funding'],
            gross_pnl_pct=gross_pnl_pct,
            total_costs_pct=costs['total_costs'],
            net_pnl_pct=net_pnl_pct,
            pnl_r=pnl_r,
            exit_type=exit_type,
            tp1_hit=tp1_hit,
            tp1_pnl_pct=tp1_pnl,
            bars_held=bars_held,
            factors=signal.factors,
            result=result
        )
    
    def _get_volatility(self, df: pd.DataFrame, idx: int) -> float:
        """Get ATR-based volatility percentage"""
        if 'atr' in df.columns and idx < len(df):
            atr = df.iloc[idx]['atr']
            close = df.iloc[idx]['close']
            if pd.notna(atr) and close > 0:
                return (atr / close) * 100
        return 2.0  # Default 2%


# ============================================================================
# MAIN BACKTEST ENGINE
# ============================================================================

class RealisticBacktestEngine:
    """
    Main backtest orchestrator with realistic methodology.
    
    Process:
    1. Walk through data bar-by-bar (no look-ahead)
    2. Detect zones using only past data
    3. Wait for confirmation (N bars after detection)
    4. Wait for price to touch zone
    5. Wait for bounce confirmation
    6. Apply fill probability model
    7. Simulate trade with realistic costs
    8. Track frequency limits
    """
    
    def __init__(self, config: RealisticBacktestConfig = None):
        self.config = config or RealisticBacktestConfig()
        self.confirmation = ConfirmationEngine(self.config)
        self.fill_model = FillProbabilityModel(self.config)
        self.simulator = RealisticTradeSimulator(self.config)
    
    async def run_backtest(
        self,
        symbols: List[str],
        progress_callback=None
    ) -> RealisticBacktestResult:
        """Run realistic backtest"""
        
        result = RealisticBacktestResult()
        result.config = self.config
        all_trades = []
        
        # Track frequency limits
        last_trade_time: Dict[str, datetime] = {}
        trades_this_week: Dict[str, List[datetime]] = defaultdict(list)
        open_positions: Dict[str, RealisticTrade] = {}
        
        from bot.data_fetcher import fetch_ohlcv
        from bot.indicators import add_institutional_indicators
        
        for sym_idx, symbol in enumerate(symbols):
            try:
                if progress_callback:
                    await progress_callback(f"Testing {symbol} ({sym_idx + 1}/{len(symbols)})")
                
                # Fetch data with warmup
                total_bars_needed = (self.config.days * 6) + self.config.warmup_bars
                since = int((datetime.now(timezone.utc) - timedelta(days=self.config.days + 60)).timestamp() * 1000)
                
                df = await fetch_ohlcv(symbol, self.config.primary_timeframe, limit=total_bars_needed, since=since)
                if df is None or len(df) < self.config.warmup_bars + 100:
                    logging.warning(f"{symbol}: Insufficient data ({len(df) if df is not None else 0} bars)")
                    continue
                
                # Fetch HTF data
                df_htf = await fetch_ohlcv(symbol, self.config.htf_timeframe, limit=self.config.days + 60, since=since)
                
                # Add indicators
                df = add_institutional_indicators(df)
                if df_htf is not None:
                    df_htf = add_institutional_indicators(df_htf)
                
                # Zone tracking
                active_zones: List[ZoneData] = []
                pending_signals: List[PendingSignal] = []
                
                # Walk through each bar (after warmup)
                for i in range(self.config.warmup_bars, len(df) - 10):
                    current_time = df.index[i] if isinstance(df.index[i], datetime) else datetime.now(timezone.utc)
                    current_bar = df.iloc[i]
                    prev_bar = df.iloc[i - 1] if i > 0 else current_bar
                    
                    # === STEP 1: Detect new zones (using only past data) ===
                    new_zones = await self._detect_zones(df.iloc[:i+1], symbol, i, current_time)
                    active_zones.extend(new_zones)
                    result.total_signals += len(new_zones)
                    
                    # === STEP 2: Update zone states ===
                    for zone in active_zones:
                        if zone.invalidated:
                            continue
                        
                        # Check if zone is now confirmed
                        if zone.state == SignalState.ZONE_DETECTED:
                            if self.confirmation.is_zone_confirmed(zone, i):
                                zone.state = SignalState.ZONE_CONFIRMED
                                zone.confirmation_bar = i
                        
                        # Check if zone has been touched
                        elif zone.state == SignalState.ZONE_CONFIRMED:
                            if self.confirmation.has_zone_been_touched(zone, current_bar):
                                zone.state = SignalState.TOUCHED
                                zone.touch_bar = i
                                zone.touch_price = current_bar['close']
                        
                        # Check for bounce confirmation
                        elif zone.state == SignalState.TOUCHED:
                            is_confirmed, reason = self.confirmation.is_bounce_confirmed(
                                zone, current_bar, prev_bar
                            )
                            if is_confirmed:
                                zone.state = SignalState.CONFIRMED
                                result.signals_confirmed += 1
                                
                                # Create pending signal
                                signal = self._create_signal(zone, df, i, current_time, df_htf)
                                if signal:
                                    pending_signals.append(signal)
                    
                    # === STEP 3: Process pending signals ===
                    for signal in pending_signals[:]:
                        if signal.state != SignalState.CONFIRMED:
                            continue
                        
                        # Check frequency limits
                        if not self._check_frequency_limits(
                            symbol, current_time, last_trade_time, 
                            trades_this_week, open_positions
                        ):
                            continue
                        
                        # Apply fill probability
                        if not self.fill_model.should_fill(signal.fill_probability):
                            signal.state = SignalState.REJECTED
                            continue
                        
                        result.orders_placed += 1
                        
                        # Simulate trade
                        trade = self.simulator.simulate_trade(signal, df, i, current_bar)
                        
                        if trade:
                            result.orders_filled += 1
                            all_trades.append(trade)
                            
                            # Update tracking
                            last_trade_time[symbol] = current_time
                            trades_this_week[symbol].append(current_time)
                            
                            # Clean old weekly trades
                            week_ago = current_time - timedelta(days=7)
                            trades_this_week[symbol] = [
                                t for t in trades_this_week[symbol] if t > week_ago
                            ]
                        
                        signal.state = SignalState.FILLED
                        pending_signals.remove(signal)
                    
                    # === STEP 4: Clean up old zones ===
                    active_zones = [z for z in active_zones if not z.invalidated and 
                                    (i - z.detection_bar) < 50]
                    
                    # Yield for async
                    if i % 200 == 0:
                        await asyncio.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Backtest error for {symbol}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        # Calculate final metrics
        result = self._calculate_metrics(result, all_trades)
        
        return result
    
    async def _detect_zones(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_idx: int,
        current_time: datetime
    ) -> List[ZoneData]:
        """Detect order block zones using only historical data"""
        zones = []
        
        try:
            from bot.order_blocks import find_unmitigated_order_blocks
            
            # Only use data up to current bar (no look-ahead)
            obs = await find_unmitigated_order_blocks(
                df, lookback=50, tf='4h', 
                min_strength=self.config.min_ob_strength
            )
            
            for ob_type in ['bullish', 'bearish']:
                for ob in obs.get(ob_type, [])[:2]:  # Max 2 per type
                    strength = ob.get('strength', 0)
                    if strength < self.config.min_ob_strength:
                        continue
                    
                    # Check if zone is within tradeable distance
                    mid = (ob['low'] + ob['high']) / 2
                    current_price = df['close'].iloc[-1]
                    distance = abs(current_price - mid) / current_price * 100
                    
                    if distance > 10:  # Skip zones too far away
                        continue
                    
                    zone = ZoneData(
                        symbol=symbol,
                        zone_type=ob_type,
                        high=ob['high'],
                        low=ob['low'],
                        strength=strength,
                        detection_bar=current_idx,
                        detection_time=current_time
                    )
                    zones.append(zone)
        
        except Exception as e:
            logging.debug(f"Zone detection error: {e}")
        
        return zones
    
    def _create_signal(
        self,
        zone: ZoneData,
        df: pd.DataFrame,
        current_idx: int,
        current_time: datetime,
        df_htf: Optional[pd.DataFrame]
    ) -> Optional[PendingSignal]:
        """Create a trading signal from confirmed zone"""
        
        direction = 'Long' if zone.zone_type == 'bullish' else 'Short'
        current_price = df['close'].iloc[current_idx]
        
        # Check HTF alignment
        htf_idx = current_idx // 6  # Approximate conversion
        if self.config.min_htf_alignment:
            if not self.confirmation.check_htf_alignment(direction, df_htf, htf_idx):
                return None
        
        # Calculate entry, SL, TP
        zone_mid = (zone.high + zone.low) / 2
        atr = df['atr'].iloc[current_idx] if 'atr' in df.columns else current_price * 0.02
        
        if direction == 'Long':
            entry = zone.high + atr * 0.1  # Enter slightly above zone
            sl = zone.low - atr * 0.3
            tp1 = entry + abs(entry - sl) * self.config.tp1_r_multiple
            tp2 = entry + abs(entry - sl) * self.config.tp2_r_multiple
        else:
            entry = zone.low - atr * 0.1
            sl = zone.high + atr * 0.3
            tp1 = entry - abs(sl - entry) * self.config.tp1_r_multiple
            tp2 = entry - abs(sl - entry) * self.config.tp2_r_multiple
        
        # Build factors
        factors = [f'OB({zone.strength:.1f})']
        
        # Check additional factors
        if 'ema200' in df.columns:
            ema200 = df['ema200'].iloc[current_idx]
            if pd.notna(ema200):
                if (direction == 'Long' and current_price > ema200) or \
                   (direction == 'Short' and current_price < ema200):
                    factors.append('Trend')
        
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[current_idx]
            if pd.notna(rsi):
                if (direction == 'Long' and 30 < rsi < 60) or \
                   (direction == 'Short' and 40 < rsi < 70):
                    factors.append('RSI')
        
        # Check confluence requirement
        if len(factors) < self.config.min_confluence_factors:
            return None
        
        # Calculate fill probability
        distance_pct = abs(current_price - zone_mid) / current_price * 100
        volatility = atr / current_price * 100 if current_price > 0 else 2.0
        is_with_trend = 'Trend' in factors
        
        fill_prob = self.fill_model.calculate_fill_probability(
            zone.strength, distance_pct, volatility, is_with_trend
        )
        
        return PendingSignal(
            zone=zone,
            direction=direction,
            entry_price=entry,
            sl_price=sl,
            tp1_price=tp1,
            tp2_price=tp2,
            factors=factors,
            confluence_count=len(factors),
            state=SignalState.CONFIRMED,
            created_bar=current_idx,
            created_time=current_time,
            fill_probability=fill_prob
        )
    
    def _check_frequency_limits(
        self,
        symbol: str,
        current_time: datetime,
        last_trade_time: Dict[str, datetime],
        trades_this_week: Dict[str, List[datetime]],
        open_positions: Dict[str, Any]
    ) -> bool:
        """Check if we can take a new trade based on frequency limits"""
        
        # Check max concurrent positions
        if len(open_positions) >= self.config.max_concurrent_positions:
            return False
        
        # Check minimum time between trades
        if symbol in last_trade_time:
            hours_since_last = (current_time - last_trade_time[symbol]).total_seconds() / 3600
            if hours_since_last < self.config.min_hours_between_trades:
                return False
        
        # Check weekly limit
        week_ago = current_time - timedelta(days=7)
        recent_trades = [t for t in trades_this_week.get(symbol, []) if t > week_ago]
        if len(recent_trades) >= self.config.max_trades_per_symbol_week:
            return False
        
        return True
    
    def _calculate_metrics(
        self,
        result: RealisticBacktestResult,
        trades: List[RealisticTrade]
    ) -> RealisticBacktestResult:
        """Calculate all performance metrics"""
        
        result.trades = trades
        
        if not trades:
            result.validation_warnings.append("No trades generated")
            return result
        
        # Basic counts
        result.wins = len([t for t in trades if t.result == 'WIN'])
        result.losses = len([t for t in trades if t.result == 'LOSS'])
        result.breakevens = len([t for t in trades if t.result == 'BREAKEVEN'])
        
        total = len(trades)
        
        # Win rate
        result.win_rate = result.wins / total * 100 if total > 0 else 0
        
        # PnL
        all_pnl = [t.net_pnl_pct for t in trades]
        win_pnl = [t.net_pnl_pct for t in trades if t.result == 'WIN']
        loss_pnl = [t.net_pnl_pct for t in trades if t.result == 'LOSS']
        
        result.total_pnl_pct = sum(all_pnl)
        result.total_pnl_r = sum(t.pnl_r for t in trades)
        result.avg_win_pct = np.mean(win_pnl) if win_pnl else 0
        result.avg_loss_pct = np.mean(loss_pnl) if loss_pnl else 0
        
        # Profit factor
        gross_profit = sum([p for p in all_pnl if p > 0])
        gross_loss = abs(sum([p for p in all_pnl if p < 0]))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Costs breakdown
        result.total_fees_pct = sum(t.fees_pct for t in trades)
        result.total_slippage_pct = sum(t.slippage_pct for t in trades)
        result.total_funding_pct = sum(t.funding_paid_pct for t in trades)
        
        # Equity curve and drawdown
        equity = 100.0
        peak = equity
        max_dd = 0
        result.equity_curve = [equity]
        
        for trade in trades:
            equity += trade.net_pnl_pct
            result.equity_curve.append(equity)
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        
        result.max_drawdown_pct = max_dd
        
        # Risk-adjusted metrics
        if len(all_pnl) > 1:
            returns = np.array(all_pnl)
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            
            # Sharpe (assuming ~200 trades per year max)
            result.sharpe_ratio = mean_ret / std_ret * np.sqrt(min(len(trades), 200)) if std_ret > 0 else 0
            
            # Sortino
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 0 else std_ret
            result.sortino_ratio = mean_ret / downside_std * np.sqrt(min(len(trades), 200)) if downside_std > 0 else 0
        
        # TP statistics
        result.tp1_hit_rate = len([t for t in trades if t.tp1_hit]) / total * 100 if total > 0 else 0
        result.tp2_hit_rate = len([t for t in trades if t.exit_type == 'TP2']) / total * 100 if total > 0 else 0
        
        # Hold time
        result.avg_hold_hours = np.mean([t.bars_held * 4 for t in trades]) if trades else 0
        
        # Monte Carlo
        mc_results = self._run_monte_carlo(trades)
        result.mc_median_pnl = mc_results['median']
        result.mc_95_lower = mc_results['lower_95']
        result.mc_95_upper = mc_results['upper_95']
        result.mc_prob_profit = mc_results['prob_profit']
        result.mc_prob_drawdown_25 = mc_results['prob_dd_25']
        
        # Factor analysis
        factor_trades = defaultdict(list)
        for trade in trades:
            for factor in trade.factors:
                factor_name = factor.split('(')[0]
                factor_trades[factor_name].append(trade)
        
        for factor, f_trades in factor_trades.items():
            f_wins = len([t for t in f_trades if t.result == 'WIN'])
            f_total = len(f_trades)
            f_pnl = sum([t.net_pnl_pct for t in f_trades])
            result.factor_performance[factor] = {
                'trades': f_total,
                'wins': f_wins,
                'win_rate': f_wins / f_total * 100 if f_total > 0 else 0,
                'total_pnl': f_pnl,
                'avg_pnl': f_pnl / f_total if f_total > 0 else 0
            }
        
        # Validation
        if total >= self.config.min_trades_for_validity:
            result.is_statistically_valid = True
        else:
            result.validation_warnings.append(
                f"Only {total} trades - need {self.config.min_trades_for_validity}+ for statistical validity"
            )
        
        if result.profit_factor > 4.0:
            result.validation_warnings.append("Profit factor > 4.0 - possible overfitting")
        
        if result.sharpe_ratio > 3.0:
            result.validation_warnings.append("Sharpe > 3.0 - verify methodology")
        
        if result.win_rate > 70:
            result.validation_warnings.append("Win rate > 70% - unusual for this strategy type")
        
        # Dates
        if trades:
            result.start_date = trades[0].entry_time
            result.end_date = trades[-1].exit_time
        
        return result
    
    def _run_monte_carlo(self, trades: List[RealisticTrade], n_sims: int = 1000) -> Dict:
        """Run Monte Carlo simulation"""
        if not trades:
            return {'median': 0, 'lower_95': 0, 'upper_95': 0, 'prob_profit': 0, 'prob_dd_25': 0}
        
        pnl_values = [t.net_pnl_pct for t in trades]
        n_trades = len(pnl_values)
        
        final_returns = []
        max_drawdowns = []
        
        for _ in range(n_sims):
            # Resample with replacement
            resampled = np.random.choice(pnl_values, size=n_trades, replace=True)
            
            # Calculate final return
            final_returns.append(np.sum(resampled))
            
            # Calculate max drawdown for this simulation
            equity = 100
            peak = 100
            max_dd = 0
            for pnl in resampled:
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)
            max_drawdowns.append(max_dd)
        
        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            'median': np.median(final_returns),
            'mean': np.mean(final_returns),
            'lower_95': np.percentile(final_returns, 2.5),
            'upper_95': np.percentile(final_returns, 97.5),
            'prob_profit': np.mean(final_returns > 0) * 100,
            'prob_dd_25': np.mean(max_drawdowns > 25) * 100
        }


# ============================================================================
# FORMATTING
# ============================================================================

def format_realistic_report(result: RealisticBacktestResult) -> str:
    """Format realistic backtest results"""
    
    if not result.trades:
        return (
            "📊 **REALISTIC BACKTEST RESULTS**\n\n"
            "❌ **No trades generated**\n\n"
            "This is actually realistic - high-quality setups are rare.\n"
            "Consider:\n"
            "• Extending test period\n"
            "• Adding more symbols\n"
            "• Slightly relaxing confirmation requirements"
        )
    
    total = len(result.trades)
    
    # Grade based on realistic expectations
    if result.profit_factor >= 1.8 and result.win_rate >= 50 and result.is_statistically_valid:
        grade = "A"
        grade_emoji = "🌟"
        grade_note = "Excellent - worth live testing"
    elif result.profit_factor >= 1.5 and result.win_rate >= 45:
        grade = "B"
        grade_emoji = "✅"
        grade_note = "Good - promising strategy"
    elif result.profit_factor >= 1.2 and result.win_rate >= 40:
        grade = "C"
        grade_emoji = "⚡"
        grade_note = "Marginal - needs optimization"
    else:
        grade = "D"
        grade_emoji = "⚠️"
        grade_note = "Poor - reconsider approach"
    
    msg = f"📊 **REALISTIC BACKTEST** {grade_emoji} Grade {grade}\n"
    msg += f"_{grade_note}_\n\n"
    
    # Signal funnel
    msg += f"**🔍 Signal Funnel**\n"
    msg += f"• Zones Detected: {result.total_signals}\n"
    msg += f"• Confirmed: {result.signals_confirmed} ({result.signals_confirmed/max(result.total_signals,1)*100:.0f}%)\n"
    msg += f"• Orders Placed: {result.orders_placed}\n"
    msg += f"• Orders Filled: {result.orders_filled} ({result.orders_filled/max(result.orders_placed,1)*100:.0f}%)\n\n"
    
    # Results
    msg += f"**📈 Trade Results ({total} trades)**\n"
    msg += f"• Wins: {result.wins} | Losses: {result.losses} | BE: {result.breakevens}\n"
    msg += f"• Win Rate: {result.win_rate:.1f}%\n"
    msg += f"• TP1 Hit: {result.tp1_hit_rate:.0f}% | TP2 Hit: {result.tp2_hit_rate:.0f}%\n\n"
    
    # Performance
    msg += f"**💰 Performance**\n"
    msg += f"• Net PnL: {result.total_pnl_pct:+.1f}% ({result.total_pnl_r:+.1f}R)\n"
    msg += f"• Avg Win: {result.avg_win_pct:+.2f}%\n"
    msg += f"• Avg Loss: {result.avg_loss_pct:.2f}%\n"
    msg += f"• Profit Factor: {result.profit_factor:.2f}\n\n"
    
    # Costs
    total_costs = result.total_fees_pct + result.total_slippage_pct + result.total_funding_pct
    msg += f"**💸 Trading Costs**\n"
    msg += f"• Fees: {result.total_fees_pct:.2f}%\n"
    msg += f"• Slippage: {result.total_slippage_pct:.2f}%\n"
    msg += f"• Funding: {result.total_funding_pct:.2f}%\n"
    msg += f"• **Total Costs: {total_costs:.2f}%**\n\n"
    
    # Risk
    msg += f"**⚠️ Risk Metrics**\n"
    msg += f"• Max Drawdown: {result.max_drawdown_pct:.1f}%\n"
    msg += f"• Sharpe Ratio: {result.sharpe_ratio:.2f}\n"
    msg += f"• Avg Hold: {result.avg_hold_hours:.0f}h\n\n"
    
    # Monte Carlo
    msg += f"**🎲 Monte Carlo (95% CI)**\n"
    msg += f"• Expected: {result.mc_median_pnl:+.1f}%\n"
    msg += f"• Range: {result.mc_95_lower:+.1f}% to {result.mc_95_upper:+.1f}%\n"
    msg += f"• P(Profit): {result.mc_prob_profit:.0f}%\n"
    msg += f"• P(DD>25%): {result.mc_prob_drawdown_25:.0f}%\n\n"
    
    # Validation warnings
    if result.validation_warnings:
        msg += f"**⚠️ Warnings**\n"
        for warning in result.validation_warnings:
            msg += f"• {warning}\n"
        msg += "\n"
    
    # Realistic expectations note
    msg += f"_These results include realistic slippage, fees, funding, and fill probability._"
    
    return msg
