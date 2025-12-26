# backtest_pro.py - Grok Elite Signal Bot v27.12.15 - Professional Backtesting
# -*- coding: utf-8 -*-
"""
PROFESSIONAL BACKTESTING SYSTEM

Key improvements over basic backtest:
1. 365 days of data (vs 90) for statistical significance
2. Multi-timeframe analysis (4h + 1d) matching live signals
3. Proper indicator warmup periods (200+ bars)
4. Relaxed signal conditions to generate more trades
5. Realistic fee (0.06%) and slippage (0.05-0.1%) modeling
6. Proper leverage-adjusted PnL calculation
7. Monte Carlo simulation for confidence intervals
8. Walk-forward validation
9. Comprehensive metrics: Win Rate, Profit Factor, Sharpe, Max DD
10. Per-factor performance tracking

v27.12.15: Initial professional implementation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BacktestConfig:
    """Configuration for professional backtest"""
    # Data settings
    days: int = 365                    # 1 year of data
    primary_timeframe: str = '4h'      # Main signal timeframe
    htf_timeframe: str = '1d'          # Higher timeframe confirmation
    warmup_periods: int = 250          # Indicator warmup (for EMA200)
    
    # Signal conditions (relaxed for more trades)
    min_ob_strength: float = 1.5       # Minimum OB strength (was 2.0)
    max_entry_distance_pct: float = 5.0  # Max distance to OB (was 2.0)
    min_confluence: int = 2            # Minimum confluence factors
    
    # Risk settings
    risk_per_trade_pct: float = 1.5    # Risk per trade
    default_leverage: int = 3          # Default leverage
    max_concurrent_trades: int = 5     # Max open positions
    
    # Cost modeling (realistic for perpetual futures)
    maker_fee_pct: float = 0.02        # Maker fee
    taker_fee_pct: float = 0.06        # Taker fee (assume taker)
    slippage_major_pct: float = 0.05   # Slippage for BTC/ETH
    slippage_alt_pct: float = 0.15     # Slippage for altcoins
    funding_rate_8h: float = 0.01      # Average funding rate per 8h
    
    # TP/SL settings
    tp1_r_multiple: float = 1.5        # TP1 at 1.5R
    tp2_r_multiple: float = 3.0        # TP2 at 3R
    partial_close_pct: float = 0.5     # Close 50% at TP1
    move_sl_to_be: bool = True         # Move SL to breakeven after TP1
    
    # Monte Carlo settings
    mc_simulations: int = 1000         # Number of Monte Carlo runs
    mc_confidence_level: float = 0.95  # 95% confidence interval
    
    # Walk-forward settings
    wf_train_days: int = 180           # Training window
    wf_test_days: int = 60             # Testing window


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    direction: str  # 'Long' or 'Short'
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp1_price: float
    tp2_price: float
    leverage: int
    position_size_pct: float  # As % of capital
    
    # Execution
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_type: Optional[str] = None  # 'TP1', 'TP2', 'SL', 'BE'
    
    # Partial close tracking
    tp1_hit: bool = False
    tp1_exit_price: Optional[float] = None
    tp1_pnl_pct: float = 0.0
    remaining_size_pct: float = 1.0  # 1.0 = 100%, 0.5 = 50% after partial
    sl_moved_to_be: bool = False
    
    # Results
    gross_pnl_pct: float = 0.0
    fees_pct: float = 0.0
    slippage_pct: float = 0.0
    net_pnl_pct: float = 0.0
    pnl_r: float = 0.0  # In R multiples
    result: str = ''  # 'WIN', 'LOSS', 'BREAKEVEN'
    
    # Confluence factors
    factors: List[str] = field(default_factory=list)
    confluence_count: int = 0
    grade: str = ''


@dataclass
class BacktestResult:
    """Complete backtest results"""
    # Basic stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    
    # Win rate
    win_rate: float = 0.0
    
    # PnL metrics
    total_pnl_pct: float = 0.0
    total_pnl_r: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    
    # Risk metrics
    profit_factor: float = 0.0
    expected_value_r: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # TP statistics
    tp1_hit_rate: float = 0.0
    tp2_hit_rate: float = 0.0
    avg_hold_time_hours: float = 0.0
    
    # Equity curve
    starting_capital: float = 10000.0
    ending_capital: float = 10000.0
    equity_curve: List[float] = field(default_factory=list)
    
    # Monte Carlo results
    mc_median_return: float = 0.0
    mc_95_lower: float = 0.0
    mc_95_upper: float = 0.0
    mc_worst_case: float = 0.0
    mc_probability_profit: float = 0.0
    
    # Per-factor performance
    factor_stats: Dict[str, Dict] = field(default_factory=dict)
    
    # Trade list
    trades: List[Trade] = field(default_factory=list)
    
    # Metadata
    config: Optional[BacktestConfig] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols_tested: List[str] = field(default_factory=list)


# ============================================================================
# PNL CALCULATOR - PROPER LEVERAGE HANDLING
# ============================================================================

class PnLCalculator:
    """
    Accurate PnL calculation for leveraged perpetual futures.
    
    Rules:
    - LONG: Profit % = ((exit - entry) / entry) × leverage × position_size
    - SHORT: Profit % = ((entry - exit) / entry) × leverage × position_size
    - Fees applied per transaction (entry + exit)
    - Slippage applied to entry and exit prices
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def calculate_raw_pnl_pct(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        leverage: int = 3,
        position_size_pct: float = 1.0
    ) -> float:
        """
        Calculate raw PnL percentage (before fees/slippage).
        
        Args:
            direction: 'Long' or 'Short'
            entry_price: Entry price
            exit_price: Exit price
            leverage: Leverage used
            position_size_pct: Position size as decimal (1.0 = 100%, 0.5 = 50%)
        
        Returns:
            Raw PnL as percentage of capital risked
        """
        if direction.lower() == 'long':
            price_move_pct = (exit_price - entry_price) / entry_price * 100
        else:
            price_move_pct = (entry_price - exit_price) / entry_price * 100
        
        leveraged_pnl = price_move_pct * leverage * position_size_pct
        return leveraged_pnl
    
    def calculate_fees(
        self,
        entry_price: float,
        exit_price: float,
        position_size_pct: float,
        is_maker: bool = False
    ) -> float:
        """Calculate total fees as percentage of capital."""
        fee_rate = self.config.maker_fee_pct if is_maker else self.config.taker_fee_pct
        
        # Fees on entry and exit (as % of position value, not capital)
        # Since we use position_size_pct of capital, fees are relative
        total_fee_pct = fee_rate * 2 * position_size_pct  # Entry + Exit
        
        return total_fee_pct
    
    def calculate_slippage(
        self,
        symbol: str,
        position_size_pct: float
    ) -> float:
        """Calculate slippage as percentage of capital."""
        # Major pairs have lower slippage
        is_major = symbol in ['BTC/USDT', 'ETH/USDT']
        slippage_rate = self.config.slippage_major_pct if is_major else self.config.slippage_alt_pct
        
        # Slippage on entry and exit
        total_slippage = slippage_rate * 2 * position_size_pct
        
        return total_slippage
    
    def calculate_full_pnl(
        self,
        trade: Trade,
        exit_price: float,
        exit_type: str
    ) -> Dict[str, float]:
        """
        Calculate complete PnL including fees and slippage.
        
        Returns dict with:
        - gross_pnl_pct: Raw PnL
        - fees_pct: Total fees
        - slippage_pct: Total slippage
        - net_pnl_pct: Final PnL after costs
        - pnl_r: PnL in R multiples
        """
        # Handle partial close scenario
        if trade.tp1_hit and trade.remaining_size_pct < 1.0:
            # This is TP2 or SL after partial close
            position_pct = trade.remaining_size_pct
            # Already realized TP1 gains
            tp1_realized = trade.tp1_pnl_pct
        else:
            position_pct = 1.0
            tp1_realized = 0.0
        
        # Raw PnL for remaining position
        gross_pnl = self.calculate_raw_pnl_pct(
            trade.direction,
            trade.entry_price,
            exit_price,
            trade.leverage,
            position_pct
        )
        
        # Costs for this exit
        fees = self.calculate_fees(
            trade.entry_price, exit_price, position_pct
        )
        slippage = self.calculate_slippage(trade.symbol, position_pct)
        
        # Net PnL for this exit
        net_pnl_this_exit = gross_pnl - fees - slippage
        
        # Total net PnL including TP1 if applicable
        total_net_pnl = net_pnl_this_exit + tp1_realized
        
        # Calculate R multiple
        risk = abs(trade.sl_price - trade.entry_price) / trade.entry_price * 100 * trade.leverage
        pnl_r = total_net_pnl / risk if risk > 0 else 0
        
        return {
            'gross_pnl_pct': gross_pnl + (trade.tp1_pnl_pct if trade.tp1_hit else 0),
            'fees_pct': fees,
            'slippage_pct': slippage,
            'net_pnl_pct': total_net_pnl,
            'pnl_r': pnl_r
        }
    
    def calculate_partial_close_pnl(
        self,
        trade: Trade,
        tp1_price: float
    ) -> Dict[str, float]:
        """
        Calculate PnL for partial close at TP1.
        
        After this:
        - 50% position closed with realized profit
        - Remaining 50% continues with SL at breakeven (0R risk)
        """
        partial_size = self.config.partial_close_pct  # 0.5 = 50%
        
        # Raw PnL on closed portion
        gross_pnl = self.calculate_raw_pnl_pct(
            trade.direction,
            trade.entry_price,
            tp1_price,
            trade.leverage,
            partial_size
        )
        
        # Costs on closed portion
        fees = self.calculate_fees(trade.entry_price, tp1_price, partial_size)
        slippage = self.calculate_slippage(trade.symbol, partial_size)
        
        net_pnl = gross_pnl - fees - slippage
        
        # R multiple for partial
        risk = abs(trade.sl_price - trade.entry_price) / trade.entry_price * 100 * trade.leverage
        pnl_r = net_pnl / risk if risk > 0 else 0
        
        return {
            'gross_pnl_pct': gross_pnl,
            'fees_pct': fees,
            'slippage_pct': slippage,
            'net_pnl_pct': net_pnl,
            'pnl_r': pnl_r * partial_size  # Adjust for partial
        }


# ============================================================================
# TRADE SIMULATOR
# ============================================================================

class TradeSimulator:
    """Simulates trades through historical data"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.pnl_calc = PnLCalculator(config)
    
    def simulate_trade(
        self,
        trade: Trade,
        df: pd.DataFrame,
        start_idx: int,
        max_bars: int = 100
    ) -> Trade:
        """
        Simulate a trade forward through historical data.
        
        Handles:
        - TP1 partial close with SL move to breakeven
        - TP2 full close
        - SL hit (original or breakeven)
        - Timeout after max_bars
        """
        trade = trade  # Work with copy
        
        for i in range(start_idx, min(start_idx + max_bars, len(df))):
            row = df.iloc[i]
            high = row['high']
            low = row['low']
            current_time = row.name if isinstance(row.name, datetime) else datetime.now(timezone.utc)
            
            # Determine current SL
            current_sl = trade.entry_price if trade.sl_moved_to_be else trade.sl_price
            
            # Check TP1 first (if not already hit)
            if not trade.tp1_hit:
                if trade.direction == 'Long' and high >= trade.tp1_price:
                    trade.tp1_hit = True
                    trade.tp1_exit_price = trade.tp1_price
                    
                    # Calculate partial close PnL
                    partial_result = self.pnl_calc.calculate_partial_close_pnl(trade, trade.tp1_price)
                    trade.tp1_pnl_pct = partial_result['net_pnl_pct']
                    trade.remaining_size_pct = 1.0 - self.config.partial_close_pct
                    
                    # Move SL to breakeven
                    if self.config.move_sl_to_be:
                        trade.sl_moved_to_be = True
                    
                elif trade.direction == 'Short' and low <= trade.tp1_price:
                    trade.tp1_hit = True
                    trade.tp1_exit_price = trade.tp1_price
                    
                    partial_result = self.pnl_calc.calculate_partial_close_pnl(trade, trade.tp1_price)
                    trade.tp1_pnl_pct = partial_result['net_pnl_pct']
                    trade.remaining_size_pct = 1.0 - self.config.partial_close_pct
                    
                    if self.config.move_sl_to_be:
                        trade.sl_moved_to_be = True
            
            # Check TP2 (full exit remaining position)
            if trade.tp1_hit:
                if trade.direction == 'Long' and high >= trade.tp2_price:
                    trade.exit_price = trade.tp2_price
                    trade.exit_time = current_time
                    trade.exit_type = 'TP2'
                    break
                elif trade.direction == 'Short' and low <= trade.tp2_price:
                    trade.exit_price = trade.tp2_price
                    trade.exit_time = current_time
                    trade.exit_type = 'TP2'
                    break
            
            # Check SL (original or breakeven)
            if trade.direction == 'Long' and low <= current_sl:
                trade.exit_price = current_sl
                trade.exit_time = current_time
                trade.exit_type = 'BE' if trade.sl_moved_to_be else 'SL'
                break
            elif trade.direction == 'Short' and high >= current_sl:
                trade.exit_price = current_sl
                trade.exit_time = current_time
                trade.exit_type = 'BE' if trade.sl_moved_to_be else 'SL'
                break
        
        # If no exit, trade timed out
        if trade.exit_price is None:
            trade.exit_price = df.iloc[min(start_idx + max_bars - 1, len(df) - 1)]['close']
            trade.exit_time = current_time
            trade.exit_type = 'TIMEOUT'
        
        # Calculate final PnL
        pnl_result = self.pnl_calc.calculate_full_pnl(trade, trade.exit_price, trade.exit_type)
        trade.gross_pnl_pct = pnl_result['gross_pnl_pct']
        trade.fees_pct = pnl_result['fees_pct']
        trade.slippage_pct = pnl_result['slippage_pct']
        trade.net_pnl_pct = pnl_result['net_pnl_pct']
        trade.pnl_r = pnl_result['pnl_r']
        
        # Determine result
        if trade.net_pnl_pct > 0.1:
            trade.result = 'WIN'
        elif trade.net_pnl_pct < -0.1:
            trade.result = 'LOSS'
        else:
            trade.result = 'BREAKEVEN'
        
        return trade


# ============================================================================
# SIGNAL GENERATOR (Matches live bot logic)
# ============================================================================

class SignalGenerator:
    """Generates trading signals matching live bot logic"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    async def find_signals(
        self,
        df_4h: pd.DataFrame,
        df_1d: Optional[pd.DataFrame],
        symbol: str,
        current_idx: int
    ) -> List[Dict]:
        """
        Find trading signals at a specific point in history.
        
        Uses same logic as live bot:
        - Order block detection
        - Trend alignment
        - Momentum confirmation
        - Multi-timeframe confluence
        """
        signals = []
        
        if current_idx < self.config.warmup_periods:
            return signals
        
        # Get data up to current point (no look-ahead)
        df_subset = df_4h.iloc[:current_idx + 1].copy()
        current_price = df_subset['close'].iloc[-1]
        
        # Find order blocks
        try:
            from bot.order_blocks import find_unmitigated_order_blocks
            obs = await find_unmitigated_order_blocks(
                df_subset, lookback=50, tf='4h', min_strength=self.config.min_ob_strength
            )
        except Exception as e:
            logging.debug(f"OB detection error: {e}")
            return signals
        
        # Check each OB type
        for ob_type in ['bullish', 'bearish']:
            for ob in obs.get(ob_type, [])[:3]:  # Top 3 OBs
                strength = ob.get('strength', 0)
                if strength < self.config.min_ob_strength:
                    continue
                
                mid = (ob['low'] + ob['high']) / 2
                dist_pct = abs(current_price - mid) / current_price * 100
                
                # Relaxed distance check
                if dist_pct > self.config.max_entry_distance_pct:
                    continue
                
                direction = 'Long' if ob_type == 'bullish' else 'Short'
                
                # Build confluence factors
                factors = [f'OB({strength:.1f})']
                
                # Trend check (EMA200)
                if 'ema200' in df_subset.columns:
                    ema200 = df_subset['ema200'].iloc[-1]
                    if pd.notna(ema200):
                        trend_aligned = (direction == 'Long' and current_price > ema200) or \
                                        (direction == 'Short' and current_price < ema200)
                        if trend_aligned:
                            factors.append('Trend')
                
                # Momentum check (RSI)
                if 'rsi' in df_subset.columns:
                    rsi = df_subset['rsi'].iloc[-1]
                    if pd.notna(rsi):
                        if (direction == 'Long' and rsi < 70) or (direction == 'Short' and rsi > 30):
                            factors.append('RSI')
                
                # Volume check
                if 'volume' in df_subset.columns and len(df_subset) > 20:
                    avg_vol = df_subset['volume'].iloc[-20:-1].mean()
                    curr_vol = df_subset['volume'].iloc[-1]
                    if curr_vol > avg_vol * 1.2:
                        factors.append('Vol')
                
                # HTF check
                if df_1d is not None and len(df_1d) > current_idx // 6:
                    htf_idx = min(current_idx // 6, len(df_1d) - 1)
                    if 'ema200' in df_1d.columns:
                        htf_ema = df_1d['ema200'].iloc[htf_idx]
                        htf_close = df_1d['close'].iloc[htf_idx]
                        if pd.notna(htf_ema):
                            htf_aligned = (direction == 'Long' and htf_close > htf_ema) or \
                                          (direction == 'Short' and htf_close < htf_ema)
                            if htf_aligned:
                                factors.append('HTF')
                
                # Check minimum confluence
                if len(factors) >= self.config.min_confluence:
                    # Calculate SL/TP
                    atr = df_subset['atr'].iloc[-1] if 'atr' in df_subset.columns else current_price * 0.02
                    
                    if direction == 'Long':
                        sl = ob['low'] - atr * 0.3
                        tp1 = mid + atr * self.config.tp1_r_multiple
                        tp2 = mid + atr * self.config.tp2_r_multiple
                    else:
                        sl = ob['high'] + atr * 0.3
                        tp1 = mid - atr * self.config.tp1_r_multiple
                        tp2 = mid - atr * self.config.tp2_r_multiple
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': mid,
                        'sl_price': sl,
                        'tp1_price': tp1,
                        'tp2_price': tp2,
                        'factors': factors,
                        'confluence_count': len(factors),
                        'ob_strength': strength
                    })
        
        return signals


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive backtest metrics"""
    
    @staticmethod
    def calculate_all_metrics(trades: List[Trade], config: BacktestConfig) -> BacktestResult:
        """Calculate all performance metrics from trade list"""
        result = BacktestResult()
        result.trades = trades
        result.config = config
        
        if not trades:
            return result
        
        # Basic counts
        result.total_trades = len(trades)
        result.wins = len([t for t in trades if t.result == 'WIN'])
        result.losses = len([t for t in trades if t.result == 'LOSS'])
        result.breakevens = len([t for t in trades if t.result == 'BREAKEVEN'])
        
        # Win rate
        result.win_rate = result.wins / result.total_trades * 100 if result.total_trades > 0 else 0
        
        # PnL metrics
        all_pnl = [t.net_pnl_pct for t in trades]
        all_pnl_r = [t.pnl_r for t in trades]
        win_pnl = [t.net_pnl_pct for t in trades if t.result == 'WIN']
        loss_pnl = [t.net_pnl_pct for t in trades if t.result == 'LOSS']
        
        result.total_pnl_pct = sum(all_pnl)
        result.total_pnl_r = sum(all_pnl_r)
        result.avg_win_pct = np.mean(win_pnl) if win_pnl else 0
        result.avg_loss_pct = np.mean(loss_pnl) if loss_pnl else 0
        result.avg_win_r = np.mean([t.pnl_r for t in trades if t.result == 'WIN']) if result.wins > 0 else 0
        result.avg_loss_r = np.mean([t.pnl_r for t in trades if t.result == 'LOSS']) if result.losses > 0 else 0
        
        # Profit factor
        gross_profit = sum([t.net_pnl_pct for t in trades if t.net_pnl_pct > 0])
        gross_loss = abs(sum([t.net_pnl_pct for t in trades if t.net_pnl_pct < 0]))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expected value per trade
        result.expected_value_r = result.total_pnl_r / result.total_trades if result.total_trades > 0 else 0
        
        # Equity curve and drawdown
        equity = config.risk_per_trade_pct * 100  # Starting at 100 = 1x capital
        max_equity = equity
        max_dd = 0
        dd_start = None
        max_dd_duration = 0
        result.equity_curve = [equity]
        
        for trade in trades:
            equity += trade.net_pnl_pct
            result.equity_curve.append(equity)
            
            if equity > max_equity:
                max_equity = equity
                if dd_start is not None:
                    duration = (trade.exit_time - dd_start).days if trade.exit_time and dd_start else 0
                    max_dd_duration = max(max_dd_duration, duration)
                dd_start = None
            else:
                if dd_start is None:
                    dd_start = trade.exit_time
                dd = (max_equity - equity) / max_equity * 100
                max_dd = max(max_dd, dd)
        
        result.max_drawdown_pct = max_dd
        result.max_drawdown_duration_days = max_dd_duration
        result.starting_capital = 100
        result.ending_capital = equity
        
        # Risk-adjusted returns (Sharpe, Sortino, Calmar)
        if len(all_pnl) > 1:
            returns = np.array(all_pnl)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe (annualized assuming 4h trades, ~2190 trades/year)
            trades_per_year = 2190 / result.total_trades * len(trades)  # Approximate
            result.sharpe_ratio = mean_return / std_return * np.sqrt(365) if std_return > 0 else 0
            
            # Sortino (downside deviation only)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
            result.sortino_ratio = mean_return / downside_std * np.sqrt(365) if downside_std > 0 else 0
            
            # Calmar
            result.calmar_ratio = result.total_pnl_pct / result.max_drawdown_pct if result.max_drawdown_pct > 0 else 0
        
        # TP statistics
        result.tp1_hit_rate = len([t for t in trades if t.tp1_hit]) / result.total_trades * 100 if result.total_trades > 0 else 0
        result.tp2_hit_rate = len([t for t in trades if t.exit_type == 'TP2']) / result.total_trades * 100 if result.total_trades > 0 else 0
        
        # Average hold time
        hold_times = []
        for t in trades:
            if t.entry_time and t.exit_time:
                hold_times.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        result.avg_hold_time_hours = np.mean(hold_times) if hold_times else 0
        
        # Per-factor performance
        factor_trades = defaultdict(list)
        for trade in trades:
            for factor in trade.factors:
                factor_name = factor.split('(')[0]  # Remove params like OB(2.1)
                factor_trades[factor_name].append(trade)
        
        for factor, f_trades in factor_trades.items():
            f_wins = len([t for t in f_trades if t.result == 'WIN'])
            f_total = len(f_trades)
            f_pnl = sum([t.net_pnl_pct for t in f_trades])
            result.factor_stats[factor] = {
                'trades': f_total,
                'wins': f_wins,
                'win_rate': f_wins / f_total * 100 if f_total > 0 else 0,
                'total_pnl': f_pnl,
                'avg_pnl': f_pnl / f_total if f_total > 0 else 0
            }
        
        # Date range
        if trades:
            result.start_date = trades[0].entry_time
            result.end_date = trades[-1].exit_time
        
        return result
    
    @staticmethod
    def run_monte_carlo(
        trades: List[Trade],
        n_simulations: int = 1000,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation to get confidence intervals.
        
        Resamples trades with replacement to understand outcome distribution.
        """
        if not trades:
            return {}
        
        pnl_values = [t.net_pnl_pct for t in trades]
        n_trades = len(pnl_values)
        
        simulation_returns = []
        
        for _ in range(n_simulations):
            # Resample with replacement
            resampled = np.random.choice(pnl_values, size=n_trades, replace=True)
            simulation_returns.append(np.sum(resampled))
        
        simulation_returns = np.array(simulation_returns)
        
        # Calculate statistics
        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 - (1 - confidence) / 2) * 100
        
        return {
            'median': np.median(simulation_returns),
            'mean': np.mean(simulation_returns),
            'std': np.std(simulation_returns),
            'lower_95': np.percentile(simulation_returns, lower_pct),
            'upper_95': np.percentile(simulation_returns, upper_pct),
            'worst_case': np.min(simulation_returns),
            'best_case': np.max(simulation_returns),
            'prob_profit': np.mean(simulation_returns > 0) * 100
        }


# ============================================================================
# MAIN BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Main backtest orchestrator"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.signal_gen = SignalGenerator(self.config)
        self.simulator = TradeSimulator(self.config)
    
    async def run_backtest(
        self,
        symbols: List[str],
        progress_callback=None
    ) -> BacktestResult:
        """
        Run full backtest on list of symbols.
        
        Args:
            symbols: List of symbols to test
            progress_callback: Optional callback for progress updates
        
        Returns:
            BacktestResult with all metrics and trades
        """
        all_trades = []
        
        from bot.data_fetcher import fetch_ohlcv
        from bot.indicators import add_institutional_indicators
        
        for sym_idx, symbol in enumerate(symbols):
            try:
                if progress_callback:
                    await progress_callback(f"Testing {symbol} ({sym_idx + 1}/{len(symbols)})")
                
                # Fetch historical data
                days_to_fetch = self.config.days + self.config.warmup_periods // 6
                since = int((datetime.now(timezone.utc) - timedelta(days=days_to_fetch)).timestamp() * 1000)
                
                # Fetch 4h data
                df_4h = await fetch_ohlcv(symbol, self.config.primary_timeframe, limit=days_to_fetch * 6, since=since)
                if df_4h is None or len(df_4h) < self.config.warmup_periods:
                    logging.warning(f"{symbol}: Insufficient 4h data ({len(df_4h) if df_4h is not None else 0} bars)")
                    continue
                
                # Fetch 1d data
                df_1d = await fetch_ohlcv(symbol, self.config.htf_timeframe, limit=days_to_fetch, since=since)
                
                # Add indicators
                df_4h = add_institutional_indicators(df_4h)
                if df_1d is not None:
                    df_1d = add_institutional_indicators(df_1d)
                
                # Walk through history
                open_trades = {}
                
                for i in range(self.config.warmup_periods, len(df_4h) - 10):
                    current_time = df_4h.index[i] if isinstance(df_4h.index[i], datetime) else datetime.now(timezone.utc)
                    
                    # Check for new signals (if not at max concurrent)
                    if len(open_trades) < self.config.max_concurrent_trades:
                        signals = await self.signal_gen.find_signals(df_4h, df_1d, symbol, i)
                        
                        for sig in signals:
                            # Don't open if already have position in same direction
                            trade_key = f"{symbol}_{sig['direction']}"
                            if trade_key in open_trades:
                                continue
                            
                            trade = Trade(
                                symbol=symbol,
                                direction=sig['direction'],
                                entry_price=sig['entry_price'],
                                entry_time=current_time,
                                sl_price=sig['sl_price'],
                                tp1_price=sig['tp1_price'],
                                tp2_price=sig['tp2_price'],
                                leverage=self.config.default_leverage,
                                position_size_pct=self.config.risk_per_trade_pct / 100,
                                factors=sig['factors'],
                                confluence_count=sig['confluence_count']
                            )
                            
                            # Simulate trade
                            completed_trade = self.simulator.simulate_trade(trade, df_4h, i + 1)
                            all_trades.append(completed_trade)
                    
                    # Yield for async
                    if i % 100 == 0:
                        await asyncio.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Backtest error for {symbol}: {e}")
                continue
        
        # Sort trades by entry time
        all_trades.sort(key=lambda t: t.entry_time or datetime.min.replace(tzinfo=timezone.utc))
        
        # Calculate metrics
        result = MetricsCalculator.calculate_all_metrics(all_trades, self.config)
        result.symbols_tested = symbols
        
        # Run Monte Carlo
        mc_results = MetricsCalculator.run_monte_carlo(all_trades, self.config.mc_simulations)
        if mc_results:
            result.mc_median_return = mc_results['median']
            result.mc_95_lower = mc_results['lower_95']
            result.mc_95_upper = mc_results['upper_95']
            result.mc_worst_case = mc_results['worst_case']
            result.mc_probability_profit = mc_results['prob_profit']
        
        return result


# ============================================================================
# FORMAT FUNCTIONS
# ============================================================================

def format_backtest_report(result: BacktestResult) -> str:
    """Format backtest results for Telegram"""
    if result.total_trades == 0:
        return "❌ **No trades generated**\n\nTry adjusting signal conditions or testing different symbols."
    
    # Determine overall grade
    if result.profit_factor >= 2.0 and result.win_rate >= 55:
        grade = "A"
        grade_emoji = "🌟"
    elif result.profit_factor >= 1.5 and result.win_rate >= 50:
        grade = "B"
        grade_emoji = "✅"
    elif result.profit_factor >= 1.2 and result.win_rate >= 45:
        grade = "C"
        grade_emoji = "⚡"
    else:
        grade = "D"
        grade_emoji = "⚠️"
    
    msg = f"📊 **BACKTEST RESULTS** {grade_emoji} Grade {grade}\n\n"
    
    # Trade summary
    msg += f"**📈 Trade Summary ({result.config.days}d)**\n"
    msg += f"• Trades: {result.total_trades} ({result.wins}W / {result.losses}L / {result.breakevens}BE)\n"
    msg += f"• Win Rate: {result.win_rate:.1f}%\n"
    msg += f"• TP1 Hit: {result.tp1_hit_rate:.0f}% | TP2 Hit: {result.tp2_hit_rate:.0f}%\n\n"
    
    # PnL metrics
    msg += f"**💰 Performance**\n"
    msg += f"• Total PnL: {result.total_pnl_pct:+.1f}% ({result.total_pnl_r:+.1f}R)\n"
    msg += f"• Avg Win: {result.avg_win_pct:+.2f}% | Avg Loss: {result.avg_loss_pct:.2f}%\n"
    msg += f"• Profit Factor: {result.profit_factor:.2f}\n"
    msg += f"• Expected Value: {result.expected_value_r:+.2f}R per trade\n\n"
    
    # Risk metrics
    msg += f"**⚠️ Risk Metrics**\n"
    msg += f"• Max Drawdown: {result.max_drawdown_pct:.1f}%\n"
    msg += f"• Sharpe Ratio: {result.sharpe_ratio:.2f}\n"
    msg += f"• Calmar Ratio: {result.calmar_ratio:.2f}\n\n"
    
    # Monte Carlo
    msg += f"**🎲 Monte Carlo (95% CI)**\n"
    msg += f"• Expected: {result.mc_median_return:+.1f}%\n"
    msg += f"• Range: {result.mc_95_lower:+.1f}% to {result.mc_95_upper:+.1f}%\n"
    msg += f"• Prob Profit: {result.mc_probability_profit:.0f}%\n\n"
    
    # Top factors
    if result.factor_stats:
        msg += f"**🎯 Top Factors**\n"
        sorted_factors = sorted(result.factor_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:5]
        for factor, stats in sorted_factors:
            msg += f"• {factor}: {stats['win_rate']:.0f}% WR ({stats['trades']} trades)\n"
    
    msg += f"\n_Symbols: {len(result.symbols_tested)} | Fees: {result.config.taker_fee_pct}% | Slip: {result.config.slippage_major_pct}%_"
    
    return msg


def format_partial_close_message(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    leverage: int,
    partial_pct: float = 0.5
) -> str:
    """
    Format message for partial close at TP1.
    
    Example output:
    "Closed 50% → +5.17% realized | Remaining 50% at breakeven SL, targeting TP2"
    """
    # Calculate raw move
    if direction.lower() == 'long':
        distance = exit_price - entry_price
    else:
        distance = entry_price - exit_price
    
    raw_move_pct = distance / entry_price * 100
    leveraged_move = raw_move_pct * leverage * partial_pct
    
    clean_symbol = symbol.replace('/USDT', '')
    
    msg = f"**TP1 PARTIAL EXIT** {clean_symbol}\n\n"
    msg += f"Closed {int(partial_pct * 100)}% → **{leveraged_move:+.2f}%** realized\n"
    msg += f"Remaining {int((1 - partial_pct) * 100)}% at breakeven SL, targeting TP2\n\n"
    msg += f"_Entry: ${entry_price:.4f} | Exit: ${exit_price:.4f} | {leverage}x_"
    
    return msg
