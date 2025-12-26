# main.py - Grok Elite Signal Bot v27.12.12 - Main Orchestration
# -*- coding: utf-8 -*-
"""
Main entry point - Claude-only signal generation with full module integration.

v27.12.12 UPDATES - RENDER LOGGING FIX:
1. FIXED: All logging now properly outputs to stdout for Render visibility
2. NEW: Periodic heartbeat logging during long waits
3. NEW: Enhanced signal cycle logging with detailed activity
4. NEW: Track callback now logs even when no trades
5. NEW: flush_logs() called after all important operations
6. All v27.12.11 features preserved

v27.12.11 UPDATES:
1. NEW: Blofin Auto-Trading Integration
2. NEW: execute_trade_signal() called on valid signals
3. NEW: /blofin, /blofin_toggle, /blofin_close Telegram commands
4. NEW: Blofin status in welcome message
5. NEW: initialize_blofin() in deferred_init
6. All v27.12.10 features preserved

v27.12.10 UPDATES:
1. Added manipulation detector imports with proper try/except
2. MANIPULATION_AVAILABLE flag for runtime checks
3. Verified config imports include MANIPULATION_DETECTION_ENABLED
4. All v27.12.9 fixes preserved
"""
import asyncio
import sys
import os
import logging
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import Update
from telegram.ext import ContextTypes

# ============================================================================
# EMOJI CONSTANTS (Unicode escape sequences for proper encoding)
# ============================================================================
EMOJI = {
    'rocket': '\U0001F680',
    'chart': '\U0001F4C8',
    'chart_down': '\U0001F4C9',
    'money': '\U0001F4B0',
    'paper': '\U0001F4DD',
    'warning': '\u26A0\uFE0F',
    'check': '\u2705',
    'cross': '\u274C',
    'star': '\U0001F31F',
    'lightning': '\u26A1',
    'fire': '\U0001F525',
    'target': '\U0001F3AF',
    'pin': '\U0001F4CD',
    'stop': '\U0001F6D1',
    'alarm': '\U0001F6A8',
    'candle': '\U0001F56F',
    'green': '\U0001F7E2',
    'red': '\U0001F534',
    'yellow': '\U0001F7E1',
    'white': '\u26AA',
    'gem': '\U0001F48E',
    'brain': '\U0001F9E0',
    'eyes': '\U0001F440',
    'muscle': '\U0001F4AA',
    'clock': '\U0001F551',
    'calendar': '\U0001F4C5',
    'graph': '\U0001F4CA',
    'bulb': '\U0001F4A1',
    'gear': '\u2699\uFE0F',
    'link': '\U0001F517',
    'wave': '\U0001F44B',
    'bullet': '\u2022',
}

# ============================================================================
# CONFIG IMPORTS
# ============================================================================
from bot.config import (
    TELEGRAM_TOKEN, CHAT_ID, PORT, RENDER_EXTERNAL_HOSTNAME,
    CHECK_INTERVAL, TRACK_INTERVAL, SYMBOLS, TIMEFRAMES,
    MAX_CONCURRENT_TRADES, MAX_DRAWDOWN_PCT, BOT_VERSION,
    SIMULATED_CAPITAL, PAPER_TRADING,
    VOL_SPIKE_ENABLED, USE_STRUCTURE_SL, OB_SCORING_ENABLED,
    OB_MIN_QUALITY_SCORE, COUNTER_TREND_TP1_ONLY,
    SIGNAL_GRADING_ENABLED, GROK_OPINION_ENABLED,
    STRUCTURE_DETECTION_ENABLED, PSYCHOLOGY_ENABLED,
    MIN_CONFLUENCE_LIVE, MIN_CONFLUENCE_ROADMAP, EXECUTABLE_GRADES,
    MANIPULATION_DETECTION_ENABLED  # v27.12.10: Added
)

# v27.12.11: Blofin config imports
try:
    from bot.config import (
        AUTO_TRADE_ENABLED, AUTO_TRADE_MIN_GRADE,
        is_blofin_configured, get_blofin_config_summary
    )
except ImportError:
    AUTO_TRADE_ENABLED = False
    AUTO_TRADE_MIN_GRADE = "B"
    def is_blofin_configured(): return False
    def get_blofin_config_summary(): return "Blofin: Not configured"

# Safe config imports with fallbacks
try:
    from bot.config import (
        DYNAMIC_COOLDOWN_MAP, FLAG_FILE, RECAP_FILE, validate_config,
        VOL_SURGE_MULTIPLIER, ROADMAP_ENTRY_PROXIMITY_PCT,
        CLAUDE_MIN_CONFIDENCE, CLAUDE_MIN_STRENGTH,
        MIN_CONFLUENCE_FACTORS, MIN_TRADE_QUALITY_SCORE
    )
except ImportError:
    DYNAMIC_COOLDOWN_MAP = {'1h': 2, '4h': 4, '1d': 8}
    FLAG_FILE = "data/welcome_sent.flag"
    RECAP_FILE = "data/last_recap.txt"
    def validate_config(): pass
    VOL_SURGE_MULTIPLIER = 3.0
    ROADMAP_ENTRY_PROXIMITY_PCT = 0.5
    CLAUDE_MIN_CONFIDENCE = 60
    CLAUDE_MIN_STRENGTH = 2.0
    MIN_CONFLUENCE_FACTORS = 4
    MIN_TRADE_QUALITY_SCORE = 55

# ============================================================================
# MODULE IMPORTS
# ============================================================================
from bot.utils import (
    setup_logging, send_throttled, BanManager, get_session_info,
    extract_factors_from_reason, is_roadmap_generation_time, format_price,
    flush_logs, log_with_flush
)

try:
    from bot.utils import (
        zones_overlap, validate_grok_trade, get_dynamic_check_interval,
        format_validation_failure
    )
except ImportError:
    def zones_overlap(*args, **kwargs): return 0
    def validate_grok_trade(trade, price, symbol): return True, "OK"
    def get_dynamic_check_interval(*args): return 7200
    def format_validation_failure(*args): return "Validation failed"

from bot.models import (
    load_stats, save_stats_async, load_trades, save_trades_async,
    load_protected, save_protected_async, HISTORICAL_DATA
)

from bot.data_fetcher import (
    fetch_ohlcv, fetch_ticker_batch, fetch_order_flow_batch,
    fetch_open_interest, price_background_task, close_exchanges,
    analyze_orderbook_imbalance  # v27.12.14: Orderbook analysis
)

try:
    from bot.data_fetcher import (
        calculate_exchange_divergence, get_combined_funding, get_volume_comparison
    )
except ImportError:
    async def calculate_exchange_divergence(*args): return None
    async def get_combined_funding(*args): return None
    async def get_volume_comparison(*args): return None

from bot.indicators import (
    add_institutional_indicators, detect_market_regime, get_current_volatility
)

try:
    from bot.indicators import (
        is_consolidation, detect_fvg, detect_candle_patterns,
        detect_divergence, detect_pre_cross, detect_liquidity_sweep,
        calculate_order_flow, get_momentum_confluence, get_trend_strength
    )
except ImportError:
    def is_consolidation(*args): return False
    def detect_fvg(*args): return []
    def detect_candle_patterns(*args): return {}
    def detect_divergence(*args): return None
    def detect_pre_cross(*args): return None
    def detect_liquidity_sweep(*args): return pd.Series()
    def calculate_order_flow(df, *args): return df
    def get_momentum_confluence(*args): return {}
    def get_trend_strength(*args): return 0

from bot.order_blocks import find_unmitigated_order_blocks, find_next_premium_zones

try:
    from bot.order_blocks import calculate_mtf_confluence
except ImportError:
    async def calculate_mtf_confluence(*args): return 0

from bot.grok_api import query_grok_daily_recap

try:
    from bot.grok_api import get_grok_opinion, get_grok_opinion_boost
except ImportError:
    async def get_grok_opinion(*args, **kwargs):
        return {'opinion': 'neutral', 'reason': 'Not available', 'display': ''}
    def get_grok_opinion_boost(opinion): return 0

from bot.claude_api import query_claude_analysis, claude_health_check

from bot.trading import (
    process_trade, get_risk_scaling_factor, check_portfolio_correlation,
    calculate_expected_value, get_atr_values, factor_tracker
)

# Signal evaluator (unified scoring)
try:
    from bot.signal_evaluator import (
        SignalEvaluator, evaluate_signal, format_grade_display,
        build_confluence_data, build_ob_data, build_psychology_data,
        build_structure_data, build_trade_params
    )
    UNIFIED_EVALUATOR_AVAILABLE = True
except ImportError:
    UNIFIED_EVALUATOR_AVAILABLE = False
    logging.warning("Unified SignalEvaluator not available")

# Phase 2 trading functions
try:
    from bot.trading import (
        grade_signal, detect_structure_break, get_structure_confluence,
        get_psychology_confluence, get_fear_greed, get_long_short_ratio
    )
except ImportError:
    def grade_signal(**kwargs):
        return {'grade': 'B', 'score': 75, 'executable': True, 'size_mult': 1.0, 'reasons': []}
    def detect_structure_break(*args): return None
    def get_structure_confluence(*args): return False, ""
    async def get_psychology_confluence(*args): return 0, ""
    async def get_fear_greed(): return None
    async def get_long_short_ratio(*args): return None

# Structure breaker imports
try:
    from bot.structure_breaker import (
        detect_stop_hunt, detect_fake_breakout, get_structure_confluence as sb_get_structure_confluence
    )
    STRUCTURE_BREAKER_AVAILABLE = True
except ImportError:
    STRUCTURE_BREAKER_AVAILABLE = False
    def detect_stop_hunt(*args): return None
    def detect_fake_breakout(*args): return None

# v27.11.0: Wick detector imports
try:
    from bot.wick_detector import (
        WickDetector, detect_euphoria_capitulation,
        format_wick_for_telegram, get_wick_confluence
    )
    WICK_DETECTOR_AVAILABLE = True
except ImportError:
    WICK_DETECTOR_AVAILABLE = False
    logging.warning("Wick detector not available")

# Volatility monitor
try:
    from bot.volatility_monitor import vol_monitor
except ImportError:
    class DummyVolMonitor:
        async def check_for_spikes(self, *args): return None
    vol_monitor = DummyVolMonitor()

# Dynamic SL
try:
    from bot.dynamic_sl import calculate_structure_based_sl
except ImportError:
    async def calculate_structure_based_sl(*args): return None

# OB scorer
try:
    from bot.ob_scorer import select_best_ob, filter_by_strength
except ImportError:
    def select_best_ob(zones, *args, **kwargs): return zones[:3]
    def filter_by_strength(zones, min_strength=2.0):
        return [z for z in zones if z.get('strength', 0) >= min_strength]

# OTE module
try:
    from bot.ote import analyze_ote, format_ote_info
    OTE_AVAILABLE = True
except ImportError:
    OTE_AVAILABLE = False
    def analyze_ote(*args, **kwargs): return False, None, "OTE not loaded"
    def format_ote_info(*args, **kwargs): return ""

# Daily recap
try:
    from bot.daily_recap import (
        daily_callback_fixed, build_previous_day_summary,
        fetch_crypto_news_summary, generate_direct_recap
    )
    FIXED_RECAP_AVAILABLE = True
except ImportError:
    FIXED_RECAP_AVAILABLE = False

# Telegram commands
from bot.telegram_commands import (
    stats_cmd, health_cmd, recap_cmd, backtest_cmd,
    backtest_all_cmd, validate_cmd, dashboard_cmd, set_trade_dicts,
    roadmap_cmd, zones_cmd, commands_cmd
)

try:
    from bot.telegram_commands import structural_cmd
except ImportError:
    async def structural_cmd(update, context):
        await update.message.reply_text("Structural command not available")

# v27.12.13: Advanced analytics commands
try:
    from bot.telegram_commands import factor_analysis_cmd, system_health_cmd
except ImportError:
    async def factor_analysis_cmd(update, context):
        await update.message.reply_text("Factor analysis not available")
    async def system_health_cmd(update, context):
        await update.message.reply_text("System health not available")

# v27.12.15: Liquidity and orderbook commands
try:
    from bot.telegram_commands import liquidity_cmd, orderbook_cmd
except ImportError:
    async def liquidity_cmd(update, context):
        await update.message.reply_text("Liquidity command not available")
    async def orderbook_cmd(update, context):
        await update.message.reply_text("Orderbook command not available")

# Roadmap imports
from bot.roadmap import (
    roadmap_zones, initialize_roadmaps, roadmap_generation_callback,
    monitor_roadmap_proximity, send_proximity_alert,
    validate_roadmap_for_conversion, convert_roadmap_to_live
)

# Structural bounce
try:
    from bot.structural_bounce import detect_structural_bounces_batch, validate_structural_zone
except ImportError:
    async def detect_structural_bounces_batch(*args): return {}
    def validate_structural_zone(*args): return False, {}

# v27.12.10: Manipulation detector imports
try:
    from bot.manipulation import manipulation_detector, ManipulationDetector
    MANIPULATION_AVAILABLE = True
    logging.info("Manipulation detector loaded successfully")
except ImportError:
    MANIPULATION_AVAILABLE = False
    manipulation_detector = None
    logging.warning("Manipulation detector not available")

# v27.12.14: Early Reversal Detection imports
try:
    from bot.early_reversal import detect_early_reversal, get_reversal_confluence_factors
    EARLY_REVERSAL_AVAILABLE = True
    logging.info("Early reversal detection loaded successfully")
except ImportError:
    EARLY_REVERSAL_AVAILABLE = False
    def detect_early_reversal(*args, **kwargs): return None
    def get_reversal_confluence_factors(*args): return []
    logging.warning("Early reversal detection not available")

# ============================================================================
# v27.12.11: BLOFIN AUTO-TRADING IMPORTS
# ============================================================================
try:
    from bot.blofin_trader import (
        BlofinAutoTrader,
        execute_trade_signal,
        get_auto_trader,
        BlofinClient
    )
    BLOFIN_AVAILABLE = True
    logging.info("Blofin auto-trading module loaded successfully")
except ImportError:
    BLOFIN_AVAILABLE = False
    logging.warning("Blofin auto-trading module not available")
    
    # Fallback implementations
    async def execute_trade_signal(signal): return None
    async def get_auto_trader(): return None
    class BlofinAutoTrader:
        pass

# Global Blofin trader instance
blofin_trader: Optional[BlofinAutoTrader] = None

# ============================================================================
# v27.12.14: ADVANCED FEATURES (Dynamic TP, Liquidity Map)
# ============================================================================
try:
    from bot.advanced_features import (
        detect_volatility_regime,
        calculate_dynamic_tp,
        enhance_signal_with_advanced_features,
        calculate_mtf_confluence as calculate_mtf_confluence_weighted  # v27.12.15: New weighted MTF
    )
    from bot.config import DYNAMIC_TP_ENABLED
    DYNAMIC_TP_AVAILABLE = True
    logging.info("Advanced features (Dynamic TP, MTF Weighting) loaded successfully")
except ImportError:
    DYNAMIC_TP_AVAILABLE = False
    DYNAMIC_TP_ENABLED = False
    def detect_volatility_regime(*args): return None
    def calculate_dynamic_tp(*args, **kwargs): return None
    def enhance_signal_with_advanced_features(*args, **kwargs): return args[0] if args else {}
    def calculate_mtf_confluence_weighted(*args, **kwargs): return None
    logging.warning("Advanced features not available")

try:
    from bot.liquidity_map import generate_liquidity_map, get_liquidity_signal
    LIQUIDITY_MAP_AVAILABLE = True
    logging.info("Liquidity map module loaded successfully")
except ImportError:
    LIQUIDITY_MAP_AVAILABLE = False
    def generate_liquidity_map(*args, **kwargs): return None
    def get_liquidity_signal(*args, **kwargs): return {'adjustment': 0, 'reason': 'N/A'}
    logging.warning("Liquidity map not available")

# ============================================================================
# GLOBAL STATE
# ============================================================================
stats = load_stats()
open_trades = load_trades()
protected_trades = load_protected()
last_signal_time: Dict[str, datetime] = {}
last_trade_result: Dict[str, str] = {}
prices_global: Dict[str, Optional[float]] = {s: None for s in SYMBOLS}
btc_trend_global = "Unknown"
background_task = None
stats_lock = asyncio.Lock()
data_cache: Dict[str, Dict] = {}

# v27.11.0: Initialize wick detector
wick_detector = WickDetector() if WICK_DETECTOR_AVAILABLE else None

# Set trade dicts for telegram commands
set_trade_dicts(open_trades, protected_trades)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_emoji(name: str) -> str:
    """Get emoji by name with fallback."""
    return EMOJI.get(name, '')


def safe_get_momentum_signal(momentum_data: Optional[Dict]) -> Tuple[str, float]:
    """Safely extract momentum signal with null handling."""
    if not momentum_data:
        return 'neutral', 0.0
    signal = momentum_data.get('signal')
    roc = momentum_data.get('roc', 0)
    if signal is None or (isinstance(signal, float) and pd.isna(signal)):
        signal = 'neutral'
    if pd.isna(roc) if isinstance(roc, float) else False:
        roc = 0.0
    return str(signal).lower(), float(roc)


def get_cooldown_hours(symbol: str, timeframe: str) -> int:
    """Get dynamic cooldown based on recent performance."""
    base_cooldown = DYNAMIC_COOLDOWN_MAP.get(timeframe, 4)
    last_result = last_trade_result.get(symbol)
    if last_result == 'WIN':
        return max(1, base_cooldown - 1)
    elif last_result == 'LOSS':
        return base_cooldown + 2
    return base_cooldown


def count_confluence_factors(
    trade: Dict,
    momentum_data: Optional[Dict],
    divergence_data: Optional[Dict],
    funding_data: Optional[Dict],
    oi_data: Optional[Dict],
    volume_comparison: Optional[Dict],
    timeframe_agreement: int,
    has_ote: bool,
    structure_aligned: bool = False,
    psych_aligned: bool = False,
    wick_detected: bool = False,
    reversal_signal: Optional[Dict] = None,  # v27.12.14: Early reversal
    orderbook_signal: Optional[Dict] = None  # v27.12.14: Orderbook imbalance
) -> Tuple[int, List[str]]:
    """Count total confluence factors for a trade."""
    count = 0
    factors = []
    
    # v27.12.11: Sanitize inputs - ensure they are dicts, not exceptions
    def safe_dict(obj):
        return obj if isinstance(obj, dict) else None
    
    trade = safe_dict(trade) or {}
    momentum_data = safe_dict(momentum_data)
    divergence_data = safe_dict(divergence_data)
    funding_data = safe_dict(funding_data)
    oi_data = safe_dict(oi_data)
    volume_comparison = safe_dict(volume_comparison)

    zone_confluence = trade.get('confluence', '')
    if 'OB' in zone_confluence or 'Order Block' in zone_confluence:
        count += 1
        factors.append('OB')
    if 'FVG' in zone_confluence:
        count += 1
        factors.append('FVG')
    if 'Sweep' in zone_confluence or 'Liquidity' in zone_confluence:
        count += 1
        factors.append('Sweep')

    if momentum_data:
        mom_signal, roc = safe_get_momentum_signal(momentum_data)
        if mom_signal in ['bullish', 'bearish'] and abs(roc) > 2:
            count += 1
            factors.append(f'Mom({mom_signal[:4]})')

    if divergence_data and divergence_data.get('significant'):
        count += 1
        factors.append(f"Div({divergence_data.get('signal', 'N/A')[:4]})")

    if funding_data:
        sentiment = funding_data.get('sentiment', '')
        if 'extreme' in sentiment.lower():
            count += 1
            factors.append('ExtFund')

    if oi_data:
        oi_change = oi_data.get('oi_change_pct', 0)
        if abs(oi_change) > 5:
            count += 1
            factors.append(f'OI({oi_change:+.0f}%)')

    if volume_comparison:
        dominant = volume_comparison.get('dominant', '')
        if dominant in ['bybit', 'binance']:
            count += 1
            factors.append(f'Vol({dominant[:3]})')

    if timeframe_agreement >= 2:
        count += 1
        factors.append(f'MTF({timeframe_agreement})')

    if has_ote:
        count += 1
        factors.append('OTE')

    if structure_aligned:
        count += 1
        factors.append('Struct')

    if psych_aligned:
        count += 1
        factors.append('Psych')

    if wick_detected:
        count += 1
        factors.append('Wick')

    # v27.12.14: Early reversal detection factors
    if reversal_signal:
        # Add divergence factors
        for div in reversal_signal.get('divergences', []):
            if 'regular' in div.get('divergence_type', ''):
                count += 2  # Regular divergences are strongest
                factors.append(f"Div({div.get('indicator', 'RSI')})")
            elif 'hidden' in div.get('divergence_type', ''):
                count += 1
                factors.append(f"HidDiv")
        
        # Add candlestick pattern factors
        for pattern in reversal_signal.get('candlestick_patterns', []):
            count += 1
            factors.append(pattern.get('pattern_name', 'Candle')[:6])
        
        # Add chart pattern factors
        for pattern in reversal_signal.get('chart_patterns', []):
            count += 2  # Chart patterns are significant
            factors.append(pattern.get('pattern_name', 'Pattern')[:6])

    # v27.12.14: Orderbook imbalance confluence
    if orderbook_signal:
        direction = trade.get('direction', '').upper()
        ob_signal = orderbook_signal.get('signal', 'NEUTRAL')
        ob_confidence = orderbook_signal.get('confidence', 0)
        
        # Add confluence if orderbook aligns with trade direction
        if ob_signal == direction and ob_confidence >= 60:
            count += 1
            ratio = orderbook_signal.get('imbalance_ratio', 1.0)
            factors.append(f"OB({ratio:.1f})")
        
        # Extra point for walls supporting the trade
        if direction == 'LONG' and orderbook_signal.get('bid_wall'):
            count += 1
            factors.append("BidWall")
        elif direction == 'SHORT' and orderbook_signal.get('ask_wall'):
            count += 1
            factors.append("AskWall")

    return count, factors


async def fetch_cross_exchange_data_fast(symbol: str) -> Dict:
    """Fetch cross-exchange data with parallel requests."""
    result = {
        'divergence_data': None,
        'funding_data': None,
        'volume_comparison': None
    }
    try:
        tasks = [
            asyncio.wait_for(calculate_exchange_divergence(symbol), timeout=5.0),
            asyncio.wait_for(get_combined_funding(symbol), timeout=5.0),
            asyncio.wait_for(get_volume_comparison(symbol), timeout=5.0)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # v27.12.11: Check for BaseException (includes CancelledError) not just Exception
        if not isinstance(results[0], BaseException):
            result['divergence_data'] = results[0]
        if not isinstance(results[1], BaseException):
            result['funding_data'] = results[1]
        if not isinstance(results[2], BaseException):
            result['volume_comparison'] = results[2]
    except asyncio.CancelledError:
        logging.debug(f"{symbol}: Cross-exchange fetch cancelled")
    except Exception as e:
        logging.debug(f"{symbol}: Cross-exchange fetch error: {e}")
    return result


# ============================================================================
# v27.12.11: BLOFIN AUTO-TRADING FUNCTIONS
# ============================================================================

async def initialize_blofin():
    """Initialize Blofin auto-trading if configured."""
    global blofin_trader
    
    if not BLOFIN_AVAILABLE:
        logging.info("Blofin module not available")
        return False
    
    try:
        if not is_blofin_configured():
            logging.info("Blofin API not configured - auto-trading disabled")
            return False
        
        if not AUTO_TRADE_ENABLED:
            logging.info("Auto-trading is disabled in config")
            return False
        
        blofin_trader = await get_auto_trader()
        if blofin_trader and blofin_trader.config.auto_trade_enabled:
            logging.info(f"{get_emoji('target')} Blofin Auto-Trader initialized successfully")
            return True
        return False
        
    except Exception as e:
        logging.error(f"Failed to initialize Blofin: {e}")
        return False


async def process_signal_auto_trade(trade_data: Dict, symbol: str):
    """
    Process a generated signal and execute via Blofin if enabled.
    
    Args:
        trade_data: The trade dictionary from signal generation
        symbol: Trading pair (e.g., "BTC/USDT")
    """
    global blofin_trader
    
    if not blofin_trader or not AUTO_TRADE_ENABLED:
        return
    
    grade = trade_data.get('grade', 'C')
    
    # Check grade threshold
    valid_grades = {"A": 3, "B": 2, "C": 1, "D": 0}
    if valid_grades.get(grade, 0) < valid_grades.get(AUTO_TRADE_MIN_GRADE, 2):
        logging.debug(f"Signal grade {grade} below minimum {AUTO_TRADE_MIN_GRADE} - skipping auto-trade")
        return
    
    try:
        # Build signal for Blofin execution
        trade_signal = {
            "symbol": symbol,
            "direction": trade_data.get('direction', '').upper(),
            "entry": trade_data.get('entry_price') or ((trade_data.get('entry_low', 0) + trade_data.get('entry_high', 0)) / 2),
            "sl": trade_data.get('sl'),
            "tp1": trade_data.get('tp1'),
            "tp2": trade_data.get('tp2'),
            "confidence": trade_data.get('confidence', 0),
            "grade": grade
        }
        
        # Validate required fields
        if not all([trade_signal['direction'], trade_signal['entry'], 
                    trade_signal['sl'], trade_signal['tp1']]):
            logging.warning(f"Signal missing required fields for auto-trade: {trade_signal}")
            return
        
        execution_result = await execute_trade_signal(trade_signal)
        
        if execution_result:
            logging.info(f"{get_emoji('target')} Auto-trade executed: {execution_result.get('order_id')}")
            await send_auto_trade_notification(execution_result)
        
    except Exception as e:
        logging.error(f"Auto-trade execution failed: {e}")
        import traceback
        logging.error(traceback.format_exc())


async def send_auto_trade_notification(trade_result: Dict):
    """Send Telegram notification for auto-executed trade."""
    try:
        entry_str = f"${trade_result['entry']}"
        sl_str = f"${trade_result['sl']}"
        tp_str = f"${trade_result['tp1']}"
        
        message = (
            f"{get_emoji('target')} **AUTO-TRADE EXECUTED**\n\n"
            f"{get_emoji('chart')} **{trade_result['symbol']}** {trade_result['direction']}\n"
            f"{get_emoji('bullet')} Order ID: `{trade_result['order_id']}`\n"
            f"{get_emoji('bullet')} Size: {trade_result['size']} contracts\n"
            f"{get_emoji('bullet')} Entry: {entry_str}\n"
            f"{get_emoji('bullet')} Stop Loss: {sl_str}\n"
            f"{get_emoji('bullet')} Take Profit: {tp_str}\n"
            f"{get_emoji('bullet')} Leverage: {trade_result['leverage']}x\n"
            f"{get_emoji('bullet')} Grade: {trade_result['grade']}\n"
            f"{get_emoji('bullet')} Confidence: {trade_result['confidence']}%\n\n"
            f"{get_emoji('check')} Executed at {trade_result['timestamp'][:19]}"
        )
        await send_throttled(CHAT_ID, message, parse_mode='Markdown')
    except Exception as e:
        logging.error(f"Failed to send auto-trade notification: {e}")


# ============================================================================
# v27.12.11: BLOFIN TELEGRAM COMMANDS
# ============================================================================

async def blofin_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check Blofin auto-trading status - /blofin command"""
    global blofin_trader
    
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text(f"{get_emoji('cross')} Unauthorized")
        return
    
    if not BLOFIN_AVAILABLE or not blofin_trader:
        await update.message.reply_text(
            f"{get_emoji('cross')} Blofin Auto-Trading not configured\n\n"
            "Set these environment variables:\n"
            "• BLOFIN_API_KEY\n"
            "• BLOFIN_SECRET_KEY\n"
            "• BLOFIN_PASSPHRASE\n"
            "• AUTO_TRADE_ENABLED=true"
        )
        return
    
    try:
        balance = await blofin_trader.client.get_balance()
        positions = await blofin_trader.check_positions()
        
        equity = balance.get('totalEquity', '0')
        details = balance.get('details', [{}])
        available = details[0].get('available', '0') if details else '0'
        
        pos_text = ""
        for pos in positions:
            if float(pos.get('positions', 0)) != 0:
                pnl = pos.get('unrealizedPnl', '0')
                mark_price = pos['markPrice']
                pos_text += f"\n{get_emoji('bullet')} {pos['instId']}: {pos['positions']} @ {mark_price} (PnL: {pnl})"
        
        if not pos_text:
            pos_text = "\n" + get_emoji('bullet') + " No open positions"
        
        mode = "DEMO" if blofin_trader.config.demo_mode else "LIVE"
        status_emoji = get_emoji('check') if blofin_trader.config.auto_trade_enabled else get_emoji('cross')
        auto_status = 'ON' if blofin_trader.config.auto_trade_enabled else 'OFF'
        
        equity_str = f"${float(equity):,.2f}"
        available_str = f"${float(available):,.2f}"
        risk_str = f"{blofin_trader.config.risk_per_trade*100:.1f}%"
        
        message = (
            f"{get_emoji('graph')} **BLOFIN AUTO-TRADING STATUS**\n\n"
            f"{get_emoji('money')} **Account**\n"
            f"{get_emoji('bullet')} Equity: {equity_str}\n"
            f"{get_emoji('bullet')} Available: {available_str}\n"
            f"{get_emoji('bullet')} Mode: {mode}\n\n"
            f"{get_emoji('chart')} **Positions** {pos_text}\n\n"
            f"{get_emoji('gear')} **Settings**\n"
            f"{get_emoji('bullet')} Auto-Trade: {status_emoji} {auto_status}\n"
            f"{get_emoji('bullet')} Risk/Trade: {risk_str}\n"
            f"{get_emoji('bullet')} Leverage: {blofin_trader.config.default_leverage}x\n"
            f"{get_emoji('bullet')} Min Grade: {AUTO_TRADE_MIN_GRADE}\n"
            f"{get_emoji('bullet')} Margin: {blofin_trader.config.margin_mode}\n\n"
            f"{get_emoji('bullet')} Executed trades: {len(blofin_trader.executed_trades)}"
        )
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        await update.message.reply_text(f"{get_emoji('cross')} Error: {e}")


async def blofin_toggle_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle auto-trading on/off - /blofin_toggle command"""
    global blofin_trader
    
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text(f"{get_emoji('cross')} Unauthorized")
        return
    
    if not BLOFIN_AVAILABLE or not blofin_trader:
        await update.message.reply_text(f"{get_emoji('cross')} Blofin not configured")
        return
    
    blofin_trader.config.auto_trade_enabled = not blofin_trader.config.auto_trade_enabled
    status = f"{get_emoji('check')} ENABLED" if blofin_trader.config.auto_trade_enabled else f"{get_emoji('cross')} DISABLED"
    await update.message.reply_text(f"{get_emoji('target')} Auto-Trading: {status}")


async def blofin_close_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Close all Blofin positions - /blofin_close command"""
    global blofin_trader
    
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text(f"{get_emoji('cross')} Unauthorized")
        return
    
    if not BLOFIN_AVAILABLE or not blofin_trader:
        await update.message.reply_text(f"{get_emoji('cross')} Blofin not configured")
        return
    
    try:
        await update.message.reply_text(f"{get_emoji('warning')} Closing all positions...")
        results = await blofin_trader.close_all_positions()
        
        if not results:
            await update.message.reply_text(f"{get_emoji('check')} No positions to close")
        else:
            success = sum(1 for v in results.values() if v)
            failed = len(results) - success
            await update.message.reply_text(
                f"{get_emoji('check')} Closed {success} positions\n"
                f"{get_emoji('cross')} Failed: {failed} positions"
            )
    except Exception as e:
        await update.message.reply_text(f"{get_emoji('cross')} Error: {e}")


# ============================================================================
# SIGNAL CALLBACK - v27.12.12 WITH COMPREHENSIVE LOGGING
# ============================================================================
async def signal_callback(context):
    """Main signal generation loop with unified evaluator and wick detection."""
    global last_signal_time, open_trades, protected_trades, data_cache

    cycle_start = time.time()
    
    # v27.12.12: Comprehensive logging for Render visibility
    now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*70}", flush=True)
    print(f"[SIGNAL] ========== CYCLE START ==========", flush=True)
    print(f"[SIGNAL] Time: {now_str} UTC", flush=True)
    print(f"[SIGNAL] BTC Trend: {btc_trend_global}", flush=True)
    print(f"[SIGNAL] Open Trades: {len(open_trades)} | Protected: {len(protected_trades)}", flush=True)
    print(f"{'='*70}", flush=True)
    
    logging.info("=" * 70)
    logging.info(f"=== SIGNAL CYCLE START v{BOT_VERSION} ===")
    logging.info(f"BTC Trend: {btc_trend_global} | Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
    flush_logs()

    try:
        if await BanManager.check_and_sleep(skip_long_sleep=True):
            return

        btc_trend = btc_trend_global
        now = datetime.now(timezone.utc)

        prices = await fetch_ticker_batch()
        logging.info(f"Prices: {sum(1 for p in prices.values() if p)}/{len(SYMBOLS)}")
        print(f"[SIGNAL] Prices fetched: {sum(1 for p in prices.values() if p)}/{len(SYMBOLS)} symbols", flush=True)

        order_books = await fetch_order_flow_batch()
        logging.info(f"Order books: {len(order_books)}")
        print(f"[SIGNAL] Order books fetched: {len(order_books)}", flush=True)

        oi_tasks = [fetch_open_interest(s) for s in SYMBOLS]
        oi_results = await asyncio.gather(*oi_tasks, return_exceptions=True)
        oi_data_dict = {
            SYMBOLS[i]: r if not isinstance(r, Exception) else None
            for i, r in enumerate(oi_results)
        }

        # v27.12.11: Wrap async data fetching in try/except to prevent CancelledError propagation
        fear_greed_data = None
        long_short_data = None
        
        if PSYCHOLOGY_ENABLED:
            try:
                fear_greed_data = await get_fear_greed()
            except asyncio.CancelledError:
                logging.warning("Fear & Greed fetch cancelled")
                fear_greed_data = None
            except Exception as e:
                logging.debug(f"Fear & Greed fetch error: {e}")
                fear_greed_data = None

        signals_sent = 0
        analyzed = 0
        skipped = 0

        for symbol in SYMBOLS:
            try:
                print(f"[SIGNAL] Processing: {symbol}", flush=True)
                price = prices.get(symbol)
                if not price:
                    logging.warning(f"{symbol}: No price data")
                    skipped += 1
                    continue

                prices_global[symbol] = price

                if symbol in open_trades or symbol in protected_trades:
                    logging.debug(f"{symbol}: Position open, skipping")
                    skipped += 1
                    continue

                active_count = len(open_trades) + len(protected_trades)
                if active_count >= MAX_CONCURRENT_TRADES:
                    logging.info(f"Max concurrent trades ({MAX_CONCURRENT_TRADES}) reached")
                    break

                last_time = last_signal_time.get(symbol)
                if last_time:
                    cooldown_hours = get_cooldown_hours(symbol, '4h')
                    if (now - last_time).total_seconds() < cooldown_hours * 3600:
                        logging.debug(f"{symbol}: In cooldown")
                        skipped += 1
                        continue

                data = {}
                for tf in ['1h', '4h', '1d']:
                    df = await fetch_ohlcv(symbol, tf, 200)
                    if len(df) > 50:
                        df = add_institutional_indicators(df)
                        data[tf] = df

                if '4h' not in data or len(data['4h']) < 50:
                    logging.warning(f"{symbol}: Insufficient 4h data")
                    skipped += 1
                    continue

                data_cache[symbol] = data

                df_4h = data['4h']
                if 'ema200' in df_4h.columns and pd.notna(df_4h['ema200'].iloc[-1]):
                    if df_4h['close'].iloc[-1] > df_4h['ema200'].iloc[-1]:
                        trend = 'Uptrend'
                    elif df_4h['close'].iloc[-1] < df_4h['ema200'].iloc[-1]:
                        trend = 'Downtrend'
                    else:
                        trend = 'Sideways'
                else:
                    trend = 'Unknown'

                regime = detect_market_regime(df_4h) if len(df_4h) > 50 else 'unknown'

                obs = await find_unmitigated_order_blocks(df_4h, tf='4h', min_strength=1.5)
                oi_data = oi_data_dict.get(symbol)

                premium_zones = await find_next_premium_zones(
                    df_4h, price, '4h', symbol,
                    oi_data=oi_data, trend=trend,
                    min_strength=1.5
                )

                if not premium_zones:
                    logging.debug(f"{symbol}: No premium zones found")
                    skipped += 1
                    continue

                premium_zones = filter_by_strength(premium_zones, min_strength=2.0)
                if not premium_zones:
                    logging.debug(f"{symbol}: No zones meet strength threshold")
                    skipped += 1
                    continue

                cross_data = await fetch_cross_exchange_data_fast(symbol)
                divergence_data = cross_data['divergence_data']
                funding_data = cross_data['funding_data']
                volume_comparison = cross_data['volume_comparison']

                # v27.12.14: Orderbook imbalance analysis
                orderbook_signal = None
                if symbol in order_books:
                    orderbook_signal = analyze_orderbook_imbalance(order_books[symbol])
                    if orderbook_signal and orderbook_signal.get('signal') != 'NEUTRAL':
                        logging.info(f"{symbol}: Orderbook {orderbook_signal['signal']} (ratio: {orderbook_signal['imbalance_ratio']:.2f})")

                momentum_data = None
                if '1d' in data and len(data['1d']) > 0:
                    df_1d = data['1d']
                    if 'momentum_signal' in df_1d.columns:
                        mom_signal = df_1d['momentum_signal'].iloc[-1]
                        roc_val = df_1d['roc'].iloc[-1] if 'roc' in df_1d.columns else 0
                        if pd.notna(mom_signal):
                            momentum_data = {
                                'signal': str(mom_signal) if mom_signal else 'neutral',
                                'roc': float(roc_val) if pd.notna(roc_val) else 0.0
                            }

                structure_break = None
                if STRUCTURE_DETECTION_ENABLED and '4h' in data:
                    structure_break = detect_structure_break(data['4h'])
                    if structure_break:
                        logging.info(f"{symbol}: Structure: {structure_break['type']} ({structure_break['direction']})")

                wick_result = None
                if WICK_DETECTOR_AVAILABLE and wick_detector and '4h' in data:
                    try:
                        wick_result = wick_detector.detect_wick_reversal(
                            df=data['4h'],
                            symbol=symbol,
                            current_price=price,
                            htf_bias=btc_trend.lower() if btc_trend else None,
                            order_blocks=premium_zones[:3] if premium_zones else None,
                            stop_hunt_result=detect_stop_hunt(data['4h']) if STRUCTURE_BREAKER_AVAILABLE else None,
                            fear_greed=fear_greed_data,
                            structure_break=structure_break
                        )
                        if wick_result and wick_result.detected:
                            logging.info(
                                f"{symbol}: {get_emoji('candle')} WICK: {wick_result.wick_type.value} | "
                                f"Dir: {wick_result.direction} | Conf: {wick_result.confidence:.0f}%"
                            )
                    except Exception as e:
                        logging.error(f"{symbol}: Wick detection error: {e}")

                stop_hunt_result = None
                fake_breakout_result = None

                if STRUCTURE_BREAKER_AVAILABLE and '4h' in data:
                    try:
                        stop_hunt_result = detect_stop_hunt(data['4h'])
                        fake_breakout_result = detect_fake_breakout(data['4h'])

                        if stop_hunt_result:
                            logging.info(f"{symbol}: {get_emoji('lightning')} STOP HUNT: {stop_hunt_result['direction']}")
                        if fake_breakout_result:
                            logging.info(f"{symbol}: FAKE BREAKOUT: {fake_breakout_result['direction']}")
                    except Exception as e:
                        logging.debug(f"{symbol}: Structure pattern error: {e}")

                # ================================================================
                # v27.12.2: VOLATILITY PROFILE CALCULATION
                # ================================================================
                # v27.12.2: VOLATILITY PROFILE FOR CLAUDE CONTEXT
                # ================================================================
                vol_data = None
                try:
                    from bot.volatility_profile import calculate_volatility_profile
                    # Get BTC data for beta calculation
                    btc_df = data_cache.get('BTC/USDT', {}).get('1d')
                    symbol_df = data.get('1d')
                    if symbol_df is not None and len(symbol_df) > 20:
                        vol_data = calculate_volatility_profile(symbol, symbol_df, btc_df)
                        if vol_data and vol_data.get('regime') != 'MEDIUM':
                            logging.info(f"{symbol}: Vol regime={vol_data['regime']} beta={vol_data['beta']:.2f}")
                except ImportError:
                    pass  # Module not available, continue without vol data
                except Exception as e:
                    logging.debug(f"{symbol}: Volatility calc error: {e}")

                # ================================================================
                # v27.12.14: EARLY REVERSAL DETECTION
                # ================================================================
                reversal_signal = None
                try:
                    if EARLY_REVERSAL_AVAILABLE and '4h' in data:
                        reversal_signal = detect_early_reversal(data['4h'], timeframe='4h')
                        if reversal_signal and reversal_signal.confluence_count >= 2:
                            logging.info(f"{symbol}: 🔄 EARLY REVERSAL: {reversal_signal.direction} | Conf: {reversal_signal.confidence:.0f}% | {reversal_signal.confluence_count} signals")
                            print(f"[REVERSAL] {symbol}: {reversal_signal.summary}", flush=True)
                except Exception as e:
                    logging.debug(f"{symbol}: Early reversal detection error: {e}")

                # ================================================================
                # v27.12.1: BULLETPROOF CLAUDE ANALYSIS
                # ================================================================
                logging.info(f"{symbol}: Querying Claude...")
                print(f"[CLAUDE] {symbol}: Sending analysis request...", flush=True)

                # Wrap in try/except to catch ANY exception including CancelledError
                try:
                    claude_result = await query_claude_analysis(
                        premium_zones, symbol, price, trend, btc_trend,
                        oi_data=oi_data,
                        funding_data=funding_data,
                        divergence_data=divergence_data,
                        momentum_data=momentum_data,
                        volume_comparison=volume_comparison,
                        structure_break=structure_break,
                        fear_greed=fear_greed_data,
                        long_short_ratio=long_short_data,
                        regime=regime,
                        stop_hunt_result=stop_hunt_result,
                        fake_breakout_result=fake_breakout_result,
                        wick_result=wick_result,
                        vol_data=vol_data,  # v27.12.2: Pass volatility profile
                        reversal_signal=reversal_signal  # v27.12.14: Pass reversal signal
                    )
                except asyncio.CancelledError:
                    # v27.12.1: Catch CancelledError explicitly
                    logging.warning(f"{symbol}: Claude request cancelled (CancelledError)")
                    claude_result = {'no_trade': True, 'reason': 'Request cancelled'}
                except Exception as e:
                    # v27.12.1: Catch any other exception
                    logging.error(f"{symbol}: Claude call exception: {type(e).__name__}: {e}")
                    claude_result = {'no_trade': True, 'reason': str(e)}

                # v27.12.1: ENSURE claude_result is always a dict before calling .get()
                if claude_result is None:
                    logging.warning(f"{symbol}: Claude returned None")
                    claude_result = {'no_trade': True, 'reason': 'Null response'}
                elif not isinstance(claude_result, dict):
                    logging.error(f"{symbol}: Claude returned non-dict: {type(claude_result).__name__}")
                    claude_result = {'no_trade': True, 'reason': f'Invalid response type: {type(claude_result).__name__}'}

                # NOW it's safe to call .get()
                if 'trade' not in claude_result:
                    reason = claude_result.get('reason', 'No setup found')
                    logging.info(f"{symbol}: Claude declined - {reason}")
                    print(f"[CLAUDE] {symbol}: No trade - {reason[:50]}", flush=True)
                    analyzed += 1
                    continue

                grok = claude_result['trade']
                logging.info(f"{symbol}: Claude suggests {grok['direction']} @ {grok.get('entry_low', 0):.4f}")
                print(f"[CLAUDE] {symbol}: TRADE SUGGESTED - {grok['direction']} @ {grok.get('entry_low', 0):.4f}", flush=True)

                entry_low = grok.get('entry_low', 0)
                entry_high = grok.get('entry_high', 0)
                entry_mid = (entry_low + entry_high) / 2

                entry_dist_pct = abs(price - entry_mid) / price * 100
                if entry_dist_pct > 5.0:
                    logging.info(f"{symbol}: Entry too far ({entry_dist_pct:.1f}%)")
                    analyzed += 1
                    continue

                confidence = grok.get('confidence', 0)
                if confidence < CLAUDE_MIN_CONFIDENCE:
                    logging.info(f"{symbol}: Confidence too low ({confidence}%)")
                    analyzed += 1
                    continue

                sl_dist = abs(grok['sl'] - entry_mid)
                tp1_dist = abs(grok['tp1'] - entry_mid)
                rr_ratio = tp1_dist / sl_dist if sl_dist > 0 else 0

                if rr_ratio < 1.5:
                    logging.info(f"{symbol}: R:R too low ({rr_ratio:.2f})")
                    analyzed += 1
                    continue

                direction = grok['direction']
                is_counter_trend = False

                if btc_trend == 'Uptrend' and direction == 'Short':
                    is_counter_trend = True
                elif btc_trend == 'Downtrend' and direction == 'Long':
                    is_counter_trend = True

                structure_aligned = False
                structure_reason = ""
                if structure_break:
                    struct_signal = structure_break.get('signal', '')
                    if struct_signal.upper() == direction.upper():
                        structure_aligned = True
                        structure_reason = f"+{structure_break['type']}"

                psych_aligned = False
                if fear_greed_data:
                    fg_value = fear_greed_data.get('value', 50)
                    if direction == 'Long' and fg_value <= 25:
                        psych_aligned = True
                    elif direction == 'Short' and fg_value >= 75:
                        psych_aligned = True

                mom_signal, roc = safe_get_momentum_signal(momentum_data)
                momentum_aligned = (
                    (direction == 'Long' and mom_signal == 'bullish') or
                    (direction == 'Short' and mom_signal == 'bearish')
                )

                # v27.12.15: Enhanced MTF confluence with weighting
                timeframe_agreement = 0
                mtf_weighted_score = 0.0
                
                # Old simple count (for backward compatibility)
                for tf in ['4h', '1d']:
                    if tf in data:
                        tf_df = data[tf]
                        if 'ema200' in tf_df.columns and pd.notna(tf_df['ema200'].iloc[-1]):
                            if direction == 'Long' and tf_df['close'].iloc[-1] > tf_df['ema200'].iloc[-1]:
                                timeframe_agreement += 1
                            elif direction == 'Short' and tf_df['close'].iloc[-1] < tf_df['ema200'].iloc[-1]:
                                timeframe_agreement += 1
                
                # New weighted MTF calculation
                if DYNAMIC_TP_AVAILABLE:
                    try:
                        mtf_result = calculate_mtf_confluence_weighted(data)
                        if mtf_result and mtf_result.alignment_score > 0:
                            mtf_weighted_score = mtf_result.alignment_score
                            # Boost confluence if weighted MTF strongly agrees
                            if mtf_result.primary_direction == direction.upper() and mtf_weighted_score >= 0.6:
                                timeframe_agreement = max(timeframe_agreement, 2)  # Ensure at least 2 if strongly aligned
                                logging.info(f"{symbol}: MTF weighted score: {mtf_weighted_score:.2f} ({mtf_result.primary_direction})")
                    except Exception as e:
                        logging.debug(f"{symbol}: Weighted MTF error: {e}")

                has_ote = False
                if OTE_AVAILABLE and '4h' in data and premium_zones:
                    try:
                        in_ote, ote_data, _ = analyze_ote(data['4h'], price, premium_zones, direction)
                        has_ote = in_ote
                    except Exception as e:
                        logging.debug(f"{symbol}: OTE analysis error: {e}")
                        has_ote = False

                wick_detected = wick_result is not None and wick_result.detected

                # v27.12.14: Convert reversal signal to dict for confluence counting
                reversal_dict = None
                if reversal_signal and reversal_signal.direction.upper() == grok['direction'].upper():
                    reversal_dict = {
                        'divergences': [{'divergence_type': d.divergence_type, 'indicator': d.indicator} for d in reversal_signal.divergences],
                        'candlestick_patterns': [{'pattern_name': p.pattern_name} for p in reversal_signal.candlestick_patterns],
                        'chart_patterns': [{'pattern_name': p.pattern_name} for p in reversal_signal.chart_patterns]
                    }

                confluence_count, confluence_factors = count_confluence_factors(
                    grok, momentum_data, divergence_data, funding_data,
                    oi_data, volume_comparison, timeframe_agreement, has_ote,
                    structure_aligned, psych_aligned, wick_detected,
                    reversal_signal=reversal_dict,  # v27.12.14: Include reversal
                    orderbook_signal=orderbook_signal  # v27.12.14: Include orderbook
                )

                if confluence_count < MIN_CONFLUENCE_FACTORS:
                    logging.info(f"{symbol}: Insufficient confluence ({confluence_count}/{MIN_CONFLUENCE_FACTORS})")
                    analyzed += 1
                    continue

                if UNIFIED_EVALUATOR_AVAILABLE:
                    eval_result = evaluate_signal(
                        trade=grok,
                        current_price=price,
                        zone=premium_zones[0] if premium_zones else {},
                        momentum_data=momentum_data,
                        divergence_data=divergence_data,
                        funding_data=funding_data,
                        oi_data=oi_data,
                        volume_comparison=volume_comparison,
                        timeframe_agreement=timeframe_agreement,
                        has_ote=has_ote,
                        structure_break=structure_break,
                        htf_trend=btc_trend,
                        fear_greed=fear_greed_data,
                        long_short=long_short_data
                    )
                    quality_grade = eval_result['grade']
                    quality_score = eval_result['score']
                    size_mult = eval_result.get('size_multiplier', 1.0)
                else:
                    grade_result = grade_signal(
                        ob_score=int(grok.get('strength', 2) * 30),
                        confluence_count=confluence_count,
                        trend_aligned=not is_counter_trend,
                        rr_ratio=rr_ratio,
                        entry_distance_pct=entry_dist_pct,
                        is_counter_trend=is_counter_trend,
                        claude_confidence=confidence,
                        momentum_aligned=momentum_aligned,
                        structure_aligned=structure_aligned
                    )
                    quality_grade = grade_result['grade']
                    quality_score = grade_result['score']
                    size_mult = grade_result.get('size_mult', 1.0)

                if quality_grade not in EXECUTABLE_GRADES:
                    logging.info(f"{symbol}: Grade {quality_grade} not executable")
                    analyzed += 1
                    continue

                # ================================================================
                # v27.12.14: DYNAMIC TP CALCULATION
                # ================================================================
                final_tp1 = grok['tp1']
                final_tp2 = grok['tp2']
                dynamic_tp_info = None
                
                if DYNAMIC_TP_AVAILABLE and DYNAMIC_TP_ENABLED and '4h' in data:
                    try:
                        vol_regime = detect_volatility_regime(data['4h'])
                        if vol_regime:
                            dynamic_tp_result = calculate_dynamic_tp(
                                entry_price=entry_mid,
                                sl_price=grok['sl'],
                                direction=direction,
                                volatility_regime=vol_regime
                            )
                            if dynamic_tp_result:
                                final_tp1 = dynamic_tp_result.tp1
                                final_tp2 = dynamic_tp_result.tp2
                                dynamic_tp_info = {
                                    'regime': vol_regime.regime,
                                    'tp1_r': dynamic_tp_result.tp1_r,
                                    'tp2_r': dynamic_tp_result.tp2_r
                                }
                                logging.info(f"{symbol}: Dynamic TP applied - Regime: {vol_regime.regime}, TP1 R:{dynamic_tp_result.tp1_r:.1f}, TP2 R:{dynamic_tp_result.tp2_r:.1f}")
                    except Exception as e:
                        logging.debug(f"{symbol}: Dynamic TP calculation error: {e}")

                grok_opinion_result = {'opinion': 'neutral', 'reason': '', 'display': ''}
                if GROK_OPINION_ENABLED:
                    try:
                        grok_opinion_result = await get_grok_opinion(
                            grok, symbol, price, trend, btc_trend, timeout=10.0
                        )
                    except Exception as e:
                        logging.debug(f"{symbol}: Grok opinion error: {e}")

                open_trades[symbol] = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_low': entry_low,
                    'entry_high': entry_high,
                    'entry_price': entry_mid,
                    'sl': grok['sl'],
                    'tp1': final_tp1,  # v27.12.14: Use dynamic TP
                    'tp2': final_tp2,  # v27.12.14: Use dynamic TP
                    'original_tp1': grok['tp1'],  # Keep original for reference
                    'original_tp2': grok['tp2'],
                    'dynamic_tp_info': dynamic_tp_info,
                    'leverage': grok['leverage'],
                    'confidence': confidence,
                    'strength': grok.get('strength', 2.0),
                    'reason': grok.get('reason', ''),
                    'entry_time': now,
                    'last_check': now,
                    'active': False,
                    'tp1_exited': False,
                    'trailing_sl': None,
                    'timeframe': '4h',
                    'factors': confluence_factors,
                    'confluence_count': confluence_count,
                    'is_counter_trend': is_counter_trend,
                    'use_tp2': not is_counter_trend if COUNTER_TREND_TP1_ONLY else True,
                    'grade': quality_grade,
                    'grade_score': quality_score,
                    'size_mult': size_mult,
                    'structure_aligned': structure_aligned,
                    'psych_aligned': psych_aligned,
                    'wick_signal': wick_result.wick_type.value if wick_detected else None,
                    'grok_opinion': grok_opinion_result['opinion']
                }

                await save_trades_async(open_trades)
                last_signal_time[symbol] = now

                # v27.12.11: Execute auto-trade if enabled
                if blofin_trader and AUTO_TRADE_ENABLED:
                    await process_signal_auto_trade(open_trades[symbol], symbol)

                # v27.12.14: Use final (dynamic) TP values for R:R calculation
                rr1 = abs(final_tp1 - entry_mid) / sl_dist if sl_dist > 0 else 0
                rr2 = abs(final_tp2 - entry_mid) / sl_dist if sl_dist > 0 else 0

                ev_r = calculate_expected_value({
                    'direction': direction,
                    'entry_low': entry_low,
                    'entry_high': entry_high,
                    'sl': grok['sl'],
                    'tp1': final_tp1,
                    'tp2': final_tp2
                }, HISTORICAL_DATA)

                grade_emoji = {'A': get_emoji('star'), 'B': get_emoji('check'), 'C': get_emoji('lightning'),
                               'D': get_emoji('warning'), 'F': get_emoji('cross')}.get(quality_grade, '')
                counter_tag = f" {get_emoji('lightning')}CT" if is_counter_trend else ""
                tp_note = " (TP1 only)" if is_counter_trend and COUNTER_TREND_TP1_ONLY else ""

                symbol_short = symbol.replace('/USDT', '')
                msg = (
                    f"{get_emoji('alarm')} **{symbol_short} LIVE SIGNAL** {grade_emoji} Grade {quality_grade}{counter_tag}\n\n"
                    f"*Score:* {quality_score}/100 | *EV:* {ev_r:.2f}R | *MTF:* {timeframe_agreement}/2 {get_emoji('check')}\n"
                )

                if fear_greed_data:
                    fg_value = fear_greed_data.get('value', 50)
                    fg_emoji = get_emoji('chart_down') if fg_value <= 25 else get_emoji('chart') if fg_value >= 75 else get_emoji('graph')
                    msg += f"*Psychology:* {fg_emoji} F&G={fg_value}\n"

                if divergence_data and divergence_data.get('significant'):
                    msg += f"*Divergence:* {divergence_data['signal'].upper()} {divergence_data['divergence_pct']:+.2f}%\n"
                if momentum_aligned:
                    msg += f"*Momentum:* {mom_signal.upper()} {get_emoji('check')}\n"
                if structure_aligned:
                    msg += f"*Structure:* {structure_reason}\n"

                if wick_detected:
                    wick_type_display = wick_result.wick_type.value.replace('_', ' ').title()
                    msg += f"*Wick Signal:* {get_emoji('candle')} {wick_type_display} ({wick_result.confidence:.0f}%)\n"

                # v27.12.14: Show dynamic TP info if applied
                if dynamic_tp_info:
                    msg += f"*Dynamic TP:* {dynamic_tp_info['regime']} Vol (R:{dynamic_tp_info['tp1_r']:.1f}/{dynamic_tp_info['tp2_r']:.1f})\n"

                msg += f"\n**{direction}** | *Conf:* {confidence}%{tp_note}\n"
                msg += f"*Entry:* {format_price(entry_low)} - {format_price(entry_high)}\n"
                msg += f"*SL:* {format_price(grok['sl'])} | *TP1:* {format_price(final_tp1)}"

                if not is_counter_trend or not COUNTER_TREND_TP1_ONLY:
                    msg += f" | *TP2:* {format_price(final_tp2)}"

                msg += f"\n*Leverage:* {grok['leverage']}x | *R:R* 1:{rr1:.1f}"
                if not is_counter_trend or not COUNTER_TREND_TP1_ONLY:
                    msg += f"/1:{rr2:.1f}"

                msg += f"\n\n**Factors:** {', '.join(confluence_factors[:6])}\n"
                msg += f"**Reason:** {grok['reason']}"

                if grok_opinion_result['display']:
                    msg += f"\n\n{grok_opinion_result['display']}"

                await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
                signals_sent += 1
                analyzed += 1

                logging.info(f"{symbol}: Signal sent - {direction} Grade {quality_grade}")

            except Exception as e:
                logging.error(f"{symbol}: Error - {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue

        cycle_duration = time.time() - cycle_start
        logging.info("=" * 70)
        logging.info(f"=== CYCLE COMPLETE: {cycle_duration:.1f}s | Analyzed: {analyzed} | Skipped: {skipped} | Signals: {signals_sent} ===")
        
        # v27.12.12: Comprehensive summary for Render visibility
        print(f"\n{'='*70}", flush=True)
        print(f"[SIGNAL] ========== CYCLE COMPLETE ==========", flush=True)
        print(f"[SIGNAL] Duration: {cycle_duration:.1f}s", flush=True)
        print(f"[SIGNAL] Analyzed: {analyzed} symbols", flush=True)
        print(f"[SIGNAL] Skipped: {skipped} symbols", flush=True)
        print(f"[SIGNAL] Signals Sent: {signals_sent}", flush=True)
        print(f"[SIGNAL] Next cycle in: {CHECK_INTERVAL//60} minutes", flush=True)
        print(f"{'='*70}\n", flush=True)
        flush_logs()

    except Exception as e:
        logging.error(f"Signal callback error: {e}")
        print(f"[ERROR] Signal callback: {e}", flush=True)
        import traceback
        logging.error(traceback.format_exc())


# ============================================================================
# TRACK CALLBACK - Position Management - v27.12.12 with logging
# ============================================================================
async def track_callback(context):
    """Track and manage open positions."""
    global open_trades, protected_trades, stats

    # v27.12.12: Always log track callback for visibility
    total_positions = len(open_trades) + len(protected_trades)
    logging.debug(f"Track callback: {total_positions} positions")
    
    if not open_trades and not protected_trades:
        # v27.12.12: Periodic heartbeat logging even when no trades
        return

    try:
        prices = await fetch_ticker_batch()
        now = datetime.now(timezone.utc)
        
        invalid_trades = []
        for trade_key, trade in list(open_trades.items()):
            if trade.get('sl') is None or trade.get('tp1') is None:
                logging.warning(f"Trade {trade_key} has missing SL/TP1, removing")
                invalid_trades.append(trade_key)
        for key in invalid_trades:
            del open_trades[key]
        
        invalid_protected = []
        for trade_key, trade in list(protected_trades.items()):
            if trade.get('sl') is None or trade.get('tp1') is None:
                logging.warning(f"Protected trade {trade_key} has missing SL/TP1, removing")
                invalid_protected.append(trade_key)
        for key in invalid_protected:
            del protected_trades[key]
        
        if not open_trades and not protected_trades:
            return
        
        symbols_to_track = list(open_trades.keys()) + list(protected_trades.keys())
        atr_values = await get_atr_values(symbols_to_track, data_cache)
        
        current_capital = stats.get('capital', SIMULATED_CAPITAL)
        
        to_delete_open = []
        to_delete_protected = []
        updated_keys_open = []
        updated_keys_protected = []

        if open_trades:
            await process_trade(
                trades=open_trades,
                to_delete=to_delete_open,
                now=now,
                current_capital=current_capital,
                prices=prices,
                updated_keys=updated_keys_open,
                is_protected=False,
                stats_lock=stats_lock,
                stats=stats,
                atr_values=atr_values
            )
        
        if protected_trades:
            await process_trade(
                trades=protected_trades,
                to_delete=to_delete_protected,
                now=now,
                current_capital=current_capital,
                prices=prices,
                updated_keys=updated_keys_protected,
                is_protected=True,
                stats_lock=stats_lock,
                stats=stats,
                atr_values=atr_values
            )
        
        for key in to_delete_open:
            if key in open_trades:
                del open_trades[key]
                last_trade_result[key] = 'CLOSED'
        
        for key in to_delete_protected:
            if key in protected_trades:
                del protected_trades[key]
                last_trade_result[key] = 'CLOSED'

        await save_trades_async(open_trades)
        await save_protected_async(protected_trades)
        await save_stats_async(stats)

    except Exception as e:
        logging.error(f"Track callback error: {e}")
        import traceback
        logging.error(traceback.format_exc())


# ============================================================================
# SCHEDULED CALLBACKS - v27.12.12 with heartbeat logging
# ============================================================================

# v27.12.12: Heartbeat counter for periodic status
_heartbeat_counter = 0

async def heartbeat_callback(context):
    """
    v27.12.12: Periodic heartbeat to show bot is alive.
    Runs every 5 minutes to provide Render visibility.
    """
    global _heartbeat_counter
    _heartbeat_counter += 1
    
    now = datetime.now(timezone.utc)
    btc_price = prices_global.get('BTC/USDT', 0)
    active_trades = len(open_trades) + len(protected_trades)
    
    # Log heartbeat every 5 minutes
    print(f"[HEARTBEAT #{_heartbeat_counter}] {now.strftime('%H:%M:%S')} UTC | BTC: ${btc_price:,.0f} | Trades: {active_trades} | Trend: {btc_trend_global}", flush=True)
    logging.debug(f"Heartbeat #{_heartbeat_counter}: BTC=${btc_price:,.0f}, Trades={active_trades}")
    flush_logs()


async def price_update_callback(context):
    """Update global prices."""
    global prices_global
    try:
        prices = await fetch_ticker_batch()
        updated = 0
        for symbol, price in prices.items():
            if price:
                prices_global[symbol] = price
                updated += 1
        logging.debug(f"Prices updated: {updated}/{len(SYMBOLS)}")
    except Exception as e:
        logging.debug(f"Price update error: {e}")


async def btc_trend_update(context):
    """Update BTC trend indicator."""
    global btc_trend_global
    try:
        df = await fetch_ohlcv('BTC/USDT', '1d', 250)
        if len(df) >= 200:
            df = add_institutional_indicators(df)
            df = df.dropna(subset=['ema200'])
            if len(df) >= 100:
                last = df.iloc[-1]
                if last['close'] > last['ema200'] * 1.02:
                    btc_trend_global = "Uptrend"
                elif last['close'] < last['ema200'] * 0.98:
                    btc_trend_global = "Downtrend"
                else:
                    btc_trend_global = "Sideways"
                logging.info(f"BTC trend updated: {btc_trend_global}")
    except Exception as e:
        logging.error(f"BTC trend update error: {e}")


async def roadmap_scheduled_callback(context):
    """Check if it's time to generate roadmaps and monitor proximity."""
    try:
        if is_roadmap_generation_time():
            print(f"[ROADMAP] Generating roadmaps at {datetime.now(timezone.utc).strftime('%H:%M')} UTC", flush=True)
            await roadmap_generation_callback(data_cache, btc_trend_global)

        from bot.roadmap import monitor_roadmap_proximity
        await monitor_roadmap_proximity()

    except Exception as e:
        logging.error(f"Roadmap callback error: {e}")
        print(f"[ERROR] Roadmap callback: {e}", flush=True)


async def daily_callback(context):
    """Daily recap at 00:05 UTC."""
    try:
        if FIXED_RECAP_AVAILABLE:
            await daily_callback_fixed(context, query_grok_daily_recap)
        else:
            now = datetime.now(timezone.utc)
            msg = f"{get_emoji('graph')} **Daily Summary** - {now.strftime('%B %d, %Y')}\n\n"

            prices = await fetch_ticker_batch()
            for sym in SYMBOLS[:5]:
                price = prices.get(sym)
                if price:
                    msg += f"{get_emoji('bullet')} {sym.replace('/USDT', '')}: {format_price(price)}\n"

            await send_throttled(CHAT_ID, msg, parse_mode='Markdown')

    except Exception as e:
        logging.error(f"Daily callback error: {e}")


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def dummy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Dummy handler for unhandled messages."""
    pass


async def force_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Force immediate signal check."""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text(f"{get_emoji('cross')} Unauthorized")
        return

    await update.message.reply_text(f"{get_emoji('rocket')} Forcing signal check...")
    await signal_callback(context)


async def market_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show market overview."""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text(f"{get_emoji('cross')} Unauthorized")
        return

    try:
        session_info = get_session_info()
        prices = await fetch_ticker_batch()

        df_btc = await fetch_ohlcv('BTC/USDT', '1d', 50)
        df_btc = add_institutional_indicators(df_btc)
        btc_vol = get_current_volatility(df_btc) if len(df_btc) > 0 else 2.0
        regime = detect_market_regime(df_btc) if len(df_btc) > 0 else 'unknown'

        active = len([t for trades in [open_trades, protected_trades]
                      for t in trades.values() if t.get('active')])
        pending = len([t for t in open_trades.values() if not t.get('active')])

        roadmap_count = sum(len(z) for z in roadmap_zones.values())

        btc_price = prices.get('BTC/USDT', 0)
        eth_price = prices.get('ETH/USDT', 0)

        msg = (
            f"{get_emoji('graph')} **Market Overview**\n\n"
            f"**BTC:** {format_price(btc_price)} ({btc_trend_global})\n"
            f"**ETH:** {format_price(eth_price)}\n"
            f"**Regime:** {regime.upper()}\n"
            f"**Volatility:** {btc_vol:.2f}%\n"
            f"**Session:** {session_info['session'].upper()}\n\n"
            f"**Positions:** {active} active | {pending} pending\n"
            f"**Roadmaps:** {roadmap_count} zones"
        )

        if WICK_DETECTOR_AVAILABLE:
            msg += f"\n**Wick Detection:** {get_emoji('check')} Active"

        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')

    except Exception as e:
        logging.error(f"/market error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}")


async def factors_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show factor performance."""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text(f"{get_emoji('cross')} Unauthorized")
        return

    try:
        factor_stats = factor_tracker.get_stats()

        if not factor_stats:
            await update.message.reply_text("No factor data yet")
            return

        msg = f"{get_emoji('graph')} **Factor Performance**\n\n"

        sorted_factors = sorted(
            factor_stats.items(),
            key=lambda x: x[1].get('win_rate', 0),
            reverse=True
        )

        for factor, fdata in sorted_factors[:10]:
            wins = fdata.get('wins', 0)
            total = fdata.get('total', 0)
            win_rate = fdata.get('win_rate', 0)
            emoji = get_emoji('green') if win_rate >= 60 else get_emoji('yellow') if win_rate >= 50 else get_emoji('red')
            msg += f"{emoji} **{factor}**: {win_rate:.0f}% ({wins}/{total})\n"

        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')

    except Exception as e:
        logging.error(f"/factors error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}")


async def genroadmap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually generate roadmaps."""
    # v27.12.13: Fix NoneType error - handle both message and callback_query
    if update.effective_user is None:
        return
    
    if str(update.effective_user.id) != CHAT_ID:
        if update.message:
            await update.message.reply_text(f"{get_emoji('cross')} Unauthorized")
        return

    # Send initial message - handle both message and callback contexts
    if update.message:
        await update.message.reply_text(f"{get_emoji('target')} Generating roadmaps...")
    else:
        # Called from callback or scheduled job - log instead
        print(f"[ROADMAP] Starting roadmap generation...", flush=True)
    
    await roadmap_generation_callback(data_cache, btc_trend_global)
    
    # Send completion message
    if update.message:
        await update.message.reply_text(f"{get_emoji('check')} Roadmaps generated")
    else:
        print(f"[ROADMAP] Generation complete", flush=True)


# ============================================================================
# WELCOME MESSAGE
# ============================================================================

async def send_welcome_once():
    """Send welcome message only once per deploy."""
    if os.path.exists(FLAG_FILE):
        return

    try:
        evaluator_status = f"{get_emoji('check')} Unified" if UNIFIED_EVALUATOR_AVAILABLE else f"{get_emoji('warning')} Legacy"
        recap_status = f"{get_emoji('check')} Fixed" if FIXED_RECAP_AVAILABLE else f"{get_emoji('warning')} Legacy"
        wick_status = f"{get_emoji('check')}" if WICK_DETECTOR_AVAILABLE else f"{get_emoji('cross')}"
        
        # v27.12.11: Blofin status
        if is_blofin_configured():
            blofin_status = f"{get_emoji('check')} {'ON' if AUTO_TRADE_ENABLED else 'OFF'}"
        else:
            blofin_status = f"{get_emoji('cross')} Not configured"

        mode_emoji = get_emoji('paper') if PAPER_TRADING else get_emoji('money')

        msg = (
            f"{get_emoji('rocket')} **Grok Elite Bot v{BOT_VERSION}** Started!\n\n"
            f"**Mode:** {mode_emoji} {'Paper' if PAPER_TRADING else 'Live'}\n"
            f"**AI:** Claude (Primary) + Grok (Opinion)\n"
            f"**Symbols:** {len(SYMBOLS)}\n"
            f"**Capital:** ${SIMULATED_CAPITAL:,.0f}\n\n"
            f"**v27.12.15 Features:**\n"
            f"{get_emoji('bullet')} Early Reversal Detection (RSI/MACD Div)\n"
            f"{get_emoji('bullet')} Dynamic TPs (volatility-adjusted)\n"
            f"{get_emoji('bullet')} Liquidity Map Analysis\n"
            f"{get_emoji('bullet')} Orderbook Imbalance Detection\n"
            f"{get_emoji('bullet')} Candlestick & Chart Patterns\n\n"
            f"**Signal Evaluator:** {evaluator_status}\n"
            f"**Daily Recap:** {recap_status}\n"
            f"**Wick Detection:** {wick_status}\n"
            f"**Blofin Auto-Trade:** {blofin_status}\n"
            f"**Dynamic TP:** {get_emoji('check') if DYNAMIC_TP_AVAILABLE else get_emoji('cross')}\n"
            f"**Early Reversal:** {get_emoji('check') if EARLY_REVERSAL_AVAILABLE else get_emoji('cross')}\n"
            f"**Liquidity Map:** {get_emoji('check') if LIQUIDITY_MAP_AVAILABLE else get_emoji('cross')}\n\n"
            f"Type /commands for available commands"
        )

        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')

        os.makedirs(os.path.dirname(FLAG_FILE), exist_ok=True)
        with open(FLAG_FILE, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())

    except Exception as e:
        logging.error(f"Failed to send welcome: {e}")


# ============================================================================
# INITIALIZATION
# ============================================================================

async def deferred_init(context: ContextTypes.DEFAULT_TYPE):
    """Run slow initialization tasks AFTER webhook is up."""
    global btc_trend_global, blofin_trader

    logging.info("Running deferred initialization...")
    print(f"\n[INIT] ========== DEFERRED INIT STARTING ==========", flush=True)

    try:
        print(f"[INIT] Fetching BTC data for trend calculation...", flush=True)
        df_btc = await fetch_ohlcv('BTC/USDT', '1d', limit=500)
        if len(df_btc) >= 200:
            df_btc = add_institutional_indicators(df_btc)
            df_btc = df_btc.dropna(subset=['ema200'])
            if len(df_btc) >= 100:
                last_row = df_btc.iloc[-1]
                if last_row['close'] > last_row['ema200']:
                    btc_trend_global = "Uptrend"
                elif last_row['close'] < last_row['ema200']:
                    btc_trend_global = "Downtrend"
                else:
                    btc_trend_global = "Sideways"
                logging.info(f"Initial BTC trend: {btc_trend_global}")
    except Exception as e:
        logging.error(f"BTC trend init error: {e}")
        btc_trend_global = "Sideways"

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.bybit.com/v5/market/tickers?category=linear")
            if r.status_code == 200:
                logging.info("Bybit API: OK")

        claude_ok = await claude_health_check()
        logging.info(f"Claude API: {'OK' if claude_ok else 'FAILED'}")

    except Exception as e:
        logging.error(f"Health check error: {e}")

    # v27.12.11: Initialize Blofin Auto-Trading
    try:
        blofin_ok = await initialize_blofin()
        logging.info(f"Blofin Auto-Trade: {'OK' if blofin_ok else 'DISABLED'}")
    except Exception as e:
        logging.error(f"Blofin init error: {e}")

    await send_welcome_once()
    logging.info("Deferred initialization complete")
    
    # v27.12.12: Comprehensive init complete message
    print(f"\n{'='*70}", flush=True)
    print(f"[INIT] ========== BOT FULLY INITIALIZED ==========", flush=True)
    print(f"[INIT] BTC Trend: {btc_trend_global}", flush=True)
    print(f"[INIT] Signal Interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL//60}min)", flush=True)
    print(f"[INIT] Next signal cycle in approximately 30 seconds", flush=True)
    print(f"[INIT] All systems operational - monitoring started", flush=True)
    print(f"{'='*70}\n", flush=True)
    flush_logs()


async def post_init(application: Application) -> None:
    """Initialize bot after application setup."""
    global background_task

    logging.info("Starting post_init...")

    initialize_roadmaps()

    await application.bot.delete_webhook(drop_pending_updates=True)

    if not RENDER_EXTERNAL_HOSTNAME:
        logging.error("RENDER_EXTERNAL_HOSTNAME not set!")
        return

    webhook_url = f"https://{RENDER_EXTERNAL_HOSTNAME}/webhook"
    await application.bot.set_webhook(url=webhook_url)
    logging.info(f"Webhook set: {webhook_url}")

    background_task = asyncio.create_task(price_background_task())

    application.job_queue.run_once(deferred_init, when=5, name='deferred_init')

    logging.info("Post_init complete")


async def shutdown_handler(application: Application) -> None:
    """Properly shutdown all async tasks and connections."""
    global background_task, blofin_trader

    logging.info(f"{get_emoji('wave')} Starting graceful shutdown...")

    # v27.12.11: Close Blofin connection
    if blofin_trader:
        logging.info("  Closing Blofin connection...")
        try:
            await blofin_trader.close()
            logging.info(f"  {get_emoji('check')} Blofin closed")
        except Exception as e:
            logging.warning(f"  {get_emoji('warning')} Blofin close error: {e}")

    logging.info("  Closing exchange connections...")
    try:
        await asyncio.wait_for(close_exchanges(), timeout=10.0)
        logging.info(f"  {get_emoji('check')} Exchanges closed")
    except asyncio.TimeoutError:
        logging.warning(f"  {get_emoji('warning')} Exchange close timed out")
    except Exception as e:
        logging.warning(f"  {get_emoji('warning')} Exchange close error: {e}")

    if background_task and not background_task.done():
        logging.info("  Cancelling background task...")
        background_task.cancel()
        try:
            await asyncio.wait_for(background_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logging.info(f"  {get_emoji('check')} Background task cancelled")

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logging.info(f"  Cancelling {len(tasks)} remaining tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.sleep(0.5)

    logging.info(f"{get_emoji('check')} Shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    global stats

    # v27.12.12: Force unbuffered output for Render - CRITICAL
    import sys
    import os
    
    # Set environment variable for Python unbuffered mode
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Reconfigure stdout/stderr for line buffering
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Initialize logging FIRST
    setup_logging()
    
    # v27.12.12: Comprehensive startup banner
    print(f"\n{'='*70}", flush=True)
    print(f"[STARTUP] Grok Elite Bot v{BOT_VERSION} INITIALIZING", flush=True)
    print(f"[STARTUP] Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
    print(f"{'='*70}", flush=True)

    try:
        validate_config()
    except ValueError as e:
        logging.error(str(e))
        print(f"[ERROR] Config validation failed: {e}", flush=True)
        sys.exit(1)

    logging.info(f"{get_emoji('rocket')} Starting Grok Elite Bot v{BOT_VERSION}")
    print(f"[STARTUP] Configuration validated successfully", flush=True)
    print(f"[STARTUP] Symbols: {len(SYMBOLS)} | Check Interval: {CHECK_INTERVAL//60}min", flush=True)

    try:
        from bot import print_version_banner
        print(print_version_banner())
    except ImportError:
        print(f"Grok Elite Bot v{BOT_VERSION}")

    # v27.12.12: Log all module availability
    logging.info(f"Unified SignalEvaluator: {'ENABLED' if UNIFIED_EVALUATOR_AVAILABLE else 'DISABLED'}")
    logging.info(f"Fixed Daily Recap: {'ENABLED' if FIXED_RECAP_AVAILABLE else 'DISABLED'}")
    logging.info(f"Wick Detection: {'ENABLED' if WICK_DETECTOR_AVAILABLE else 'DISABLED'}")
    logging.info(f"Blofin Module: {'AVAILABLE' if BLOFIN_AVAILABLE else 'NOT AVAILABLE'}")
    logging.info(f"Auto-Trade Enabled: {AUTO_TRADE_ENABLED}")
    
    print(f"[STARTUP] Modules loaded:", flush=True)
    print(f"[STARTUP]   - SignalEvaluator: {'OK' if UNIFIED_EVALUATOR_AVAILABLE else 'LEGACY'}", flush=True)
    print(f"[STARTUP]   - Daily Recap: {'OK' if FIXED_RECAP_AVAILABLE else 'LEGACY'}", flush=True)
    print(f"[STARTUP]   - Wick Detection: {'OK' if WICK_DETECTOR_AVAILABLE else 'OFF'}", flush=True)
    print(f"[STARTUP]   - Blofin Trading: {'OK' if BLOFIN_AVAILABLE else 'N/A'}", flush=True)
    print(f"[STARTUP]   - Auto-Trade: {'ON' if AUTO_TRADE_ENABLED else 'OFF'}", flush=True)
    flush_logs()

    stats = load_stats()

    application = Application.builder() \
        .token(TELEGRAM_TOKEN) \
        .post_init(post_init) \
        .post_shutdown(shutdown_handler) \
        .build()

    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("health", health_cmd))
    application.add_handler(CommandHandler("recap", recap_cmd))
    application.add_handler(CommandHandler("backtest", backtest_cmd))
    application.add_handler(CommandHandler("backtest_all", backtest_all_cmd))
    application.add_handler(CommandHandler("validate", validate_cmd))
    application.add_handler(CommandHandler("dashboard", dashboard_cmd))
    application.add_handler(CommandHandler("force", force_cmd))
    application.add_handler(CommandHandler("market", market_cmd))
    application.add_handler(CommandHandler("factors", factors_cmd))
    application.add_handler(CommandHandler("roadmap", roadmap_cmd))
    application.add_handler(CommandHandler("zones", zones_cmd))
    application.add_handler(CommandHandler("structural", structural_cmd))
    application.add_handler(CommandHandler("genroadmap", genroadmap_cmd))
    application.add_handler(CommandHandler("commands", commands_cmd))
    
    # v27.12.11: Blofin auto-trading commands
    application.add_handler(CommandHandler("blofin", blofin_status_cmd))
    application.add_handler(CommandHandler("blofin_toggle", blofin_toggle_cmd))
    application.add_handler(CommandHandler("blofin_close", blofin_close_cmd))
    
    # v27.12.13: Advanced analytics commands
    application.add_handler(CommandHandler("factor_analysis", factor_analysis_cmd))
    application.add_handler(CommandHandler("system_health", system_health_cmd))
    
    # v27.12.15: Liquidity and orderbook commands
    application.add_handler(CommandHandler("liquidity", liquidity_cmd))
    application.add_handler(CommandHandler("orderbook", orderbook_cmd))

    application.add_handler(MessageHandler(filters.ALL, dummy_handler))

    application.job_queue.run_repeating(
        signal_callback,
        interval=CHECK_INTERVAL,
        first=30,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 60}
    )

    application.job_queue.run_repeating(
        track_callback,
        interval=TRACK_INTERVAL,
        first=15,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )

    application.job_queue.run_repeating(
        price_update_callback,
        interval=60,
        first=10,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )

    application.job_queue.run_repeating(
        btc_trend_update,
        interval=300,
        first=60,
        job_kwargs={'max_instances': 1}
    )

    application.job_queue.run_repeating(
        roadmap_scheduled_callback,
        interval=300,
        first=120,
        job_kwargs={'max_instances': 1}
    )

    # v27.12.12: Heartbeat job for Render visibility
    application.job_queue.run_repeating(
        heartbeat_callback,
        interval=300,  # Every 5 minutes
        first=180,     # First after 3 minutes
        job_kwargs={'max_instances': 1}
    )

    now = datetime.now(timezone.utc)
    target = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if now.time() > target.time():
        target += timedelta(days=1)
    first_delay = int((target - now).total_seconds())

    application.job_queue.run_repeating(
        daily_callback,
        interval=86400,
        first=first_delay
    )

    logging.info(f"{get_emoji('check')} Jobs scheduled")
    print(f"[STARTUP] Jobs scheduled:", flush=True)
    print(f"[STARTUP]   - Signal check: every {CHECK_INTERVAL//60} minutes (first in 30s)", flush=True)
    print(f"[STARTUP]   - Position tracking: every {TRACK_INTERVAL}s", flush=True)
    print(f"[STARTUP]   - Price updates: every 60s", flush=True)
    print(f"[STARTUP]   - BTC trend: every 5min", flush=True)
    print(f"[STARTUP]   - Roadmap check: every 5min", flush=True)
    print(f"[STARTUP]   - Heartbeat: every 5min (Render visibility)", flush=True)
    print(f"[STARTUP]   - Daily recap: at 00:05 UTC", flush=True)
    flush_logs()

    webhook_path = "/webhook"
    webhook_url = f"https://{RENDER_EXTERNAL_HOSTNAME}{webhook_path}"
    
    print(f"\n{'='*70}", flush=True)
    print(f"[STARTUP] WEBHOOK STARTING", flush=True)
    print(f"[STARTUP] URL: {webhook_url}", flush=True)
    print(f"[STARTUP] Port: {PORT}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"[STARTUP] Bot is now LIVE and listening for updates!", flush=True)
    print(f"[STARTUP] First signal cycle will run in 30 seconds...", flush=True)
    print(f"{'='*70}\n", flush=True)
    flush_logs()

    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=webhook_path,
        webhook_url=webhook_url
    )


if __name__ == "__main__":
    main()
