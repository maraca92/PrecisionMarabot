# config.py - Grok Elite Signal Bot v27.12.13 - Configuration
# -*- coding: utf-8 -*-
"""
All constants, settings, and environment variables.

v27.12.13 CRITICAL FIX - BLOFIN BROKER ID:
- FIXED: "Unmatched brokerId" error (152013) that blocked Blofin trades
- NEW: BLOFIN_BROKER_ID environment variable (leave empty for Transaction API Keys)
- FIXED: genroadmap_cmd NoneType error when update.message is None
- NEW: Session-based trading multipliers (London/NY overlap = 1.2x)
- NEW: Correlation-based position sizing (reduces size for correlated assets)
- NEW: Drawdown protection (auto-pause at 8% drawdown)

v27.12.12 UPDATES - RENDER LOGGING FIX:
- All logging now properly outputs to stdout
- Enhanced debug output for Render visibility

v27.12.11 UPDATES:
- NEW: Blofin Auto-Trading Integration
- NEW: BLOFIN_API_KEY, BLOFIN_SECRET_KEY, BLOFIN_PASSPHRASE environment variables
- NEW: AUTO_TRADE_ENABLED, AUTO_TRADE_RISK_PCT, AUTO_TRADE_LEVERAGE settings
- NEW: is_blofin_configured() and get_blofin_config_summary() functions
- All v27.12.10 settings preserved

v27.12.10 UPDATES:
- NEW: ROADMAP_MAX_DISTANCE_PCT = 7.0 (only zones within 7% of current price)
- VERIFIED: RELAXED_MAX_ZONES_TREND = 5 (max 5 trend roadmaps)
- VERIFIED: RELAXED_MAX_ZONES_STRUCTURAL = 2 (max 2 structural bounces)
- VERIFIED: MANIPULATION_DETECTION_ENABLED = True
- Updated get_config_summary() to show new settings

v27.12.9 FIXES:
- FIXED: STRUCTURAL_EXPECTED_WIN_RATE changed from 0.55 to 65.0 (percentage)
- FIXED: STRUCTURAL_EXPECTED_AVG_BOUNCE changed from 2.0 to 3.5 (percentage)
- telegram_commands.py uses these as percentages with :.1f% formatting

v27.12.0 UPDATES:
- FIXED: Added missing variable aliases (MAX_CONCURRENT_TRADES, MAX_DRAWDOWN_PCT, RISK_PER_TRADE_PCT)
- FIXED: Added missing OB_MIN_QUALITY_SCORE, CLAUDE_MIN_STRENGTH
- FIXED: GROK_MODELS updated from deprecated "grok-beta" to "grok-3", "grok-3-mini"
- All previous v27.10.1/v27.11.0 fixes included
"""
import os
from typing import List, Dict

# ============================================================================
# VERSION
# ============================================================================
BOT_VERSION = "27.12.15"

# ============================================================================
# TRADING PAIRS & TIMEFRAMES
# ============================================================================
SYMBOLS: List[str] = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'BNB/USDT',
    'XRP/USDT',
    'ADA/USDT',
    'AVAX/USDT'
]

TIMEFRAMES: List[str] = ['1h', '4h', '1d', '1w']
SIGNAL_TIMEFRAMES: List[str] = ['4h', '1d']

# ============================================================================
# API CREDENTIALS
# ============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
XAI_API_KEY = os.getenv("XAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.getenv("PORT", 10000))

# ============================================================================
# BLOFIN AUTO-TRADING CONFIGURATION (v27.12.13)
# ============================================================================
BLOFIN_API_KEY = os.getenv("BLOFIN_API_KEY")
BLOFIN_SECRET_KEY = os.getenv("BLOFIN_SECRET_KEY")
BLOFIN_PASSPHRASE = os.getenv("BLOFIN_PASSPHRASE")

# v27.12.13: CRITICAL - Broker ID configuration
# For Transaction API Keys: Leave empty (BLOFIN_BROKER_ID="")
# For Broker API Keys: Set to your assigned broker ID
# Error 152013 means your brokerId doesn't match your API key type
BLOFIN_BROKER_ID = os.getenv("BLOFIN_BROKER_ID", "")

# Trading Mode
BLOFIN_DEMO_MODE = os.getenv("BLOFIN_DEMO_MODE", "false").lower() == "true"

# Auto-Trading Settings
AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
AUTO_TRADE_RISK_PCT = float(os.getenv("AUTO_TRADE_RISK_PCT", "0.015"))  # 1.5% per trade
AUTO_TRADE_MAX_LEVERAGE = int(os.getenv("AUTO_TRADE_MAX_LEVERAGE", "5"))
AUTO_TRADE_DEFAULT_LEVERAGE = int(os.getenv("AUTO_TRADE_DEFAULT_LEVERAGE", "3"))
AUTO_TRADE_MARGIN_MODE = os.getenv("AUTO_TRADE_MARGIN_MODE", "isolated")
AUTO_TRADE_POSITION_MODE = os.getenv("AUTO_TRADE_POSITION_MODE", "net_mode")
AUTO_TRADE_MIN_GRADE = os.getenv("AUTO_TRADE_MIN_GRADE", "B")

# v27.12.13: Session-based trading multipliers
SESSION_TRADING_ENABLED = os.getenv("SESSION_TRADING_ENABLED", "true").lower() == "true"

# v27.12.13: Correlation-based position sizing
CORRELATION_SIZING_ENABLED = os.getenv("CORRELATION_SIZING_ENABLED", "true").lower() == "true"

# v27.12.13: Drawdown protection thresholds
DRAWDOWN_WARNING_PCT = float(os.getenv("DRAWDOWN_WARNING_PCT", "0.03"))  # 3%
DRAWDOWN_CAUTION_PCT = float(os.getenv("DRAWDOWN_CAUTION_PCT", "0.05"))  # 5%
DRAWDOWN_STOP_PCT = float(os.getenv("DRAWDOWN_STOP_PCT", "0.08"))  # 8%

# Blofin API Endpoints
BLOFIN_BASE_URL = "https://demo-trading-openapi.blofin.com" if BLOFIN_DEMO_MODE else "https://openapi.blofin.com"
BLOFIN_WS_PUBLIC = f"wss://{'demo-trading-' if BLOFIN_DEMO_MODE else ''}openapi.blofin.com/ws/public"
BLOFIN_WS_PRIVATE = f"wss://{'demo-trading-' if BLOFIN_DEMO_MODE else ''}openapi.blofin.com/ws/private"

# Symbol mapping (Grok format -> Blofin format)
BLOFIN_SYMBOL_MAP = {
    "BTC/USDT": "BTC-USDT",
    "ETH/USDT": "ETH-USDT",
    "SOL/USDT": "SOL-USDT",
    "BNB/USDT": "BNB-USDT",
    "XRP/USDT": "XRP-USDT",
    "ADA/USDT": "ADA-USDT",
    "AVAX/USDT": "AVAX-USDT",
}

def is_blofin_configured() -> bool:
    """Check if Blofin API credentials are configured"""
    return bool(BLOFIN_API_KEY and BLOFIN_SECRET_KEY and BLOFIN_PASSPHRASE)

def get_blofin_config_summary() -> str:
    """Get Blofin configuration summary for status display"""
    if not is_blofin_configured():
        return "Blofin: Not Configured"
    
    status = "ON" if AUTO_TRADE_ENABLED else "OFF"
    mode = "DEMO" if BLOFIN_DEMO_MODE else "LIVE"
    broker_info = f"'{BLOFIN_BROKER_ID}'" if BLOFIN_BROKER_ID else "(none - Transaction API Key)"
    
    return f"""
**Blofin Auto-Trading v27.12.13**
- Mode: {mode}
- Enabled: {AUTO_TRADE_ENABLED}
- Broker ID: {broker_info}
- Risk/Trade: {AUTO_TRADE_RISK_PCT*100:.1f}%
- Max Leverage: {AUTO_TRADE_MAX_LEVERAGE}x
- Default Leverage: {AUTO_TRADE_DEFAULT_LEVERAGE}x
- Margin Mode: {AUTO_TRADE_MARGIN_MODE}
- Min Grade: {AUTO_TRADE_MIN_GRADE}
- Session Trading: {SESSION_TRADING_ENABLED}
- Correlation Sizing: {CORRELATION_SIZING_ENABLED}
- Drawdown Protection: {DRAWDOWN_STOP_PCT*100:.0f}% stop
"""

# ============================================================================
# FEATURE FLAGS
# ============================================================================
VOL_SPIKE_ENABLED = False
USE_STRUCTURE_SL = True
OB_SCORING_ENABLED = True
COUNTER_TREND_TP1_ONLY = True
MANIPULATION_DETECTION_ENABLED = True  # v27.12.10: Verified enabled

# Phase 2 features
SIGNAL_GRADING_ENABLED = True
GROK_OPINION_ENABLED = True
STRUCTURE_DETECTION_ENABLED = True
PSYCHOLOGY_ENABLED = True
BACKGROUND_MONITOR_ENABLED = False

# v27.10.0: Roadmap features
STRUCTURAL_BOUNCE_ENABLED = True

# ============================================================================
# SIGNAL GRADING THRESHOLDS
# ============================================================================
GRADE_A_THRESHOLD = 85
GRADE_B_THRESHOLD = 70
GRADE_C_THRESHOLD = 55
GRADE_D_THRESHOLD = 40

EXECUTABLE_GRADES = ['A', 'B', 'C']

GRADE_SIZE_MULT: Dict[str, float] = {
    'A': 1.0,
    'B': 0.85,
    'C': 0.65,
    'D': 0.0,
    'F': 0.0
}

MIN_GRADE_LIVE = 'B'
MIN_GRADE_ROADMAP = 'C'

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
SIMULATED_CAPITAL = 10000.0
RISK_PER_TRADE = 0.015
MAX_DRAWDOWN = 0.03
MAX_OPEN_TRADES = 3
MAX_LEVERAGE = 5
DEFAULT_LEVERAGE = 3

# v27.12.0 FIX: Add aliases for variables imported by main.py and trading.py
MAX_CONCURRENT_TRADES = MAX_OPEN_TRADES  # Alias used by main.py
MAX_DRAWDOWN_PCT = MAX_DRAWDOWN          # Alias used by main.py, trading.py
RISK_PER_TRADE_PCT = RISK_PER_TRADE      # Alias used by trading.py

# ============================================================================
# CONFLUENCE & QUALITY SCORING
# ============================================================================
MIN_CONFLUENCE_FACTORS = 3
MIN_CONFLUENCE_LIVE = 4
MIN_CONFLUENCE_ROADMAP = 2
MIN_TRADE_QUALITY_SCORE = 55
OB_MIN_QUALITY_SCORE = 50  # v27.12.0 FIX: Was missing, used by main.py

# ============================================================================
# ORDER BLOCK PARAMETERS
# ============================================================================
OB_MIN_STRENGTH = 2.0
OB_MAX_MITIGATION = 0.6
OB_DISTANCE_PCT = 5.0
OB_LOOKBACK = 100
OB_MIN_MOVE_PCT = 0.5
OB_OVERLAP_THRESHOLD = 0.3

# ============================================================================
# ROADMAP SETTINGS - ORDER BLOCKS
# ============================================================================
ROADMAP_MIN_OB_STRENGTH = 1.5
ROADMAP_MIN_DEPTH_BTC = 300000
ROADMAP_MIN_DEPTH_ALT = 100000

# ============================================================================
# FVG & LIQUIDITY PARAMETERS
# ============================================================================
FVG_MIN_SIZE_ATR = 0.5
LIQ_SWEEP_PCT = 0.3
LIQ_PROXIMITY_PCT = 0.5
SWEEP_THRESHOLD = 0.002

# ============================================================================
# VOLUME & ATR
# ============================================================================
VOL_SURGE_MULTIPLIER = 3.0
DAILY_ATR_MULT = 1.5
ATR_LOOKBACK = 14

# ============================================================================
# PREMIUM/DISCOUNT ZONES
# ============================================================================
PREMIUM_PCT = 75.0
DISCOUNT_PCT = 25.0
EQ_ZONE_UPPER = 55.0
EQ_ZONE_LOWER = 45.0
ZONE_LOOKBACK = 100
EXTREME_PREMIUM_PCT = 85.0
EXTREME_DISCOUNT_PCT = 15.0
STRONG_PREMIUM_PCT = 70.0
STRONG_DISCOUNT_PCT = 30.0

# ============================================================================
# FUNDING RATE & OI
# ============================================================================
FUNDING_EXTREME = 0.05
OI_CHANGE_SIGNIFICANT = 3.0
FUNDING_THRESHOLD = 0.01
OI_DIVERGENCE_THRESHOLD = 5.0
ORDERBOOK_IMBALANCE_THRESHOLD = 0.3

# ============================================================================
# STRUCTURAL BOUNCE SETTINGS
# ============================================================================
STRUCTURAL_MIN_DISTANCE_PCT = 0.5
STRUCTURAL_MAX_DISTANCE_PCT = 8.0
STRUCTURAL_TP1_PCT = 2.5
STRUCTURAL_TP2_PCT = 5.0
STRUCTURAL_SL_PCT = 2.5
STRUCTURAL_MIN_OB_STRENGTH = 1.5
STRUCTURAL_MIN_CONFIDENCE = 55
STRUCTURAL_COOLDOWN_HOURS = 8
STRUCTURAL_MIN_TOUCHES = 2
STRUCTURAL_COUNTER_TREND_ENABLED = True
STRUCTURAL_MAX_ZONES = 2

# v27.10.1: ATR-based TP multipliers for structural bounces
STRUCTURAL_TP1_ATR_MULT = 2.0
STRUCTURAL_TP2_ATR_MULT = 4.0

STRUCTURAL_PSYCH_LEVELS = {
    'BTC/USDT': [80000, 85000, 90000, 95000, 100000, 105000, 110000],
    'ETH/USDT': [3000, 3500, 4000, 4500, 5000],
    'SOL/USDT': [100, 125, 150, 175, 200, 225, 250],
    'BNB/USDT': [500, 600, 700, 800, 900, 1000],
    'XRP/USDT': [1.0, 1.5, 2.0, 2.5, 3.0],
    'ADA/USDT': [0.5, 0.75, 1.0, 1.25, 1.5],
    'AVAX/USDT': [30, 40, 50, 60, 70, 80]
}

STRUCTURAL_EXPECTED_WIN_RATE = 65.0
STRUCTURAL_EXPECTED_AVG_BOUNCE = 3.5

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================
MIN_CONFIDENCE = 60
BASE_CONFIDENCE_THRESHOLD = 60
STOP_LOSS_PCT = 5.0

# ============================================================================
# TAKE PROFIT & STOP LOSS
# ============================================================================
TP1_RATIO = 0.5
TP2_RATIO = 0.5
MIN_RR_RATIO = 1.5
IDEAL_RR_RATIO = 2.0
TRAILING_STOP_PCT = 0.5

# v27.12.13: Dynamic TP based on volatility regime
DYNAMIC_TP_ENABLED = os.getenv("DYNAMIC_TP_ENABLED", "true").lower() == "true"

# TP multipliers by volatility regime (in R-multiples)
DYNAMIC_TP_HIGH_VOL = {
    'tp1_r': 2.0,    # 2R for TP1 in high volatility
    'tp2_r': 4.0,    # 4R for TP2 in high volatility
}

DYNAMIC_TP_MEDIUM_VOL = {
    'tp1_r': 1.75,   # 1.75R for TP1 in medium volatility
    'tp2_r': 3.0,    # 3R for TP2 in medium volatility
}

DYNAMIC_TP_LOW_VOL = {
    'tp1_r': 1.5,    # 1.5R for TP1 in low volatility
    'tp2_r': 2.5,    # 2.5R for TP2 in low volatility
}

# ============================================================================
# v27.12.13: MULTI-TIMEFRAME CONFLUENCE WEIGHTING
# ============================================================================
MTF_WEIGHTING_ENABLED = os.getenv("MTF_WEIGHTING_ENABLED", "true").lower() == "true"

# Timeframe weights for confluence scoring
MTF_WEIGHTS = {
    '1d': 0.40,    # Daily is most important (40%)
    '4h': 0.35,    # 4-hour is secondary (35%)
    '1h': 0.25,    # 1-hour for entry timing (25%)
}

# Minimum MTF alignment score for signal validation
MTF_MIN_ALIGNMENT_SCORE = 0.5  # At least 50% alignment

# ============================================================================
# v27.12.13: SIGNAL PERFORMANCE TRACKING
# ============================================================================
SIGNAL_TRACKING_ENABLED = os.getenv("SIGNAL_TRACKING_ENABLED", "true").lower() == "true"
FACTOR_PERFORMANCE_FILE = os.getenv("FACTOR_PERFORMANCE_FILE", "data/factor_performance.json")

# ============================================================================
# v27.12.13: LIQUIDITY ANALYSIS
# ============================================================================
LIQUIDITY_ANALYSIS_ENABLED = os.getenv("LIQUIDITY_ANALYSIS_ENABLED", "true").lower() == "true"
LIQUIDITY_MIN_CLUSTER_SIGNIFICANCE = 3.0

# ============================================================================
# v27.12.13: ORDERBOOK IMBALANCE
# ============================================================================
ORDERBOOK_ANALYSIS_ENABLED = os.getenv("ORDERBOOK_ANALYSIS_ENABLED", "true").lower() == "true"
ORDERBOOK_IMBALANCE_THRESHOLD_LONG = 1.5   # Bid/ask ratio > 1.5 = bullish
ORDERBOOK_IMBALANCE_THRESHOLD_SHORT = 0.67  # Bid/ask ratio < 0.67 = bearish

# ============================================================================
# v27.12.14: EARLY REVERSAL DETECTION
# ============================================================================
EARLY_REVERSAL_ENABLED = os.getenv("EARLY_REVERSAL_ENABLED", "true").lower() == "true"

# Divergence settings
RSI_DIVERGENCE_LOOKBACK = 30
RSI_OVERSOLD_THRESHOLD = 30
RSI_OVERBOUGHT_THRESHOLD = 70

# Candlestick pattern settings
CANDLESTICK_PATTERNS_ENABLED = True
VOLUME_SPIKE_MULTIPLIER = 1.5  # Volume must be 1.5x average for confirmation

# Chart pattern settings
CHART_PATTERNS_ENABLED = True
DOUBLE_BOTTOM_TOP_TOLERANCE_PCT = 2.0
HEAD_SHOULDERS_LOOKBACK = 60

# Confluence requirement
MIN_REVERSAL_CONFLUENCE = 2  # Need at least 2 signals for valid reversal

# ============================================================================
# EXCHANGE CONFIG
# ============================================================================
EXCHANGE_CONFIG = {
    'apiKey': None,
    'secret': None,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True
    }
}

FUTURES_EXCHANGE_CONFIG = {
    'apiKey': None,
    'secret': None,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True
    }
}

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ATR_MULT = 1.5
TRAILING_ACTIVATION_PCT = 1.5
TRADE_TIMEOUT_HOURS = 48
PROTECT_AFTER_HOURS = 4

# ============================================================================
# SYMBOL CATEGORIES
# ============================================================================
MAJOR_SYMBOLS = ['BTC/USDT', 'ETH/USDT']
L1_ALT_SYMBOLS = ['SOL/USDT', 'AVAX/USDT', 'ADA/USDT']

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# MESSAGING
# ============================================================================
MAX_MESSAGE_LENGTH = 4000
MESSAGE_RATE_LIMIT = 1.0
THROTTLE_DELAY = 0.5
TELEGRAM_MAX_LENGTH = 4000

# ============================================================================
# FEE SETTINGS
# ============================================================================
FEE_PCT = 0.001
SLIPPAGE_PCT = 0.001
ENTRY_SLIPPAGE_PCT = 0.005

# ============================================================================
# INDICATOR SETTINGS
# ============================================================================
FVG_DISPLACEMENT_MULT = 1.5
PRE_CROSS_THRESHOLD_PCT = 2.0
DISPLACEMENT_BODY_RATIO = 0.6
PINBAR_WICK_RATIO = 2.0
ORDERBOOK_DEPTH = 20
WALL_MULTIPLIER = 2.0
MOMENTUM_ROC_PERIOD = 14
MOMENTUM_LOOKBACK = 20
MOMENTUM_STRONG_THRESHOLD = 5.0

# ============================================================================
# CACHE SETTINGS
# ============================================================================
CACHE_TTL = 120
HTF_CACHE_TTL = 600
TICKER_CACHE_TTL = 10
ORDER_FLOW_CACHE_TTL = 30
FETCH_SEMAPHORE_LIMIT = 5
MAX_CACHE_SIZE = 1000

# ============================================================================
# BINANCE INTEGRATION
# ============================================================================
USE_BINANCE_DATA = True
EXCHANGE_DIVERGENCE_SIGNIFICANT_PCT = 0.3
FUNDING_EXTREME_PCT = 0.05

# ============================================================================
# VOLATILITY SPIKE PARAMETERS
# ============================================================================
VOL_SPIKE_THRESHOLD_PCT = 3.0
VOL_SPIKE_VOLUME_MULT = 3.0
VOL_SPIKE_ATR_MULT = 2.0
VOL_SPIKE_COOLDOWN_MIN = 30

# ============================================================================
# BACKGROUND MONITOR SETTINGS
# ============================================================================
BACKGROUND_CHECK_INTERVAL = 7200
BACKGROUND_TRIGGER_COOLDOWN = 1800
BACKGROUND_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
VOL_SPIKE_TRIGGER_PCT = 2.5

# ============================================================================
# CLAUDE API SETTINGS
# ============================================================================
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_TEMPERATURE = 0.1
CLAUDE_MAX_TOKENS = 2000
CLAUDE_TIMEOUT = 60.0
CLAUDE_MAX_RETRIES = 3
CLAUDE_MIN_CONFIDENCE = 60
CLAUDE_MIN_STRENGTH = 2.0  # v27.12.0 FIX: Was missing, used by main.py

# Claude trade parameters
CLAUDE_TP1_ATR_MULT = 1.5
CLAUDE_TP2_ATR_MULT = 3.0
CLAUDE_SL_ATR_MULT = 1.0

# ============================================================================
# GROK API SETTINGS - v27.12.0 FIXED
# ============================================================================
# CRITICAL FIX: "grok-beta" is DEPRECATED and returns 404
# Updated to current working model names
GROK_MODELS = ["grok-3", "grok-3-mini"]
GROK_TEMPERATURE = 0.3
GROK_VALIDATION_MAX_TOKENS = 300
GROK_RECAP_MAX_TOKENS = 1500
GROK_MAX_RETRIES = 2

# Grok validation thresholds
GROK_MAX_ENTRY_DISTANCE_PCT = 15.0
GROK_MIN_SL_DISTANCE_PCT = 0.3
GROK_MAX_SL_DISTANCE_PCT = 5.0
GROK_MIN_RR_RATIO = 1.5

# ============================================================================
# FILE PATHS
# ============================================================================
STATS_FILE = "data/stats.json"
TRADES_FILE = "data/trades.json"
PROTECTED_TRADES_FILE = "data/protected_trades.json"
BACKTEST_FILE = "data/backtest.json"
ROADMAP_FILE = "data/roadmaps.json"
RECAP_FILE = "data/last_recap.txt"
FLAG_FILE = "data/welcome_sent.flag"
TRADE_LOG_FILE = "data/trades_log.csv"
FACTOR_PERFORMANCE_FILE = "data/factor_performance.json"
HISTORICAL_DATA = "data/historical_ohlcv.json"

# ============================================================================
# SESSION TIMES (UTC)
# ============================================================================
ASIAN_START = 0
ASIAN_END = 8
LONDON_START = 7
LONDON_END = 16
NY_START = 13
NY_END = 22

# ============================================================================
# SCHEDULE TIMES
# ============================================================================
SIGNAL_CHECK_HOURS = [1, 5, 9, 13, 17, 21]
ROADMAP_GEN_HOURS = [0, 12]

# ============================================================================
# INTERVALS
# ============================================================================
CHECK_INTERVAL = 7200
CHECK_INTERVAL_HIGH_VOL = 3600
CHECK_INTERVAL_MED_VOL = 5400
TRACK_INTERVAL = 60

HIGH_VOL_ATR_PCT = 5.0
MED_VOL_ATR_PCT = 3.0

# ============================================================================
# DYNAMIC COOLDOWN MAP
# ============================================================================
DYNAMIC_COOLDOWN_MAP = {
    '1h': 2,
    '4h': 4,
    '1d': 8,
    '1w': 24
}

# ============================================================================
# GROK OPINION SETTINGS
# ============================================================================
GROK_OPINION_TIMEOUT = 12.0
GROK_OPINION_DISPLAY_VERBOSE = True

# ============================================================================
# STRUCTURE DETECTION SETTINGS
# ============================================================================
STRUCTURE_SWING_LOOKBACK = 5
CHOCH_IMMEDIATE_TRIGGER = True

# ============================================================================
# MARKET PSYCHOLOGY SETTINGS
# ============================================================================
PSYCHOLOGY_CACHE_TTL = 300

FG_EXTREME_FEAR = 20
FG_FEAR = 35
FG_GREED = 65
FG_EXTREME_GREED = 80

LS_CROWDED_LONG = 0.70
LS_CROWDED_SHORT = 0.30

# ============================================================================
# v27.12.10: ROADMAP LIMITS (QUALITY FOCUSED)
# ============================================================================
RELAXED_MAX_ZONES_TREND = 5          # v27.12.10: Max 5 trend roadmaps (was 15)
RELAXED_MAX_ZONES_STRUCTURAL = 2     # v27.12.10: Max 2 structural bounces (was 8)
ROADMAP_MAX_DISTANCE_PCT = 7.0       # v27.12.10: NEW - Only zones within 7% of price

# ============================================================================
# v27.12.0: ROADMAP PROXIMITY & CONVERSION
# ============================================================================
ROADMAP_ALERT_COOLDOWN_MINUTES = 120
ROADMAP_CONVERSION_TRIGGER_PCT = 1.0
ROADMAP_ALERT_PROXIMITY_PCT = 2.5
ROADMAP_GENERATION_TIMES = [(0, 5), (15, 0)]

# v27.12.0: Roadmap validation settings
RELAXED_MIN_OB_STRENGTH = 1.2
RELAXED_MIN_CONFIDENCE = 50
RELAXED_MIN_DISTANCE_PCT = 0.15
RELAXED_MAX_DISTANCE_PCT = 12.0
RELAXED_OB_DISTANCE_PCT = 15.0

# Validation relaxation
RELAXED_VOL_SURGE = 1.1
RELAXED_DEPTH_BTC = 250000
RELAXED_DEPTH_ALT = 100000
RELAXED_SPREAD_PCT = 0.20

# ============================================================================
# ROADMAP SETTINGS (REQUIRED BY roadmap.py, utils.py)
# ============================================================================
TREND_ROADMAP_MAX_ZONES = 10
TREND_ROADMAP_MIN_OB_STRENGTH = 1.8
TREND_ROADMAP_MIN_CONFIDENCE = 60
ROADMAP_ENTRY_PROXIMITY_PCT = 0.5
ROADMAP_MIN_VOL_SURGE = 1.2

# ============================================================================
# GROK ROADMAP OPINION (v27.12.3)
# ============================================================================
GROK_ROADMAP_OPINION_ENABLED = True

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================
def validate_config():
    """Validate required config is present."""
    required = [TELEGRAM_TOKEN, CHAT_ID]
    if not all(required):
        raise ValueError("Missing required config: TELEGRAM_TOKEN and CHAT_ID")

# ============================================================================
# HELPER FUNCTION
# ============================================================================
def get_config_summary() -> str:
    """Get summary of config for logging."""
    blofin_status = "ON" if is_blofin_configured() and AUTO_TRADE_ENABLED else "OFF"
    blofin_mode = "DEMO" if BLOFIN_DEMO_MODE else "LIVE"
    
    return f"""
Grok Elite Bot v{BOT_VERSION} Config:
- Symbols: {len(SYMBOLS)}
- Roadmap: {RELAXED_MAX_ZONES_TREND} trend + {RELAXED_MAX_ZONES_STRUCTURAL} structural
- Max Zone Distance: {ROADMAP_MAX_DISTANCE_PCT}%
- Proximity cooldown: {ROADMAP_ALERT_COOLDOWN_MINUTES}min
- TP scaling: Dynamic ATR-based
- Grading: A>={GRADE_A_THRESHOLD}, B>={GRADE_B_THRESHOLD}, C>={GRADE_C_THRESHOLD}
- Executable grades: {EXECUTABLE_GRADES}
- Min OB: Live={OB_MIN_STRENGTH}, Roadmap={ROADMAP_MIN_OB_STRENGTH}
- Min Confluence: Live={MIN_CONFLUENCE_LIVE}, Roadmap={MIN_CONFLUENCE_ROADMAP}
- Grok Opinion: {'ON' if GROK_OPINION_ENABLED else 'OFF'}
- Grok Models: {GROK_MODELS}
- Psychology: {'ON' if PSYCHOLOGY_ENABLED else 'OFF'}
- Structure Detection: {'ON' if STRUCTURE_DETECTION_ENABLED else 'OFF'}
- Manipulation Detection: {'ON' if MANIPULATION_DETECTION_ENABLED else 'OFF'}
- Blofin Auto-Trade: {blofin_status} ({blofin_mode})
"""
