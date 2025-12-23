# config.py - Grok Elite Signal Bot v27.12.3 - Configuration
"""
v27.12.3: STRICTER SIGNALS + BETTER FACTOR VISIBILITY

CHANGES:
1. MIN_CONFLUENCE_FACTORS: 4 → 5 (stricter)
2. MIN_CONFLUENCE_LIVE: 4 → 5 (stricter)
3. GRADE_B_THRESHOLD: 70 → 72 (stricter)
4. GRADE_C_THRESHOLD: 55 → 58 (stricter)
5. MIN_TRADE_QUALITY_SCORE: 55 → 60 (stricter)
6. Added SHOW_CONFLUENCE_DETAILS flag for verbose factor display
7. Added GROK_ROADMAP_OPINION_ENABLED for roadmap integration
"""
import os
from typing import Dict, List

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# ============================================================================
# BOT IDENTITY
# ============================================================================
BOT_VERSION = "27.12.3"

# ============================================================================
# API KEYS
# ============================================================================
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
CHAT_ID = os.getenv('CHAT_ID', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
XAI_API_KEY = os.getenv('XAI_API_KEY', '')

# ============================================================================
# TRADING SYMBOLS
# ============================================================================
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'AVAX/USDT']

# ============================================================================
# PAPER TRADING MODE
# ============================================================================
PAPER_TRADING = True

# ============================================================================
# v27.12.3: STRICTER GRADING THRESHOLDS
# ============================================================================
GRADE_A_THRESHOLD = 85    # Unchanged - A grade stays elite
GRADE_B_THRESHOLD = 72    # Was 70 - Slightly stricter
GRADE_C_THRESHOLD = 58    # Was 55 - Slightly stricter
GRADE_D_THRESHOLD = 40    # Unchanged

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

# Aliases for compatibility
MAX_CONCURRENT_TRADES = MAX_OPEN_TRADES
MAX_DRAWDOWN_PCT = MAX_DRAWDOWN
RISK_PER_TRADE_PCT = RISK_PER_TRADE

# ============================================================================
# v27.12.3: STRICTER CONFLUENCE & QUALITY SCORING
# ============================================================================
MIN_CONFLUENCE_FACTORS = 5      # Was 4 - Now require 5 factors minimum
MIN_CONFLUENCE_LIVE = 5         # Was 4 - Stricter for live signals
MIN_CONFLUENCE_ROADMAP = 3      # Was 2 - Slightly stricter for roadmaps
MIN_TRADE_QUALITY_SCORE = 60    # Was 55 - Higher quality bar
OB_MIN_QUALITY_SCORE = 50       # Unchanged

# v27.12.3: Factor visibility
SHOW_CONFLUENCE_DETAILS = True  # Show detailed factor breakdown in signals

# ============================================================================
# ORDER BLOCK PARAMETERS
# ============================================================================
OB_MIN_STRENGTH = 2.0           # Live signals need strong OBs
OB_MAX_MITIGATION = 0.6
OB_DISTANCE_PCT = 5.0
OB_LOOKBACK = 100
OB_MIN_MOVE_PCT = 0.5
OB_OVERLAP_THRESHOLD = 0.3

# ============================================================================
# ROADMAP SETTINGS - ORDER BLOCKS
# ============================================================================
ROADMAP_MIN_OB_STRENGTH = 1.5   # Roadmaps can use slightly weaker OBs
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
VOLUME_SURGE_THRESHOLD = 2.0
VOL_SURGE_MULTIPLIER = 3.0
ATR_PERIOD = 14

# ============================================================================
# TP SCALING - DYNAMIC ATR-BASED
# ============================================================================
TP1_ATR_MULT = 1.5
TP2_ATR_MULT = 3.0
SL_ATR_MULT = 1.5

# ============================================================================
# TRAILING STOP
# ============================================================================
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ATR_MULT = 1.5
TRAILING_ACTIVATION_PCT = 1.5
TRADE_TIMEOUT_HOURS = 48
PROTECT_AFTER_HOURS = 4

# ============================================================================
# FEES
# ============================================================================
FEE_PCT = 0.001

# ============================================================================
# ROADMAP PARAMETERS
# ============================================================================
TREND_ROADMAP_MAX_ZONES = 10
TREND_ROADMAP_MIN_OB_STRENGTH = 1.8
TREND_ROADMAP_MIN_CONFIDENCE = 60
ROADMAP_ENTRY_PROXIMITY_PCT = 0.5
STRUCTURAL_MAX_ZONES = 5
STRUCTURAL_BOUNCE_ENABLED = True
ROADMAP_MIN_VOL_SURGE = 1.2

# Relaxed parameters for more zones
RELAXED_MAX_ZONES_TREND = 15
RELAXED_MAX_ZONES_STRUCTURAL = 8
RELAXED_MIN_OB_STRENGTH = 1.5
RELAXED_MIN_CONFIDENCE = 55

# Proximity and alerts
ROADMAP_CONVERSION_TRIGGER_PCT = 1.0
ROADMAP_ALERT_PROXIMITY_PCT = 2.5
ROADMAP_ALERT_COOLDOWN_MINUTES = 120

# ============================================================================
# STRUCTURAL BOUNCE
# ============================================================================
STRUCTURAL_TP1_PCT = 2.0
STRUCTURAL_TP2_PCT = 4.0
STRUCTURAL_EXPECTED_WIN_RATE = 0.65
STRUCTURAL_EXPECTED_AVG_BOUNCE = 3.0

# ============================================================================
# CLAUDE API SETTINGS
# ============================================================================
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 600
CLAUDE_TEMPERATURE = 0.15
CLAUDE_MIN_CONFIDENCE = 65     # Minimum confidence from Claude
CLAUDE_MIN_STRENGTH = 2.0      # Minimum OB strength Claude should suggest

# ============================================================================
# v27.12.3: GROK API SETTINGS (ENHANCED)
# ============================================================================
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

# v27.12.3: Grok opinion settings
GROK_OPINION_ENABLED = True
GROK_OPINION_TIMEOUT = 12.0
GROK_ROADMAP_OPINION_ENABLED = True  # NEW: Enable Grok opinions on roadmap zones

# ============================================================================
# FEATURE FLAGS
# ============================================================================
PSYCHOLOGY_ENABLED = True
STRUCTURE_DETECTION_ENABLED = True
COUNTER_TREND_TP1_ONLY = True
WICK_DETECTION_ENABLED = True

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
# ZONE LOOKBACK
# ============================================================================
ZONE_LOOKBACK = 100

# ============================================================================
# AI CONSENSUS (for dual AI mode)
# ============================================================================
AI_CONSENSUS_MIN_CONFIDENCE = 60
AI_CONSENSUS_BOOST_AGREEMENT = 10
AI_CONSENSUS_PENALTY_DISAGREEMENT = 15
AI_MIN_CONFIDENCE_SOLO = 70

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate required config values."""
    required = [TELEGRAM_TOKEN, CHAT_ID]
    if not all(required):
        raise ValueError("Missing required config: TELEGRAM_TOKEN and CHAT_ID")

# ============================================================================
# HELPER FUNCTION
# ============================================================================
def get_config_summary() -> str:
    """Get summary of config for logging."""
    return f"""
Grok Elite Bot v{BOT_VERSION} Config:
- Symbols: {len(SYMBOLS)}
- Roadmap: {RELAXED_MAX_ZONES_TREND} trend + {RELAXED_MAX_ZONES_STRUCTURAL} structural
- TP scaling: Dynamic ATR-based
- Grading: A>={GRADE_A_THRESHOLD}, B>={GRADE_B_THRESHOLD}, C>={GRADE_C_THRESHOLD}
- Executable grades: {EXECUTABLE_GRADES}
- Min OB: Live={OB_MIN_STRENGTH}, Roadmap={ROADMAP_MIN_OB_STRENGTH}
- Min Confluence: Live={MIN_CONFLUENCE_LIVE}, Roadmap={MIN_CONFLUENCE_ROADMAP}
- Min Quality Score: {MIN_TRADE_QUALITY_SCORE}
- Grok Opinion: {'ON' if GROK_OPINION_ENABLED else 'OFF'}
- Grok Roadmap Opinion: {'ON' if GROK_ROADMAP_OPINION_ENABLED else 'OFF'}
- Grok Models: {GROK_MODELS}
- Psychology: {'ON' if PSYCHOLOGY_ENABLED else 'OFF'}
- Structure Detection: {'ON' if STRUCTURE_DETECTION_ENABLED else 'OFF'}
- Show Confluence Details: {'ON' if SHOW_CONFLUENCE_DETAILS else 'OFF'}
"""
