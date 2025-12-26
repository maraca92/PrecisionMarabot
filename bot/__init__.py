# bot/__init__.py - Grok Elite Signal Bot v27.12.3
"""
Grok Elite Trading Bot - Modular Architecture
v27.12.3: STRICTER SIGNALS + GROK ROADMAP OPINIONS + FACTOR VISIBILITY
"""
__version__ = "27.12.3"

VERSION_INFO = {
    'version': __version__,
    'phase': 'Production - Stricter Quality',
    'target_win_rate': '65%+',
    'philosophy': 'Quality over quantity - strict confluence requirements',
    
    'v27_12_3_changes': [
        'MIN_CONFLUENCE_FACTORS: 4 → 5 (stricter)',
        'MIN_CONFLUENCE_LIVE: 4 → 5 (stricter)',
        'GRADE_B_THRESHOLD: 70 → 72 (stricter)',
        'GRADE_C_THRESHOLD: 55 → 58 (stricter)',
        'MIN_TRADE_QUALITY_SCORE: 55 → 60 (stricter)',
        'Added GROK_ROADMAP_OPINION_ENABLED config',
        'Added get_grok_roadmap_opinion() function',
        'Added get_grok_market_sentiment() function',
        'Roadmap zones now show Grok opinion',
        'Added signal_formatting.py module',
        'Detailed confluence breakdown in signals',
    ],
    
    'thresholds': {
        'min_ob_strength_live': 2.0,
        'min_confidence_live': 65,
        'min_confluence_live': 5,
        'min_quality_score_live': 60,
        'min_ob_strength_roadmap': 1.5,
        'min_confidence_roadmap': 55,
        'min_confluence_roadmap': 3,
        'grade_a': 85,
        'grade_b': 72,
        'grade_c': 58,
        'grade_d': 40,
    },
    
    'signal_weights': {
        'confluence': 0.40,
        'ob_score': 0.30,
        'psychology': 0.20,
        'structure': 0.10,
    },
    
    'confluence_factors': {
        'core_ict': ['OB', 'FVG', 'Sweep', 'Liq', 'OTE'],
        'momentum': ['Mom', 'ROC', 'Momentum'],
        'cross_exchange': ['Div', 'OI', 'Vol', 'ExtFund'],
        'structure': ['Struct', 'BOS', 'CHoCH'],
        'other': ['MTF', 'Psych', 'Wick'],
    }
}

def get_version():
    return f"v{__version__}"

def get_version_info():
    return VERSION_INFO

def print_version_banner():
    """Print a nice version banner on startup"""
    banner = f"""
╔═══════════════════════════════════════════════════════════════════╗
║          GROK ELITE SIGNAL BOT v{__version__}                            ║
╠═══════════════════════════════════════════════════════════════════╣
║  Target Win Rate: 65%+                                            ║
║  Philosophy: STRICT Quality - 5+ Confluence Required              ║
╠═══════════════════════════════════════════════════════════════════╣
║  v27.12.3 HIGHLIGHTS:                                             ║
║  ✅ STRICTER: Min 5 confluence factors (was 4)                    ║
║  ✅ STRICTER: Grade B ≥72, Grade C ≥58                            ║
║  ✅ Grok opinions on ROADMAP zones (NEW)                          ║
║  ✅ Detailed factor breakdown in signals                          ║
║  ✅ All functions verified integrated                             ║
╠═══════════════════════════════════════════════════════════════════╣
║  CONFLUENCE: 5+ factors | GRADE: B+ | QUALITY: 60+                ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    return banner


def get_feature_status():
    """Return current feature status for /health command"""
    return {
        'signal_grading': True,
        'unified_evaluator': True,
        'grok_opinion_signals': True,
        'grok_opinion_roadmaps': True,
        'structure_detection': True,
        'psychology_analysis': True,
        'roadmap_trend': True,
        'roadmap_structural': True,
        'daily_recap': True,
        'wick_detection': True,
        'factor_visibility': True,
        'stricter_thresholds': True,
    }


def get_required_confluence() -> int:
    """Get minimum required confluence factors."""
    return 5


def validate_confluence(count: int) -> bool:
    """Check if confluence count meets minimum requirement."""
    return count >= get_required_confluence()
