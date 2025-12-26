# signal_formatting.py - Grok Elite Signal Bot v27.12.3 - Signal Formatting
"""
v27.12.3: DETAILED CONFLUENCE FACTOR DISPLAY

This module provides formatting functions to display exactly which
confluence factors triggered each signal, making it clear what
the bot is looking for and what it found.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone


# ============================================================================
# FACTOR CATEGORIES FOR DISPLAY
# ============================================================================

FACTOR_CATEGORIES = {
    'core': ['OB', 'FVG', 'Sweep', 'Liq'],           # Core ICT factors
    'momentum': ['Mom', 'Momentum', 'ROC'],           # Momentum factors
    'cross_exchange': ['Div', 'Divergence', 'ExtFund', 'OI', 'Vol'],  # Cross-exchange
    'structure': ['Struct', 'BOS', 'CHoCH', 'Structure'],  # Structure
    'confluence': ['MTF', 'OTE', 'Psych', 'Wick'],    # Additional confluence
}

FACTOR_EMOJIS = {
    'OB': '📦',
    'FVG': '📊',
    'Sweep': '🧹',
    'Liq': '💧',
    'Mom': '📈',
    'Momentum': '📈',
    'Div': '🔀',
    'Divergence': '🔀',
    'ExtFund': '💰',
    'OI': '📉',
    'Vol': '📊',
    'Struct': '🏗️',
    'BOS': '💥',
    'CHoCH': '🔄',
    'MTF': '⏰',
    'OTE': '🎯',
    'Psych': '🧠',
    'Wick': '🕯️',
}


# ============================================================================
# FORMAT CONFLUENCE FACTORS
# ============================================================================

def format_confluence_factors(
    factors: List[str],
    show_count: bool = True,
    max_display: int = 8,
    compact: bool = False
) -> str:
    """
    Format confluence factors for Telegram display.
    
    Args:
        factors: List of factor strings (e.g., ['OB', 'FVG', 'Mom(bull)'])
        show_count: Whether to show total count
        max_display: Maximum factors to display
        compact: Use compact format (no emojis)
    
    Returns:
        Formatted string for Telegram
    """
    if not factors:
        return "No factors" if compact else "❌ No confluence factors"
    
    # Clean and dedupe factors
    clean_factors = []
    seen = set()
    
    for f in factors:
        # Extract base factor name
        base = f.replace('+', '').split('(')[0].strip()
        if base not in seen:
            seen.add(base)
            clean_factors.append(f.replace('+', ''))
    
    # Limit display
    display_factors = clean_factors[:max_display]
    remaining = len(clean_factors) - max_display
    
    if compact:
        result = ", ".join(display_factors)
        if remaining > 0:
            result += f" +{remaining} more"
    else:
        # Add emojis
        formatted = []
        for f in display_factors:
            base = f.split('(')[0].strip()
            emoji = FACTOR_EMOJIS.get(base, '•')
            formatted.append(f"{emoji}{f}")
        
        result = " | ".join(formatted)
        if remaining > 0:
            result += f" (+{remaining})"
    
    if show_count:
        count_str = f"[{len(clean_factors)}]" if compact else f"**{len(clean_factors)} factors**"
        result = f"{count_str}: {result}"
    
    return result


def format_confluence_breakdown(
    factors: List[str],
    show_categories: bool = True
) -> str:
    """
    Format confluence factors with category breakdown.
    
    Returns multi-line formatted breakdown for verbose display.
    """
    if not factors:
        return "❌ No confluence factors detected"
    
    lines = [f"📋 **Confluence Breakdown** ({len(factors)} factors):"]
    
    if show_categories:
        # Categorize factors
        categorized = {cat: [] for cat in FACTOR_CATEGORIES}
        categorized['other'] = []
        
        for f in factors:
            base = f.replace('+', '').split('(')[0].strip()
            found = False
            for cat, cat_factors in FACTOR_CATEGORIES.items():
                if any(cf in base for cf in cat_factors):
                    categorized[cat].append(f)
                    found = True
                    break
            if not found:
                categorized['other'].append(f)
        
        # Format each category
        cat_names = {
            'core': '🔷 Core ICT',
            'momentum': '📈 Momentum',
            'cross_exchange': '🔄 Cross-Exchange',
            'structure': '🏗️ Structure',
            'confluence': '✨ Confluence',
            'other': '📌 Other'
        }
        
        for cat, cat_label in cat_names.items():
            if categorized[cat]:
                factor_str = ", ".join(categorized[cat])
                lines.append(f"  {cat_label}: {factor_str}")
    else:
        # Simple list
        lines.append("  " + ", ".join(factors))
    
    return "\n".join(lines)


# ============================================================================
# FORMAT SIGNAL MESSAGE WITH DETAILED FACTORS
# ============================================================================

def format_detailed_signal_message(
    symbol: str,
    direction: str,
    confidence: int,
    grade: str,
    quality_score: int,
    entry_low: float,
    entry_high: float,
    sl: float,
    tp1: float,
    tp2: float,
    leverage: int,
    confluence_factors: List[str],
    confluence_count: int,
    reason: str,
    grok_opinion: Dict,
    is_counter_trend: bool = False,
    timeframe_agreement: int = 0,
    rr1: float = 0,
    rr2: float = 0,
    ev_r: float = 0,
    fear_greed: Optional[Dict] = None,
    divergence: Optional[Dict] = None,
    momentum_aligned: bool = False,
    structure_aligned: bool = False,
    wick_detected: bool = False,
    structure_reason: str = "",
    format_price_func = None
) -> str:
    """
    Format a complete signal message with detailed confluence breakdown.
    
    v27.12.3: Shows exactly which factors triggered the signal.
    """
    # Use provided format_price or fallback
    if format_price_func is None:
        def format_price_func(p):
            return f"${p:.4f}" if p < 1 else f"${p:.2f}"
    
    # Emojis
    grade_emojis = {'A': '🌟', 'B': '✅', 'C': '⚡', 'D': '⚠️', 'F': '❌'}
    grade_emoji = grade_emojis.get(grade, '')
    dir_emoji = '🟢' if direction == 'Long' else '🔴'
    counter_tag = " ⚡CT" if is_counter_trend else ""
    tp_note = " (TP1 only)" if is_counter_trend else ""
    
    symbol_short = symbol.replace('/USDT', '')
    
    # Build message
    lines = []
    
    # Header
    lines.append(f"🚨 **{symbol_short} LIVE SIGNAL** {grade_emoji} Grade {grade}{counter_tag}")
    lines.append("")
    
    # Scores row
    lines.append(f"*Score:* {quality_score}/100 | *EV:* {ev_r:.2f}R | *MTF:* {timeframe_agreement}/2 ✓")
    
    # v27.12.3: Detailed Confluence Section
    lines.append("")
    lines.append(f"📊 **CONFLUENCE** ({confluence_count} factors):")
    
    # Format factors by type
    core_factors = [f for f in confluence_factors if any(x in f for x in ['OB', 'FVG', 'Sweep', 'Liq'])]
    momentum_factors = [f for f in confluence_factors if any(x in f for x in ['Mom', 'ROC'])]
    other_factors = [f for f in confluence_factors if f not in core_factors and f not in momentum_factors]
    
    if core_factors:
        lines.append(f"  🔷 Core: {', '.join(core_factors)}")
    if momentum_factors:
        lines.append(f"  📈 Momentum: {', '.join(momentum_factors)}")
    if other_factors:
        lines.append(f"  ✨ Other: {', '.join(other_factors[:5])}")
    
    # Psychology/Sentiment
    if fear_greed:
        fg_value = fear_greed.get('value', 50)
        fg_label = fear_greed.get('label', 'Neutral')
        fg_emoji = '📉' if fg_value <= 25 else '📈' if fg_value >= 75 else '📊'
        lines.append(f"*Psychology:* {fg_emoji} F&G={fg_value} ({fg_label})")
    
    # Divergence
    if divergence and divergence.get('significant'):
        div_signal = divergence.get('signal', 'N/A').upper()
        div_pct = divergence.get('divergence_pct', 0)
        lines.append(f"*Divergence:* {div_signal} {div_pct:+.2f}%")
    
    # Momentum & Structure
    if momentum_aligned:
        lines.append(f"*Momentum:* Aligned ✓")
    if structure_aligned:
        lines.append(f"*Structure:* {structure_reason}")
    
    # Wick signal
    if wick_detected:
        lines.append(f"*Wick Signal:* 🕯️ Detected")
    
    lines.append("")
    
    # Trade details
    lines.append(f"**{direction}** {dir_emoji} | *Conf:* {confidence}%{tp_note}")
    lines.append(f"*Entry:* {format_price_func(entry_low)} - {format_price_func(entry_high)}")
    
    tp_line = f"*SL:* {format_price_func(sl)} | *TP1:* {format_price_func(tp1)}"
    if not is_counter_trend:
        tp_line += f" | *TP2:* {format_price_func(tp2)}"
    lines.append(tp_line)
    
    rr_line = f"*Leverage:* {leverage}x | *R:R* 1:{rr1:.1f}"
    if not is_counter_trend:
        rr_line += f"/1:{rr2:.1f}"
    lines.append(rr_line)
    
    lines.append("")
    
    # Reason
    lines.append(f"**Reason:** {reason}")
    
    # Grok opinion
    if grok_opinion and grok_opinion.get('display'):
        lines.append("")
        lines.append(grok_opinion['display'])
    
    return "\n".join(lines)


# ============================================================================
# FORMAT FACTOR REQUIREMENTS
# ============================================================================

def format_factor_requirements(
    min_required: int = 5,
    current_count: int = 0,
    factors_found: List[str] = None
) -> str:
    """
    Format a display of factor requirements vs what was found.
    
    Useful for debugging why signals weren't generated.
    """
    factors_found = factors_found or []
    
    status = "✅ MET" if current_count >= min_required else "❌ NOT MET"
    
    lines = [
        f"**Factor Requirements** {status}",
        f"Required: {min_required} | Found: {current_count}",
    ]
    
    if factors_found:
        lines.append(f"Factors: {', '.join(factors_found[:8])}")
    
    return "\n".join(lines)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def count_unique_factors(factors: List[str]) -> int:
    """Count unique base factors (ignoring parameters)."""
    seen = set()
    for f in factors:
        base = f.replace('+', '').split('(')[0].strip()
        seen.add(base)
    return len(seen)


def get_factor_quality_score(factors: List[str]) -> int:
    """
    Calculate a quality score based on which factors are present.
    
    Core ICT factors are weighted higher than others.
    """
    score = 0
    
    # High-value factors (10 points each)
    high_value = ['OB', 'FVG', 'Sweep', 'Liq', 'OTE']
    for hv in high_value:
        if any(hv in f for f in factors):
            score += 10
    
    # Medium-value factors (5 points each)
    medium_value = ['Mom', 'Momentum', 'Struct', 'BOS', 'CHoCH', 'MTF']
    for mv in medium_value:
        if any(mv in f for f in factors):
            score += 5
    
    # Other factors (3 points each)
    other = ['Div', 'OI', 'Vol', 'Psych', 'Wick', 'ExtFund']
    for ot in other:
        if any(ot in f for f in factors):
            score += 3
    
    return min(100, score)
