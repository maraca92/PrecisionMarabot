# emoji_utils.py - Grok Elite Signal Bot v27.11.0 - Emoji Utilities
# -*- coding: utf-8 -*-
"""
Centralized emoji handling to avoid encoding issues in Telegram messages.

IMPORTANT: This file uses Unicode escape sequences instead of literal emoji
characters to ensure proper encoding when deployed on servers with different
locale settings.

Usage:
    from bot.emoji_utils import EMOJI, get_emoji, format_with_emoji
    
    msg = f"{get_emoji('rocket')} Bot started!"
    # or
    msg = f"{EMOJI['rocket']} Bot started!"
"""

from typing import Dict, Optional

# ============================================================================
# EMOJI DICTIONARY (Unicode escape sequences)
# ============================================================================
EMOJI: Dict[str, str] = {
    # Status indicators
    'check': '\u2705',           # ✅
    'cross': '\u274C',           # ❌
    'warning': '\u26A0\uFE0F',   # ⚠️
    'stop': '\U0001F6D1',        # 🛑
    'alarm': '\U0001F6A8',       # 🚨
    
    # Arrows and directions
    'up': '\u2B06\uFE0F',        # ⬆️
    'down': '\u2B07\uFE0F',      # ⬇️
    'right': '\u27A1\uFE0F',     # ➡️
    'left': '\u2B05\uFE0F',      # ⬅️
    
    # Charts and graphs
    'chart': '\U0001F4C8',       # 📈
    'chart_down': '\U0001F4C9',  # 📉
    'graph': '\U0001F4CA',       # 📊
    'candle': '\U0001F56F\uFE0F',# 🕯️
    
    # Money and trading
    'money': '\U0001F4B0',       # 💰
    'dollar': '\U0001F4B5',      # 💵
    'gem': '\U0001F48E',         # 💎
    'rocket': '\U0001F680',      # 🚀
    'fire': '\U0001F525',        # 🔥
    'lightning': '\u26A1',       # ⚡
    
    # Targets and goals
    'target': '\U0001F3AF',      # 🎯
    'pin': '\U0001F4CD',         # 📍
    'star': '\U0001F31F',        # 🌟
    'trophy': '\U0001F3C6',      # 🏆
    
    # Time and calendar
    'clock': '\U0001F551',       # 🕐
    'calendar': '\U0001F4C5',    # 📅
    'hourglass': '\u23F3',       # ⏳
    
    # Colors (circles)
    'green': '\U0001F7E2',       # 🟢
    'red': '\U0001F534',         # 🔴
    'yellow': '\U0001F7E1',      # 🟡
    'white': '\u26AA',           # ⚪
    'blue': '\U0001F535',        # 🔵
    'orange': '\U0001F7E0',      # 🟠
    
    # Documents and info
    'paper': '\U0001F4DD',       # 📝
    'book': '\U0001F4D6',        # 📖
    'bulb': '\U0001F4A1',        # 💡
    'gear': '\u2699\uFE0F',      # ⚙️
    'link': '\U0001F517',        # 🔗
    'key': '\U0001F511',         # 🔑
    
    # People and reactions
    'wave': '\U0001F44B',        # 👋
    'thumbs_up': '\U0001F44D',   # 👍
    'thumbs_down': '\U0001F44E', # 👎
    'eyes': '\U0001F440',        # 👀
    'brain': '\U0001F9E0',       # 🧠
    'muscle': '\U0001F4AA',      # 💪
    
    # Weather/Elements
    'sun': '\u2600\uFE0F',       # ☀️
    'moon': '\U0001F319',        # 🌙
    'cloud': '\u2601\uFE0F',     # ☁️
    'snowflake': '\u2744\uFE0F', # ❄️
    
    # Symbols
    'bullet': '\u2022',          # •
    'dash': '\u2014',            # —
    'arrow_right': '\u2192',     # →
    'arrow_left': '\u2190',      # ←
    'double_arrow': '\u00BB',    # »
    'circle': '\u25CF',          # ●
    'diamond': '\u25C6',         # ◆
    
    # Special trading emojis
    'long': '\U0001F7E2',        # 🟢 (same as green)
    'short': '\U0001F534',       # 🔴 (same as red)
    'profit': '\U0001F4B9',      # 💹
    'trending_up': '\U0001F4C8', # 📈
    'trending_down': '\U0001F4C9', # 📉
}


def get_emoji(name: str, fallback: str = '') -> str:
    """
    Get emoji by name with optional fallback.
    
    Args:
        name: Emoji name from EMOJI dictionary
        fallback: String to return if emoji not found (default: empty string)
        
    Returns:
        Unicode emoji string or fallback
        
    Example:
        >>> get_emoji('rocket')
        '🚀'
        >>> get_emoji('unknown', '[?]')
        '[?]'
    """
    return EMOJI.get(name, fallback)


def format_signal_header(symbol: str, direction: str, grade: str) -> str:
    """
    Format a signal header with appropriate emojis.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        direction: 'Long' or 'Short'
        grade: Signal grade ('A', 'B', 'C', 'D', 'F')
        
    Returns:
        Formatted header string
    """
    dir_emoji = EMOJI['green'] if direction == 'Long' else EMOJI['red']
    grade_emoji = {
        'A': EMOJI['star'],
        'B': EMOJI['check'],
        'C': EMOJI['lightning'],
        'D': EMOJI['warning'],
        'F': EMOJI['cross']
    }.get(grade, '')
    
    symbol_short = symbol.replace('/USDT', '')
    return f"{EMOJI['alarm']} **{symbol_short} {direction}** {dir_emoji} {grade_emoji} Grade {grade}"


def format_trade_status(status: str) -> str:
    """
    Format trade status with appropriate emoji.
    
    Args:
        status: 'win', 'loss', 'active', 'pending', 'cancelled'
        
    Returns:
        Formatted status string
    """
    status_map = {
        'win': f"{EMOJI['check']} WIN",
        'loss': f"{EMOJI['cross']} LOSS",
        'active': f"{EMOJI['green']} ACTIVE",
        'pending': f"{EMOJI['yellow']} PENDING",
        'cancelled': f"{EMOJI['white']} CANCELLED",
        'tp1': f"{EMOJI['target']} TP1 HIT",
        'tp2': f"{EMOJI['trophy']} TP2 HIT",
        'sl': f"{EMOJI['stop']} SL HIT"
    }
    return status_map.get(status.lower(), status)


def format_confidence(confidence: int) -> str:
    """
    Format confidence with color emoji.
    
    Args:
        confidence: Confidence percentage (0-100)
        
    Returns:
        Formatted confidence string
    """
    if confidence >= 80:
        emoji = EMOJI['green']
    elif confidence >= 65:
        emoji = EMOJI['yellow']
    else:
        emoji = EMOJI['red']
    return f"{emoji} {confidence}%"


def format_wick_signal(wick_type: str, confidence: float) -> str:
    """
    Format wick signal detection for Telegram.
    
    Args:
        wick_type: Type of wick pattern detected
        confidence: Detection confidence
        
    Returns:
        Formatted wick signal string
    """
    type_display = wick_type.replace('_', ' ').title()
    conf_emoji = EMOJI['fire'] if confidence >= 80 else EMOJI['lightning']
    return f"{EMOJI['candle']} **{type_display}** {conf_emoji} ({confidence:.0f}%)"


def build_status_line(label: str, enabled: bool) -> str:
    """
    Build a status line for feature display.
    
    Args:
        label: Feature name
        enabled: Whether feature is enabled
        
    Returns:
        Formatted status line
    """
    status = EMOJI['check'] if enabled else EMOJI['cross']
    return f"**{label}:** {status}"


# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

# Common emoji shortcuts
CHECK = EMOJI['check']
CROSS = EMOJI['cross']
WARNING = EMOJI['warning']
ROCKET = EMOJI['rocket']
FIRE = EMOJI['fire']
LIGHTNING = EMOJI['lightning']
TARGET = EMOJI['target']
ALARM = EMOJI['alarm']
GREEN = EMOJI['green']
RED = EMOJI['red']
YELLOW = EMOJI['yellow']
GRAPH = EMOJI['graph']
CANDLE = EMOJI['candle']


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test emoji display
    print("Testing emoji rendering:")
    print(f"Rocket: {EMOJI['rocket']}")
    print(f"Check: {EMOJI['check']}")
    print(f"Warning: {EMOJI['warning']}")
    print(f"Chart: {EMOJI['chart']}")
    print(f"Candle: {EMOJI['candle']}")
    print()
    
    # Test formatting functions
    print("Testing format functions:")
    print(format_signal_header("BTC/USDT", "Long", "A"))
    print(format_trade_status("win"))
    print(format_confidence(85))
    print(format_wick_signal("euphoria_top", 82.5))
    print(build_status_line("Wick Detection", True))
