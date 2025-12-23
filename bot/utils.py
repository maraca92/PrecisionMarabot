# utils.py - Grok Elite Signal Bot v27.12.9 - Utility Functions
"""
Utility functions for the bot.
v27.12.9: Auto-split long messages to handle Telegram's 4096 char limit
v27.8.5: Relaxed entry validation for more trades
"""
import logging
import asyncio
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from telegram import Bot
from telegram.error import RetryAfter, TelegramError

from bot.config import (
    TELEGRAM_TOKEN, CHAT_ID,
    GROK_MAX_ENTRY_DISTANCE_PCT, GROK_MIN_SL_DISTANCE_PCT,
    GROK_MAX_SL_DISTANCE_PCT, GROK_MIN_RR_RATIO,
    ROADMAP_GENERATION_TIMES, LOG_LEVEL, LOG_FORMAT, MAX_CACHE_SIZE
)

# ============================================================================
# JSON ENCODER
# ============================================================================
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

# ============================================================================
# CACHE UTILITIES
# ============================================================================
def evict_if_full(cache: OrderedDict, max_size: int = None):
    if max_size is None:
        max_size = MAX_CACHE_SIZE
    while len(cache) >= max_size:
        cache.popitem(last=False)

def cache_get(cache: OrderedDict, key: str) -> Optional[Dict]:
    return cache.get(key)

# ============================================================================
# SYMBOL UTILITIES
# ============================================================================
def get_clean_symbol(trade_key: str) -> str:
    for suffix in ['_roadmap', '_structural', '_protected']:
        if trade_key.endswith(suffix):
            trade_key = trade_key[:-len(suffix)]
            break
    return trade_key

# ============================================================================
# TELEGRAM MESSAGE SENDING - v27.12.9 FIX: Auto-split long messages
# ============================================================================
_bot: Optional[Bot] = None
_last_send_time: Dict[str, datetime] = {}
_min_interval = 1.0
MAX_TELEGRAM_LENGTH = 4000  # Leave buffer below 4096 limit

def get_bot() -> Bot:
    global _bot
    if _bot is None:
        _bot = Bot(token=TELEGRAM_TOKEN)
    return _bot

def split_message(text: str, max_length: int = MAX_TELEGRAM_LENGTH) -> List[str]:
    """Split long message into chunks that fit Telegram's limit."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Try to split on double newlines first (paragraph breaks)
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        # If single paragraph is too long, split on single newlines
        if len(para) > max_length:
            lines = para.split('\n')
            for line in lines:
                if len(current_chunk) + len(line) + 2 > max_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
        else:
            if len(current_chunk) + len(para) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + '\n\n'
            else:
                current_chunk += para + '\n\n'
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_length]]

async def send_throttled(
    chat_id: str,
    text: str,
    parse_mode: str = 'Markdown',
    disable_preview: bool = True
) -> bool:
    """Send message with throttling and automatic splitting for long messages."""
    global _last_send_time
    
    try:
        # v27.12.9: Split long messages automatically
        chunks = split_message(text)
        
        for i, chunk in enumerate(chunks):
            now = datetime.now(timezone.utc)
            last_time = _last_send_time.get(str(chat_id))
            
            if last_time:
                elapsed = (now - last_time).total_seconds()
                if elapsed < _min_interval:
                    await asyncio.sleep(_min_interval - elapsed)
            
            bot = get_bot()
            
            # Add continuation indicator for multi-part messages
            if len(chunks) > 1:
                if i == 0:
                    chunk = chunk + f"\n\n_(continued in {len(chunks)-1} more message{'s' if len(chunks) > 2 else ''})_"
                elif i < len(chunks) - 1:
                    chunk = f"_(continued)_\n\n" + chunk
                else:
                    chunk = f"_(final part)_\n\n" + chunk
            
            await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_preview
            )
            
            _last_send_time[str(chat_id)] = datetime.now(timezone.utc)
            
            # Small delay between chunks
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
        
        return True
        
    except RetryAfter as e:
        logging.warning(f"Rate limited, waiting {e.retry_after}s")
        await asyncio.sleep(e.retry_after)
        return await send_throttled(chat_id, text, parse_mode, disable_preview)
        
    except TelegramError as e:
        logging.error(f"Telegram error: {e}")
        return False
        
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        return False

# ============================================================================
# BAN MANAGER
# ============================================================================
class BanManager:
    _banned_until: Optional[datetime] = None
    _consecutive_errors = 0
    
    @classmethod
    async def check_and_sleep(cls, skip_long_sleep: bool = False) -> bool:
        if cls._banned_until is None:
            return False
        
        now = datetime.now(timezone.utc)
        
        if now < cls._banned_until:
            wait_time = (cls._banned_until - now).total_seconds()
            
            if skip_long_sleep and wait_time > 60:
                logging.info(f"Skipping long ban wait ({wait_time:.0f}s)")
                return True
            
            logging.warning(f"Ban active, waiting {wait_time:.0f}s")
            await asyncio.sleep(min(wait_time, 300))
            return True
        
        cls._banned_until = None
        return False
    
    @classmethod
    def set_ban(cls, seconds: int):
        cls._banned_until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        cls._consecutive_errors += 1
        logging.warning(f"Ban set for {seconds}s")
    
    @classmethod
    def clear_ban(cls):
        cls._banned_until = None
        cls._consecutive_errors = 0

# ============================================================================
# GROK VALIDATION HELPERS
# ============================================================================
def check_precision(trade: Dict) -> bool:
    entry_low = trade.get('entry_low', 0)
    entry_high = trade.get('entry_high', 0)
    sl = trade.get('sl', 0)
    tp1 = trade.get('tp1', 0)
    
    for val in [entry_low, entry_high, sl, tp1]:
        if val > 0:
            if val % 1 != 0 or (val % 100 != 0 and val > 1000):
                return True
    
    return False

def check_rr(trade: Dict) -> bool:
    entry_mid = (trade.get('entry_low', 0) + trade.get('entry_high', 0)) / 2
    sl = trade.get('sl', 0)
    tp1 = trade.get('tp1', 0)
    
    if entry_mid <= 0 or sl <= 0 or tp1 <= 0:
        return False
    
    risk = abs(sl - entry_mid)
    reward = abs(tp1 - entry_mid)
    
    if risk <= 0:
        return False
    
    rr = reward / risk
    return rr >= GROK_MIN_RR_RATIO

def calibrate_grok_confidence(raw_confidence: int, factors: Dict[str, bool]) -> int:
    adjusted = raw_confidence
    
    if factors.get('liq_sweep'):
        adjusted += 3
    if factors.get('vol_surge'):
        adjusted += 2
    if factors.get('htf_align'):
        adjusted += 3
    if factors.get('oi_spike'):
        adjusted += 2
    
    return min(95, max(55, adjusted))

# ============================================================================
# TRADE VALIDATION - v27.8.5 RELAXED
# ============================================================================
def validate_grok_trade(trade: Dict, current_price: float, symbol: str) -> Tuple[bool, str]:
    """
    Validate a trade proposal from AI.
    v27.8.5: Relaxed entry distance from 10% to 15%
    """
    try:
        direction = trade.get('direction', '')
        entry_low = trade.get('entry_low', 0)
        entry_high = trade.get('entry_high', 0)
        sl = trade.get('sl', 0)
        tp1 = trade.get('tp1', 0)
        tp2 = trade.get('tp2', 0)
        
        if not all([direction, entry_low, entry_high, sl, tp1]):
            return False, "Missing required fields"
        
        if direction not in ['Long', 'Short']:
            return False, f"Invalid direction: {direction}"
        
        entry_mid = (entry_low + entry_high) / 2
        
        # v27.8.5: RELAXED entry distance check (15% instead of 10%)
        entry_dist_pct = abs(entry_mid - current_price) / current_price * 100
        if entry_dist_pct > GROK_MAX_ENTRY_DISTANCE_PCT:
            return False, f"Entry too far: {entry_dist_pct:.1f}% > {GROK_MAX_ENTRY_DISTANCE_PCT}%"
        
        # v27.8.5: RELAXED SL distance check (0.3% minimum instead of 0.5%)
        sl_dist_pct = abs(sl - entry_mid) / entry_mid * 100
        if sl_dist_pct < GROK_MIN_SL_DISTANCE_PCT:
            return False, f"SL too tight: {sl_dist_pct:.2f}% < {GROK_MIN_SL_DISTANCE_PCT}%"
        if sl_dist_pct > GROK_MAX_SL_DISTANCE_PCT:
            return False, f"SL too wide: {sl_dist_pct:.1f}% > {GROK_MAX_SL_DISTANCE_PCT}%"
        
        # Direction logic check
        if direction == 'Long':
            if sl >= entry_low:
                return False, "Long SL must be below entry"
            if tp1 <= entry_high:
                return False, "Long TP1 must be above entry"
        else:
            if sl <= entry_high:
                return False, "Short SL must be above entry"
            if tp1 >= entry_low:
                return False, "Short TP1 must be below entry"
        
        # v27.8.5: RELAXED R:R check (1.5:1 instead of 2:1)
        risk = abs(sl - entry_mid)
        reward = abs(tp1 - entry_mid)
        
        if risk > 0:
            rr_ratio = reward / risk
            if rr_ratio < GROK_MIN_RR_RATIO:
                return False, f"R:R too low: 1:{rr_ratio:.1f} < 1:{GROK_MIN_RR_RATIO}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"

# ============================================================================
# ZONE UTILITIES
# ============================================================================
def zones_overlap(low1: float, high1: float, low2: float, high2: float) -> float:
    overlap_low = max(low1, low2)
    overlap_high = min(high1, high2)
    
    if overlap_low >= overlap_high:
        return 0.0
    
    overlap_range = overlap_high - overlap_low
    smaller_range = min(high1 - low1, high2 - low2)
    
    if smaller_range <= 0:
        return 0.0
    
    return overlap_range / smaller_range

def calculate_zone_proximity(zone: Dict, current_price: float) -> Dict:
    zone_low = zone.get('zone_low', zone.get('entry_low', 0))
    zone_high = zone.get('zone_high', zone.get('entry_high', 0))
    
    if current_price <= 0 or zone_low <= 0:
        return {'zone_low': 0, 'zone_high': 0, 'dist_pct': 100, 'inside': False}
    
    zone_mid = (zone_low + zone_high) / 2
    
    if zone_low <= current_price <= zone_high:
        dist_pct = 0.0
        inside = True
    else:
        dist = min(abs(current_price - zone_low), abs(current_price - zone_high))
        dist_pct = dist / current_price * 100
        inside = False
    
    return {
        'zone_low': zone_low,
        'zone_high': zone_high,
        'dist_pct': dist_pct,
        'inside': inside
    }

def is_approaching_zone(zone: Dict, current_price: float, prev_price: float) -> bool:
    zone_low = zone.get('zone_low', zone.get('entry_low', 0))
    zone_high = zone.get('zone_high', zone.get('entry_high', 0))
    zone_mid = (zone_low + zone_high) / 2
    
    current_dist = abs(current_price - zone_mid)
    prev_dist = abs(prev_price - zone_mid)
    
    return current_dist < prev_dist

# ============================================================================
# PRICE FORMATTING
# ============================================================================
def format_price(price: float) -> str:
    if price >= 10000:
        return f"${price:,.2f}"
    elif price >= 100:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"

# ============================================================================
# SESSION INFO
# ============================================================================
def get_session_info() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    if 0 <= hour < 8:
        session = 'asia'
        end_hour = 8
    elif 8 <= hour < 14:
        session = 'london'
        end_hour = 14
    elif 14 <= hour < 21:
        session = 'new_york'
        end_hour = 21
    else:
        session = 'asia_close'
        end_hour = 24
    
    hours_remaining = end_hour - hour
    
    return {
        'session': session,
        'hour': hour,
        'hours_remaining': hours_remaining
    }

def get_dynamic_check_interval(volatility: float) -> int:
    from bot.config import CHECK_INTERVAL, CHECK_INTERVAL_HIGH_VOL, CHECK_INTERVAL_MED_VOL, HIGH_VOL_ATR_PCT, MED_VOL_ATR_PCT
    
    if volatility >= HIGH_VOL_ATR_PCT:
        return CHECK_INTERVAL_HIGH_VOL
    elif volatility >= MED_VOL_ATR_PCT:
        return CHECK_INTERVAL_MED_VOL
    else:
        return CHECK_INTERVAL

# ============================================================================
# FACTOR EXTRACTION
# ============================================================================
def extract_factors_from_reason(reason: str) -> List[str]:
    factors = []
    
    factor_keywords = [
        'OB', 'Breaker', 'FVG', 'POC', 'EMA200', 'Liquidity',
        'Volume', 'OI', 'Divergence', 'Sweep', 'BTC align',
        'Uptrend', 'Downtrend', 'Premium', 'Discount',
        'Psychological', 'Support', 'Resistance', 'OTE'
    ]
    
    reason_upper = reason.upper()
    
    for factor in factor_keywords:
        if factor.upper() in reason_upper:
            factors.append(factor)
    
    return factors

# ============================================================================
# ROADMAP UTILITIES
# ============================================================================
def is_roadmap_generation_time() -> bool:
    now = datetime.now(timezone.utc)
    current_hour = now.hour
    current_minute = now.minute
    
    for gen_hour, gen_minute in ROADMAP_GENERATION_TIMES:
        if current_hour == gen_hour and abs(current_minute - gen_minute) <= 5:
            return True
    
    return False

def format_validation_failure(symbol: str, checks: Dict) -> str:
    symbol_short = symbol.replace('/USDT', '')
    
    msg = f"⚠️ **{symbol_short} Roadmap Skipped**\n\n"
    msg += "Failed validation checks:\n"
    
    for check_name, (passed, actual, required) in checks.items():
        status = "✅" if passed else "❌"
        msg += f"{status} {check_name}: {actual:.2f} (need {required})\n"
    
    return msg
