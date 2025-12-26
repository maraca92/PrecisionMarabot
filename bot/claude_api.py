# claude_api.py - Grok Elite Signal Bot v27.12.1 - Claude AI Integration
# -*- coding: utf-8 -*-
"""
v27.12.1: BULLETPROOF ERROR HANDLING

FIXES:
1. CancelledError now caught at EVERY level - no more propagation
2. JSON parsing with multiple fallback extraction methods
3. All exceptions return proper dict, never raise
4. Added extract_json_robust() for better parsing

This fixes:
- "'CancelledError' object has no attribute 'get'" 
- "Claude JSON parse failed" (now extracts from markdown/text)
"""
import logging
import asyncio
import json
import re
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
import anthropic
from anthropic import AsyncAnthropic, APIError, APITimeoutError, RateLimitError

from bot.config import ANTHROPIC_API_KEY, SYMBOLS, CHAT_ID

# ============================================================================
# CONFIGURATION
# ============================================================================
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 1500
CLAUDE_TEMPERATURE = 0.3
CLAUDE_TIMEOUT = 120
CLAUDE_MAX_RETRIES = 2
CLAUDE_RETRY_DELAY = 2
CACHE_DURATION_SECONDS = 300
ADAPTIVE_DELAY_AFTER_529 = 60

# Global async client (singleton)
_async_claude_client: Optional[AsyncAnthropic] = None

# Response cache and error tracking
_analysis_cache: Dict[str, Tuple[Dict, datetime]] = {}
_last_529_error: Optional[datetime] = None


def get_claude_client() -> AsyncAnthropic:
    """Get or create AsyncAnthropic client singleton."""
    global _async_claude_client
    if _async_claude_client is None:
        _async_claude_client = AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=CLAUDE_TIMEOUT
        )
        logging.info("AsyncAnthropic client initialized")
    return _async_claude_client


def _get_cache_key(symbol: str, price: float, zones: List[Dict]) -> str:
    """Generate cache key for response caching."""
    price_bucket = round(price, -2 if price > 1000 else -1 if price > 100 else 2)
    zones_hash = hash(str(sorted([z.get('low', 0) for z in zones[:3]]))) if zones else 0
    return f"{symbol}_{price_bucket}_{zones_hash}"


# ============================================================================
# v27.12.1: ROBUST JSON EXTRACTION
# ============================================================================
def extract_json_robust(content: str) -> Optional[Dict]:
    """
    Extract JSON from Claude's response with multiple fallback methods.
    
    v27.12.1: Handles markdown code blocks, partial JSON, and text wrapping.
    """
    if not content:
        return None
    
    content = content.strip()
    
    # Method 1: Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Extract from ```json ... ``` blocks
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # Method 3: Extract from ``` ... ``` blocks
    if "```" in content:
        try:
            parts = content.split("```")
            if len(parts) >= 2:
                json_str = parts[1].strip()
                # Remove language identifier if present
                if json_str.startswith(('json', 'JSON')):
                    json_str = json_str[4:].strip()
                return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # Method 4: Find JSON object with "trade" or "no_trade" key
    patterns = [
        r'\{[^{}]*"trade"[^{}]*\{[^{}]*\}[^{}]*\}',  # {"trade": {...}}
        r'\{[^{}]*"no_trade"[^{}]*\}',                # {"no_trade": ...}
        r'\{[^{}]*"direction"[^{}]*\}',               # {"direction": ...}
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Method 5: Find first { and last } - greedy extraction
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start >= 0 and end > start:
            json_str = content[start:end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Method 6: Try to fix common issues
    try:
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', content)
        # Find JSON object
        start = fixed.find('{')
        end = fixed.rfind('}')
        if start >= 0 and end > start:
            return json.loads(fixed[start:end + 1])
    except json.JSONDecodeError:
        pass
    
    return None


# ============================================================================
# SYSTEM PROMPT
# ============================================================================
def build_claude_system_prompt_v2(symbol: str, current_price: float, regime: str) -> str:
    """Build system prompt for Claude analysis."""
    return f"""You are an elite ICT/Smart Money Concepts analyst for a high win rate crypto signal bot (target >65% WR).

Current Analysis Context:
- Symbol: {symbol}
- Price: ${current_price:,.2f}
- Market Regime: {regime}

Core Principles:
- Prioritize unmitigated order blocks (strength >2.0), FVG, liquidity sweeps
- Discount zones for longs, premium for shorts
- Require extreme confluence: OB + FVG + sweep + volume + momentum
- Risk management first: SL beyond structure, min RR 1.5:1
- Conservative: 70%+ confidence only for perfect setups
- FADE extreme psychology (buy fear, sell greed)
- Wick reversals + stop hunts = HIGH PROBABILITY setups

Output Format (JSON only, no markdown):
{{
  "trade": {{
    "direction": "Long" | "Short",
    "entry_low": price,
    "entry_high": price,
    "sl": price,
    "tp1": price,
    "tp2": price,
    "leverage": 2-5,
    "confidence": 0-100,
    "strength": 0-5,
    "reason": "Brief explanation"
  }}
}}

OR if no valid setup:
{{
  "no_trade": true,
  "reason": "explanation"
}}

IMPORTANT: Respond with ONLY the JSON object, no additional text or markdown."""


# ============================================================================
# CONTEXT BUILDER
# ============================================================================
def build_claude_context_v2(
    premium_zones: List[Dict],
    symbol: str,
    current_price: float,
    trend: str,
    btc_trend: str,
    oi_data: Optional[Dict] = None,
    funding_rate: Optional[float] = None,
    divergence_data: Optional[Dict] = None,
    funding_data: Optional[Dict] = None,
    momentum_data: Optional[Dict] = None,
    volume_comparison: Optional[Dict] = None,
    structure_break: Optional[Dict] = None,
    fear_greed: Optional[Dict] = None,
    long_short_ratio: Optional[Dict] = None,
    ote_data: Optional[Dict] = None,
    manipulation_signals: Optional[List[Dict]] = None,
    regime: str = "Unknown",
    stop_hunt_result: Optional[Dict] = None,
    fake_breakout_result: Optional[Dict] = None,
    wick_result: Any = None,
    orderbook_data: Optional[Dict] = None,
    vol_data: Optional[Dict] = None,  # v27.12.2: Volatility profile
    reversal_signal: Any = None  # v27.12.14: Early reversal detection
) -> str:
    """Build comprehensive context for Claude analysis."""
    
    # v27.12.11: Sanitize all optional dict parameters - ensure they're actually dicts
    def safe_dict(obj):
        """Return obj if it's a dict, otherwise None"""
        return obj if isinstance(obj, dict) else None
    
    # Sanitize all inputs that might be exceptions
    vol_data = safe_dict(vol_data)
    fear_greed = safe_dict(fear_greed)
    long_short_ratio = safe_dict(long_short_ratio)
    stop_hunt_result = safe_dict(stop_hunt_result)
    fake_breakout_result = safe_dict(fake_breakout_result)
    structure_break = safe_dict(structure_break)
    ote_data = safe_dict(ote_data)
    oi_data = safe_dict(oi_data)
    funding_data = safe_dict(funding_data)
    divergence_data = safe_dict(divergence_data)
    momentum_data = safe_dict(momentum_data)
    volume_comparison = safe_dict(volume_comparison)
    orderbook_data = safe_dict(orderbook_data)
    
    # Sanitize list parameters
    if manipulation_signals is not None and not isinstance(manipulation_signals, list):
        manipulation_signals = None
    if premium_zones is not None and not isinstance(premium_zones, list):
        premium_zones = []
    
    context = f"""# Market Analysis for {symbol}

## Current State
- **Price:** ${current_price:,.4f}
- **Symbol Trend:** {trend}
- **BTC Trend:** {btc_trend}
- **Regime:** {regime}

"""
    
    # v27.12.2: Add volatility profile if available
    if vol_data:
        try:
            from bot.volatility_profile import build_volatility_context
            vol_context = build_volatility_context(symbol, vol_data)
            if vol_context:
                context += vol_context + "\n"
        except ImportError:
            # Fallback: inline simple volatility info
            beta = vol_data.get('beta', 1.0)
            atr_pct = vol_data.get('atr_pct', 4.0)
            vol_regime = vol_data.get('regime', 'MEDIUM')
            context += f"## Volatility Profile\n"
            context += f"- Beta to BTC: {beta:.2f}\n"
            context += f"- ATR%: {atr_pct:.1f}%\n"
            context += f"- Regime: {vol_regime}\n\n"
    
    # Psychology data
    context += "## Market Psychology\n\n"
    
    if fear_greed:
        fg_value = fear_greed.get('value', 50)
        fg_class = fear_greed.get('classification', 'Neutral')
        context += f"- Fear & Greed Index: **{fg_value}** ({fg_class})\n"
        
        if fg_value <= 25:
            context += f"  EXTREME FEAR - Contrarian bullish signal\n"
        elif fg_value >= 75:
            context += f"  EXTREME GREED - Contrarian bearish signal\n"
    
    if long_short_ratio:
        ratio = long_short_ratio.get('ratio', 1.0)
        context += f"- Long/Short Ratio: {ratio:.2f}\n"
        
        if ratio >= 2.0:
            context += f"  CROWDED LONG - Favor Short direction\n"
        elif ratio <= 0.5:
            context += f"  CROWDED SHORT - Favor Long direction\n"
    
    # Wick detection - wrap in try/except for safety
    try:
        if wick_result and hasattr(wick_result, 'detected') and wick_result.detected:
            context += "\n## WICK SIGNAL DETECTED\n\n"
            context += f"- **Type:** {wick_result.wick_type.value if hasattr(wick_result.wick_type, 'value') else wick_result.wick_type}\n"
            context += f"- **Direction:** {wick_result.direction}\n"
            context += f"- **Confidence:** {wick_result.confidence:.0f}%\n"
            context += f"- **Wick Ratio:** {wick_result.wick_ratio:.1f}x\n"
            
            if wick_result.confidence >= 85:
                context += f"\n**HIGH PROBABILITY REVERSAL SIGNAL**\n"
                context += f"   Strongly favor {wick_result.direction} direction.\n"
    except Exception:
        pass  # Skip wick data if any error
    
    # Stop hunt & fake breakout
    if stop_hunt_result:
        context += "\n## STOP HUNT DETECTED\n\n"
        context += f"- Direction: {stop_hunt_result.get('direction', 'N/A')}\n"
        context += f"- Swept Level: {stop_hunt_result.get('swept_level', 'N/A')}\n"
        context += "  High probability reversal after stop hunt.\n"
    
    if fake_breakout_result:
        context += "\n## FAKE BREAKOUT DETECTED\n\n"
        context += f"- Direction: {fake_breakout_result.get('direction', 'N/A')}\n"
        context += "  Failed breakout suggests reversal imminent.\n"
    
    # v27.12.14: Early reversal detection
    if reversal_signal:
        context += "\n## 🔄 EARLY REVERSAL SIGNAL DETECTED\n\n"
        context += f"- **Direction:** {reversal_signal.direction}\n"
        context += f"- **Confidence:** {reversal_signal.confidence:.0f}%\n"
        context += f"- **Confluence Count:** {reversal_signal.confluence_count} signals\n"
        
        if reversal_signal.divergences:
            context += "\n**Divergences:**\n"
            for div in reversal_signal.divergences:
                context += f"  - {div.description}\n"
        
        if reversal_signal.candlestick_patterns:
            context += "\n**Candlestick Patterns:**\n"
            for pattern in reversal_signal.candlestick_patterns:
                context += f"  - {pattern.pattern_name}: {pattern.description}\n"
        
        if reversal_signal.chart_patterns:
            context += "\n**Chart Patterns:**\n"
            for pattern in reversal_signal.chart_patterns:
                context += f"  - {pattern.pattern_name}: {pattern.description}\n"
        
        if reversal_signal.confidence >= 75:
            context += f"\n**HIGH PROBABILITY REVERSAL - Strongly favor {reversal_signal.direction}**\n"
    
    # Structure analysis
    if structure_break:
        context += "\n## Structure Analysis\n\n"
        struct_type = structure_break.get('type', structure_break.get('break_type', ''))
        struct_signal = structure_break.get('signal', '')
        
        context += f"- Structure Break: **{struct_type}**\n"
        context += f"- Signal: {struct_signal}\n"
        
        if struct_type == 'CHoCH':
            context += f"\n**CHANGE OF CHARACTER DETECTED**\n"
            context += f"   High probability trend reversal.\n"
    
    # OTE confluence
    if ote_data and ote_data.get('in_ote'):
        context += "\n## OTE (Optimal Trade Entry)\n\n"
        fib_level = ote_data.get('fib_level', 70)
        context += f"- IN OTE ZONE: {fib_level:.0f}% Fibonacci retracement\n"
        context += f"  Strong entry zone for pullback plays\n"
    
    # Order blocks & zones
    if premium_zones:
        context += "\n## Detected Order Blocks & Zones\n\n"
        
        sorted_zones = sorted(premium_zones, key=lambda z: z.get('prob', z.get('confidence', 0)), reverse=True)
        
        for i, zone in enumerate(sorted_zones[:3], 1):
            zone_low = zone.get('zone_low', zone.get('low', 0))
            zone_high = zone.get('zone_high', zone.get('high', 0))
            zone_mid = (zone_low + zone_high) / 2
            dist = abs(current_price - zone_mid) / current_price * 100 if current_price > 0 else 0
            
            context += f"""### Zone {i}: {zone.get('direction', zone.get('type', 'Unknown'))}
- Price Range: ${zone_low:.4f} - ${zone_high:.4f}
- Distance: {dist:.2f}% from current price
- Strength: {zone.get('strength', zone.get('ob_strength', 0)):.2f}
- Confluence: {zone.get('confluence', 'N/A')}

"""
    
    # Order flow & market structure
    context += "\n## Order Flow & Market Structure\n\n"
    
    if oi_data:
        oi_change = oi_data.get('oi_change_pct', 0)
        context += f"- Open Interest Change: {oi_change:+.2f}%\n"
    
    if funding_data:
        avg_rate = funding_data.get('avg_rate_pct', 0)
        sentiment = funding_data.get('sentiment', 'neutral')
        context += f"- Combined Funding Rate: {avg_rate:.4f}% ({sentiment})\n"
        
        if sentiment in ['extremely_long', 'extremely_short']:
            bias = "bullish" if sentiment == 'extremely_short' else "bearish"
            context += f"  EXTREME funding suggests {bias} reversal opportunity\n"
    elif funding_rate is not None:
        context += f"- Funding Rate: {funding_rate:.4f}%\n"
    
    if divergence_data:
        div_pct = divergence_data.get('divergence_pct', 0)
        signal = divergence_data.get('signal', 'neutral')
        context += f"- Cross-Exchange Divergence: {div_pct:+.3f}% ({signal})\n"
    
    if volume_comparison:
        dominant = volume_comparison.get('dominant', 'balanced')
        context += f"- Volume Dominant: {dominant}\n"
    
    if momentum_data:
        mom_signal = momentum_data.get('signal', 'neutral')
        roc = momentum_data.get('roc', 0)
        context += f"- Momentum: {mom_signal.upper()} (ROC: {roc:+.2f}%)\n"
    
    # Manipulation signals
    if manipulation_signals:
        context += "\n## Institutional Manipulation\n\n"
        for signal in manipulation_signals[:2]:
            context += f"- {signal.get('type', 'Unknown')}: {signal.get('reason', 'N/A')}\n"
    
    # Analysis task
    context += "\n## Analysis Task\n\n"
    context += """Analyze these order blocks with ALL available data and provide a trade setup.

**CRITICAL RULES:**
1. Entry MUST be within 5% of current price
2. TP1 should be 2-2.5% from entry
3. TP2 should be 4-5% from entry
4. SL should be 1-1.5% from entry (structure-based)
5. R:R minimum 1.5:1
6. FADE extreme psychology
7. Wick reversals + stop hunts = HIGH PROBABILITY

Respond with ONLY JSON, no markdown code blocks or additional text.
"""
    
    return context


# ============================================================================
# v27.12.1: MAIN ANALYSIS FUNCTION - BULLETPROOF ERROR HANDLING
# ============================================================================
async def query_claude_analysis(
    premium_zones: List[Dict],
    symbol: str,
    current_price: float,
    trend: str,
    btc_trend: str,
    oi_data: Optional[Dict] = None,
    funding_data: Optional[Dict] = None,
    divergence_data: Optional[Dict] = None,
    momentum_data: Optional[Dict] = None,
    volume_comparison: Optional[Dict] = None,
    structure_break: Optional[Dict] = None,
    fear_greed: Optional[Dict] = None,
    long_short_ratio: Optional[Dict] = None,
    regime: str = "Unknown",
    funding_rate: Optional[float] = None,
    ote_data: Optional[Dict] = None,
    manipulation_signals: Optional[List[Dict]] = None,
    stop_hunt_result: Optional[Dict] = None,
    fake_breakout_result: Optional[Dict] = None,
    wick_result: Any = None,
    orderbook_data: Optional[Dict] = None,
    vol_data: Optional[Dict] = None,  # v27.12.2: Volatility profile
    reversal_signal: Any = None  # v27.12.14: Early reversal detection
) -> Dict[str, Any]:
    """
    Query Claude for trade analysis.
    
    v27.12.1 BULLETPROOF VERSION:
    - NEVER raises exceptions - always returns a dict
    - CancelledError caught at every level
    - Improved JSON extraction with fallbacks
    - All error paths return {"no_trade": True, "reason": "..."}
    """
    global _last_529_error, _analysis_cache
    
    # Wrap EVERYTHING in try/except to guarantee dict return
    try:
        cache_key = _get_cache_key(symbol, current_price, premium_zones)
        now = datetime.now()
        
        # Check cache
        if cache_key in _analysis_cache:
            cached_result, cached_time = _analysis_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_DURATION_SECONDS:
                logging.debug(f"{symbol}: Using cached analysis")
                return cached_result
        
        # Check for recent 529 error
        if _last_529_error:
            elapsed = (now - _last_529_error).total_seconds()
            if elapsed < ADAPTIVE_DELAY_AFTER_529:
                wait_time = ADAPTIVE_DELAY_AFTER_529 - elapsed
                logging.info(f"{symbol}: Recent 529 error, waiting {wait_time:.0f}s")
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    return {"no_trade": True, "reason": "Request cancelled during wait"}
                _last_529_error = None
        
        client = get_claude_client()
        
        # Build context
        context = build_claude_context_v2(
            premium_zones, symbol, current_price, trend, btc_trend,
            oi_data=oi_data,
            funding_rate=funding_rate,
            divergence_data=divergence_data,
            funding_data=funding_data,
            momentum_data=momentum_data,
            volume_comparison=volume_comparison,
            structure_break=structure_break,
            fear_greed=fear_greed,
            long_short_ratio=long_short_ratio,
            ote_data=ote_data,
            manipulation_signals=manipulation_signals,
            regime=regime,
            stop_hunt_result=stop_hunt_result,
            fake_breakout_result=fake_breakout_result,
            wick_result=wick_result,
            orderbook_data=orderbook_data,
            vol_data=vol_data  # v27.12.2: Pass volatility data
        )
        
        system_prompt = build_claude_system_prompt_v2(symbol, current_price, regime)
        
        # Retry loop with bulletproof error handling
        last_error = "Unknown error"
        
        for attempt in range(CLAUDE_MAX_RETRIES + 1):
            try:
                response = await client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    temperature=CLAUDE_TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": context}]
                )
                
                # Extract content
                if not response or not response.content:
                    last_error = "Empty response"
                    continue
                
                content = response.content[0].text.strip()
                
                # v27.12.1: Use robust JSON extraction
                result = extract_json_robust(content)
                
                if result is None:
                    logging.warning(f"{symbol}: Claude JSON parse failed")
                    logging.debug(f"Raw content: {content[:300]}...")
                    return {"no_trade": True, "reason": "Invalid JSON response"}
                
                # Update cache
                _analysis_cache[cache_key] = (result, now)
                
                # Cleanup old cache entries
                if len(_analysis_cache) > 20:
                    oldest_keys = sorted(_analysis_cache.keys(), key=lambda k: _analysis_cache[k][1])[:10]
                    for old_key in oldest_keys:
                        del _analysis_cache[old_key]
                
                # Handle no_trade response
                if result.get("no_trade"):
                    return result
                
                # Validate trade response
                if "trade" in result:
                    trade = result["trade"]
                    required = ["direction", "entry_low", "entry_high", "sl", "tp1", "tp2", "leverage", "confidence"]
                    
                    if all(k in trade for k in required):
                        trade["ai_source"] = "claude"
                        
                        # Add wick signal info if present
                        if wick_result and hasattr(wick_result, 'detected') and wick_result.detected:
                            trade["wick_signal"] = True
                            trade["wick_type"] = str(wick_result.wick_type.value) if hasattr(wick_result.wick_type, 'value') else str(wick_result.wick_type)
                        
                        return result
                    else:
                        missing = [k for k in required if k not in trade]
                        logging.warning(f"{symbol}: Trade missing fields: {missing}")
                        return {"no_trade": True, "reason": f"Missing trade fields: {missing}"}
                
                return {"no_trade": True, "reason": "Invalid response structure"}
            
            # ================================================================
            # v27.12.1: BULLETPROOF EXCEPTION HANDLING
            # ================================================================
            except asyncio.CancelledError:
                # CRITICAL: Never let CancelledError propagate!
                logging.warning(f"{symbol}: Claude call cancelled (attempt {attempt + 1})")
                last_error = "Request cancelled"
                
                if attempt < CLAUDE_MAX_RETRIES:
                    try:
                        delay = CLAUDE_RETRY_DELAY * (2 ** attempt)
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        # Even sleep can be cancelled - just return
                        return {"no_trade": True, "reason": "Request cancelled"}
                    continue
                break
            
            except APITimeoutError:
                logging.warning(f"{symbol}: Claude timeout (attempt {attempt + 1})")
                last_error = "Timeout"
                
                if attempt < CLAUDE_MAX_RETRIES:
                    try:
                        await asyncio.sleep(CLAUDE_RETRY_DELAY * (2 ** attempt))
                    except asyncio.CancelledError:
                        return {"no_trade": True, "reason": "Request cancelled"}
                    continue
                break
            
            except RateLimitError:
                logging.warning(f"{symbol}: Claude rate limited")
                last_error = "Rate limited"
                
                if attempt < CLAUDE_MAX_RETRIES:
                    try:
                        await asyncio.sleep(CLAUDE_RETRY_DELAY * (3 ** attempt))
                    except asyncio.CancelledError:
                        return {"no_trade": True, "reason": "Request cancelled"}
                    continue
                break
            
            except APIError as e:
                error_code = getattr(e, 'status_code', 0)
                logging.error(f"{symbol}: Claude API error {error_code}: {str(e)}")
                
                if error_code == 529:
                    _last_529_error = datetime.now()
                    last_error = "Overloaded (529)"
                else:
                    last_error = f"API error {error_code}"
                
                if attempt < CLAUDE_MAX_RETRIES and error_code in [500, 502, 503, 529]:
                    try:
                        await asyncio.sleep(CLAUDE_RETRY_DELAY * (2 ** attempt))
                    except asyncio.CancelledError:
                        return {"no_trade": True, "reason": "Request cancelled"}
                    continue
                break
            
            except Exception as e:
                # Catch-all for any other exception
                error_msg = str(e) if e else "Unknown error"
                logging.error(f"{symbol}: Claude unexpected error: {error_msg}")
                last_error = error_msg
                break
        
        # All retries failed
        return {"no_trade": True, "reason": last_error}
    
    # ========================================================================
    # v27.12.1: OUTER CATCH-ALL - GUARANTEES DICT RETURN
    # ========================================================================
    except asyncio.CancelledError:
        # Catch any CancelledError that somehow escaped
        logging.error(f"Claude analysis CancelledError escaped for {symbol}")
        return {"no_trade": True, "reason": "Request cancelled"}
    
    except Exception as e:
        # Absolute last resort - should never reach here
        error_msg = str(e) if e else "Unknown fatal error"
        logging.error(f"Claude analysis fatal error: {error_msg}")
        return {"no_trade": True, "reason": error_msg}


# ============================================================================
# HEALTH CHECK - v27.12.1: Bulletproof async version
# ============================================================================
async def claude_health_check() -> bool:
    """Check if Claude API is accessible."""
    try:
        client = get_claude_client()
        
        response = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": "Reply with just 'OK'"}]
        )
        
        return True
    
    except asyncio.CancelledError:
        logging.warning("Claude health check cancelled")
        return False
    except Exception as e:
        logging.error(f"Claude health check failed: {str(e)}")
        return False


# ============================================================================
# DAILY RECAP - v27.12.1: Bulletproof async version
# ============================================================================
async def query_claude_recap(market_summary: str, trades_summary: str) -> str:
    """Get Claude's daily market recap analysis."""
    try:
        client = get_claude_client()
        
        prompt = f"""Provide a brief daily market recap based on this data:

MARKET SUMMARY:
{market_summary}

TRADES SUMMARY:
{trades_summary}

Keep it concise (3-4 paragraphs). Focus on:
1. Key market moves and trends
2. Trade performance highlights
3. Notable patterns or setups
4. Brief outlook for next session"""

        for attempt in range(CLAUDE_MAX_RETRIES + 1):
            try:
                response = await client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=800,
                    temperature=0.5,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                recap_text = response.content[0].text.strip()
                return recap_text
            
            except asyncio.CancelledError:
                if attempt < CLAUDE_MAX_RETRIES:
                    try:
                        await asyncio.sleep(CLAUDE_RETRY_DELAY * (2 ** attempt))
                    except asyncio.CancelledError:
                        return "Recap generation cancelled."
                    continue
                return "Recap generation cancelled."
            
            except (APITimeoutError, RateLimitError):
                if attempt < CLAUDE_MAX_RETRIES:
                    try:
                        await asyncio.sleep(CLAUDE_RETRY_DELAY * (2 ** attempt))
                    except asyncio.CancelledError:
                        return "Recap generation cancelled."
                    continue
                return "Recap generation failed after retries."
        
        return "Recap generation failed after retries."
    
    except asyncio.CancelledError:
        return "Recap generation cancelled."
    except Exception as e:
        logging.error(f"Claude recap error: {str(e)}")
        return "Market recap temporarily unavailable."
