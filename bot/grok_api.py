# grok_api.py - Grok Elite Signal Bot v27.12.3 - Grok AI Integration
"""
Grok AI integration with opinion layer for signals AND roadmaps.

v27.12.3 CHANGES:
1. Added get_grok_roadmap_opinion() for roadmap zone validation
2. Added get_grok_market_sentiment() for broader market view
3. Improved error handling and fallbacks
4. Better display formatting for opinions
"""
import logging
import json
from typing import Dict, Any, List, Optional
import httpx

# Safe imports with fallbacks
try:
    from bot.config import XAI_API_KEY
except ImportError:
    import os
    XAI_API_KEY = os.getenv('XAI_API_KEY', '')

try:
    from bot.config import GROK_TEMPERATURE, GROK_VALIDATION_MAX_TOKENS, GROK_RECAP_MAX_TOKENS
except ImportError:
    GROK_TEMPERATURE = 0.3
    GROK_VALIDATION_MAX_TOKENS = 300
    GROK_RECAP_MAX_TOKENS = 1500

try:
    from bot.config import GROK_OPINION_ENABLED, GROK_OPINION_TIMEOUT, GROK_ROADMAP_OPINION_ENABLED
except ImportError:
    GROK_OPINION_ENABLED = True
    GROK_OPINION_TIMEOUT = 12.0
    GROK_ROADMAP_OPINION_ENABLED = True

try:
    from bot.utils import check_precision, check_rr, calibrate_grok_confidence
except ImportError:
    def check_precision(trade):
        return True
    def check_rr(trade):
        return True
    def calibrate_grok_confidence(conf, factors):
        return conf

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
GROK_MODELS = [
    "grok-3",           # Primary - most reliable
    "grok-3-mini",      # Fallback - faster, cheaper
]

_working_model = None


# ============================================================================
# v27.9.5: GROK OPINION LAYER FOR LIVE SIGNALS
# ============================================================================

async def get_grok_opinion(
    trade: Dict,
    symbol: str, 
    price: float,
    trend: str,
    btc_trend: Optional[str] = None,
    timeout: float = None
) -> Dict[str, Any]:
    """
    Get Grok's quick opinion on Claude's trade.
    DISPLAY ONLY - doesn't block trades, just shows agree/disagree.
    
    Returns:
        {
            'opinion': 'agree'|'disagree'|'neutral',
            'reason': str,
            'display': str (formatted for Telegram)
        }
    """
    global _working_model
    
    if not GROK_OPINION_ENABLED or not XAI_API_KEY:
        return {'opinion': 'neutral', 'reason': 'Disabled', 'display': ''}
    
    timeout = timeout or GROK_OPINION_TIMEOUT
    
    try:
        direction = trade.get('direction', 'Long')
        entry_low = trade.get('entry_low', 0)
        entry_high = trade.get('entry_high', 0)
        sl = trade.get('sl', 0)
        tp1 = trade.get('tp1', 0)
        confidence = trade.get('confidence', 0)
        reason = trade.get('reason', '')
        
        is_alt = symbol != 'BTC/USDT'
        btc_context = f" BTC trend: {btc_trend}." if is_alt and btc_trend else ""
        
        prompt = f"""Claude suggests {direction} on {symbol} at ${price:.2f}.
Entry: ${entry_low:.4f}-${entry_high:.4f}, SL: ${sl:.4f}, TP1: ${tp1:.4f}
Confidence: {confidence}%, Trend: {trend}.{btc_context}
Reason: {reason}

Quick assessment: Any obvious risks? Sentiment concern? Momentum issue?"""
        
        models = [_working_model] + GROK_MODELS if _working_model else GROK_MODELS
        
        for model in models:
            if model is None:
                continue
                
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You're a fast market validator. Reply ONLY with JSON: {\"opinion\": \"agree\"/\"disagree\"/\"neutral\", \"reason\": \"one sentence\"}. Be concise."
                                },
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.2,
                            "max_tokens": 100
                        },
                        headers={"Authorization": f"Bearer {XAI_API_KEY}"}
                    )
                    response.raise_for_status()
                    
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    
                    # Clean up JSON
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()
                    
                    result = json.loads(content)
                    opinion = result.get('opinion', 'neutral')
                    reason_text = result.get('reason', '')
                    
                    # Format display string
                    if opinion == 'agree':
                        display = f"🤖 Grok: ✅ {reason_text}"
                    elif opinion == 'disagree':
                        display = f"🤖 Grok: ⚠️ {reason_text}"
                    else:
                        display = f"🤖 Grok: ➖ {reason_text}"
                    
                    _working_model = model
                    return {
                        'opinion': opinion,
                        'reason': reason_text,
                        'display': display
                    }
                    
            except httpx.HTTPStatusError as e:
                logging.error(f"Grok opinion HTTP error with {model}: {e.response.status_code}")
                continue
            except json.JSONDecodeError:
                logging.error(f"Grok opinion: Failed to parse JSON from {model}")
                continue
            except Exception as e:
                logging.error(f"Grok opinion error with {model}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Grok opinion error: {e}")
    
    return {'opinion': 'neutral', 'reason': 'API unavailable', 'display': ''}


# ============================================================================
# v27.12.3: GROK OPINION FOR ROADMAP ZONES
# ============================================================================

async def get_grok_roadmap_opinion(
    zone: Dict,
    symbol: str,
    price: float,
    btc_trend: str,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Get Grok's opinion on a roadmap zone.
    Used for roadmap generation and display.
    
    Returns:
        {
            'opinion': 'bullish'|'bearish'|'neutral',
            'confidence_adj': int (-10 to +10),
            'reason': str,
            'display': str (formatted for Telegram)
        }
    """
    global _working_model
    
    if not GROK_ROADMAP_OPINION_ENABLED or not XAI_API_KEY:
        return {'opinion': 'neutral', 'confidence_adj': 0, 'reason': 'Disabled', 'display': ''}
    
    try:
        direction = zone.get('direction', 'Long')
        zone_low = zone.get('zone_low', zone.get('entry_low', 0))
        zone_high = zone.get('zone_high', zone.get('entry_high', 0))
        zone_mid = (zone_low + zone_high) / 2
        dist_pct = abs(price - zone_mid) / price * 100 if price > 0 else 0
        ob_strength = zone.get('ob_strength', zone.get('strength', 1.5))
        confluence = zone.get('confluence', '')
        zone_type = zone.get('type', 'trend')
        
        is_alt = symbol != 'BTC/USDT'
        btc_context = f"BTC is {btc_trend}." if is_alt else ""
        
        prompt = f"""Roadmap zone for {symbol}:
- Direction: {direction}
- Zone: ${zone_low:.4f} - ${zone_high:.4f} ({dist_pct:.1f}% from current ${price:.2f})
- OB Strength: {ob_strength:.1f}
- Type: {zone_type}
- Confluence: {confluence}
{btc_context}

Quick take: Is this zone likely to trigger? Any concerns?"""
        
        models = [_working_model] + GROK_MODELS if _working_model else GROK_MODELS
        
        for model in models:
            if model is None:
                continue
                
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": """You analyze crypto roadmap zones. Reply ONLY with JSON:
{"opinion": "bullish"/"bearish"/"neutral", "confidence_adj": -10 to +10, "reason": "one sentence"}
- bullish: zone looks promising
- bearish: zone has issues/unlikely to work
- neutral: uncertain
confidence_adj: how much to adjust zone confidence (-10 to +10)"""
                                },
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.2,
                            "max_tokens": 120
                        },
                        headers={"Authorization": f"Bearer {XAI_API_KEY}"}
                    )
                    response.raise_for_status()
                    
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()
                    
                    result = json.loads(content)
                    opinion = result.get('opinion', 'neutral')
                    conf_adj = max(-10, min(10, result.get('confidence_adj', 0)))
                    reason_text = result.get('reason', '')
                    
                    # Format display
                    if opinion == 'bullish':
                        emoji = "🟢"
                    elif opinion == 'bearish':
                        emoji = "🔴"
                    else:
                        emoji = "⚪"
                    
                    adj_str = f"+{conf_adj}" if conf_adj > 0 else str(conf_adj) if conf_adj < 0 else "±0"
                    display = f"🤖 Grok: {emoji} {reason_text} ({adj_str}%)"
                    
                    _working_model = model
                    return {
                        'opinion': opinion,
                        'confidence_adj': conf_adj,
                        'reason': reason_text,
                        'display': display
                    }
                    
            except Exception as e:
                logging.debug(f"Grok roadmap opinion error with {model}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Grok roadmap opinion error: {e}")
    
    return {'opinion': 'neutral', 'confidence_adj': 0, 'reason': 'N/A', 'display': ''}


# ============================================================================
# v27.12.3: GROK MARKET SENTIMENT
# ============================================================================

async def get_grok_market_sentiment(
    btc_price: float,
    btc_change_24h: float,
    fear_greed: Optional[int] = None,
    timeout: float = 15.0
) -> Dict[str, Any]:
    """
    Get Grok's overall market sentiment analysis.
    Used in daily recap and market overview.
    
    Returns:
        {
            'sentiment': 'bullish'|'bearish'|'neutral',
            'outlook': str,
            'key_levels': str,
            'display': str
        }
    """
    global _working_model
    
    if not XAI_API_KEY:
        return {'sentiment': 'neutral', 'outlook': 'N/A', 'key_levels': '', 'display': ''}
    
    try:
        fg_context = f"Fear & Greed: {fear_greed}." if fear_greed else ""
        
        prompt = f"""BTC at ${btc_price:,.0f}, 24h change: {btc_change_24h:+.1f}%.
{fg_context}

Brief market sentiment (1-2 sentences) and key levels to watch."""
        
        models = [_working_model] + GROK_MODELS if _working_model else GROK_MODELS
        
        for model in models:
            if model is None:
                continue
                
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": """Crypto market analyst. Reply JSON only:
{"sentiment": "bullish"/"bearish"/"neutral", "outlook": "1-2 sentences", "key_levels": "support/resistance"}"""
                                },
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.2,
                            "max_tokens": 200
                        },
                        headers={"Authorization": f"Bearer {XAI_API_KEY}"}
                    )
                    response.raise_for_status()
                    
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()
                    
                    result = json.loads(content)
                    sentiment = result.get('sentiment', 'neutral')
                    outlook = result.get('outlook', '')
                    key_levels = result.get('key_levels', '')
                    
                    emoji = "🟢" if sentiment == 'bullish' else "🔴" if sentiment == 'bearish' else "⚪"
                    display = f"🤖 Grok Market View: {emoji} {outlook}"
                    
                    _working_model = model
                    return {
                        'sentiment': sentiment,
                        'outlook': outlook,
                        'key_levels': key_levels,
                        'display': display
                    }
                    
            except Exception as e:
                logging.debug(f"Grok sentiment error with {model}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Grok sentiment error: {e}")
    
    return {'sentiment': 'neutral', 'outlook': 'N/A', 'key_levels': '', 'display': ''}


# ============================================================================
# GROK VALIDATION FOR CLAUDE'S TRADES
# ============================================================================

async def validate_grok_trade_idea(
    trade: Dict,
    symbol: str,
    current_price: float,
    trend: str,
    btc_trend: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate Claude's trade suggestion with Grok.
    Used in dual AI consensus mode.
    
    Returns:
        {
            'approved': bool,
            'no_trade': bool,
            'reason': str
        }
    """
    global _working_model
    
    if not XAI_API_KEY:
        return {"approved": True, "reason": "Grok unavailable - defaulting to approve"}
    
    is_alt = symbol != 'BTC/USDT'
    
    # Build context
    direction = trade.get('direction', 'Long')
    entry_low = trade.get('entry_low', 0)
    entry_high = trade.get('entry_high', 0)
    sl = trade.get('sl', 0)
    tp1 = trade.get('tp1', 0)
    confidence = trade.get('confidence', 0)
    reason = trade.get('reason', '')
    
    context = f"""{symbol} @ ${current_price:.2f} | Trend: {trend}
Claude suggests: {direction}
Entry: ${entry_low:.4f} - ${entry_high:.4f}
SL: ${sl:.4f} | TP1: ${tp1:.4f}
Confidence: {confidence}%
Reason: {reason}

Any obvious concerns? Sentiment concern? Momentum issue?"""
    
    system_prompt = """You're a fast market validator. Claude (primary analyst) suggested a trade.

Your job: Quick check for:
1. Obvious risks Claude might've missed
2. Sentiment/momentum red flags
3. Setup quality gut-check

Respond ONLY:
{"approved": true, "reason": "brief confirmation"} 
OR
{"approved": false, "no_trade": true, "reason": "brief concern"}

Be concise. If no major issues: approve."""
    
    if is_alt:
        system_prompt += "\nAlts: Decline if conflicts with BTC trend."
    
    models = [_working_model] + GROK_MODELS if _working_model else GROK_MODELS
    
    for model in models:
        if model is None:
            continue
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "temperature": GROK_TEMPERATURE,
            "max_tokens": GROK_VALIDATION_MAX_TOKENS
        }
        
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {XAI_API_KEY}"}
                )
                r.raise_for_status()
                
                content = r.json()["choices"][0]["message"]["content"].strip()
                
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                
                result = json.loads(content)
                _working_model = model
                
                return result
                
        except httpx.HTTPStatusError as e:
            logging.error(f"Grok validation HTTP error with {model}: {e.response.status_code}")
            continue
        except json.JSONDecodeError:
            logging.error(f"Grok validation parse error with {model}")
            continue
        except Exception as e:
            logging.error(f"Grok validation error with {model}: {e}")
            continue
    
    return {"approved": True, "reason": "Grok unavailable - defaulting to approve"}


# ============================================================================
# GROK POTENTIAL TRADE QUERY
# ============================================================================

async def query_grok_potential(
    premium_zones: List[Dict],
    symbol: str,
    current_price: float,
    trend: str,
    btc_trend: Optional[str],
    oi_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """Query Grok for potential trade analysis."""
    global _working_model
    
    if not premium_zones:
        return {"no_live_trade": True, "roadmap": []}
    
    is_alt = symbol != 'BTC/USDT'
    
    context = f"{symbol} @ ${current_price:.2f} | Trend: {trend}"
    if is_alt and btc_trend:
        context += f" | BTC: {btc_trend}"
    context += "\n\nOrder Blocks:\n"
    
    for z in sorted(premium_zones, key=lambda x: x.get('prob', x.get('confidence', 0)), reverse=True)[:3]:
        mid = (z['zone_low'] + z['zone_high']) / 2
        dist = abs(current_price - mid) / current_price * 100
        context += f"- {z['direction']}: ${z['zone_low']:.4f}-${z['zone_high']:.4f} "
        context += f"(strength {z.get('strength', 0):.1f}, {dist:.1f}% away, {z.get('confluence', 'N/A')})\n"
    
    if oi_data:
        context += f"\nOI Change: {oi_data.get('oi_change_pct', 0):+.2f}%"
    
    system_prompt = (
        "You are a crypto analyst. Analyze the order blocks and suggest a trade if high-probability setup exists.\n\n"
        "Respond ONLY in JSON:\n"
        '{"live_trade": {"direction": "Long/Short", "entry_low": X, "entry_high": X, "sl": X, "tp1": X, "tp2": X, "confidence": 0-100, "reason": "brief"}, "roadmap": []}\n'
        "OR if no good setup:\n"
        '{"no_live_trade": true, "reason": "why", "roadmap": []}\n\n'
        f"Minimum confidence: 65%. {'For alts: respect BTC trend direction.' if is_alt else ''}"
    )
    
    models = [_working_model] + GROK_MODELS if _working_model else GROK_MODELS
    
    for model in models:
        if model is None:
            continue
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "temperature": GROK_TEMPERATURE,
            "max_tokens": 400
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {XAI_API_KEY}"}
                )
                r.raise_for_status()
                
                content = r.json()["choices"][0]["message"]["content"].strip()
                
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                
                result = json.loads(content)
                
                precise = True
                if 'live_trade' in result:
                    if not check_precision(result['live_trade']) or not check_rr(result['live_trade']):
                        precise = False
                
                if precise:
                    if 'live_trade' in result and 'confidence' in result['live_trade']:
                        factors = {
                            'liq_sweep': 'liq' in result['live_trade'].get('reason', '').lower(),
                            'vol_surge': 'vol' in result['live_trade'].get('reason', '').lower(),
                            'htf_align': 'htf' in result['live_trade'].get('reason', '').lower(),
                            'oi_spike': 'oi' in result['live_trade'].get('reason', '').lower()
                        }
                        result['live_trade']['confidence'] = calibrate_grok_confidence(
                            result['live_trade']['confidence'], factors
                        )
                    
                    _working_model = model
                    return result
                
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logging.error(f"Grok potential query error with {model}: {e}")
            continue
    
    return {"no_live_trade": True, "roadmap": []}


# ============================================================================
# GROK DAILY RECAP
# ============================================================================

async def query_grok_daily_recap(market_summary: str) -> Optional[str]:
    """Generate daily market recap using Grok."""
    global _working_model
    
    if not XAI_API_KEY:
        logging.warning("Grok API key not configured for daily recap")
        return None
    
    system_prompt = "Concise market recap focusing on momentum and sentiment. Max 150 words."
    
    full_context = market_summary + "\n\nQuick institutional recap + next 24h bias."
    
    models = [_working_model] + GROK_MODELS if _working_model else GROK_MODELS
    
    for model in models:
        if model is None:
            continue
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_context}
            ],
            "temperature": 0.2,
            "max_tokens": GROK_RECAP_MAX_TOKENS
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {XAI_API_KEY}"}
                )
                r.raise_for_status()
                
                recap = r.json()["choices"][0]["message"]["content"].strip()
                _working_model = model
                
                logging.info(f"Daily recap generated using model: {model}")
                return recap
                
        except httpx.HTTPStatusError as e:
            logging.error(f"Daily recap HTTP error with {model}: {e.response.status_code}")
            continue
        except Exception as e:
            logging.error(f"Daily recap error with {model}: {e}")
            continue
    
    logging.error("All Grok models failed for daily recap")
    return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_grok_context(zones: List[Dict], symbol: str, price: float, trend: str, btc_trend: Optional[str]) -> str:
    """Build context string for Grok queries."""
    ranked = sorted(zones, key=lambda z: z.get('prob', z.get('confidence', 0)), reverse=True)[:2]
    context = f"{symbol} | ${price:.2f} | {trend}"
    
    if btc_trend and symbol != 'BTC/USDT':
        context += f" | BTC: {btc_trend}"
    
    context += "\n"
    
    for z in ranked:
        dist = abs(price - (z['zone_low'] + z['zone_high'])/2) / price * 100
        context += f"{z['direction']}: {z['zone_low']:.4f}-{z['zone_high']:.4f} "
        context += f"(str{z.get('strength', 0):.1f}, {dist:.1f}% away, {z.get('confluence', 'N/A')})\n"
    
    return context


def get_working_model() -> Optional[str]:
    """Get the currently working Grok model."""
    return _working_model


def grok_opinion_confidence_adjust(opinion: str) -> int:
    """Get confidence adjustment based on Grok opinion."""
    if opinion == 'agree':
        return 5
    elif opinion == 'disagree':
        return -10
    return 0


def get_grok_opinion_boost(opinion: str) -> int:
    """Alias for grok_opinion_confidence_adjust."""
    return grok_opinion_confidence_adjust(opinion)
