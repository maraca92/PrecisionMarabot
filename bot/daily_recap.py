# daily_recap.py - Grok Elite Signal Bot v27.12.1 - Daily Recap System
# -*- coding: utf-8 -*-
"""
v27.12.1: FIXED EXPORTS FOR MAIN.PY COMPATIBILITY

Key Features:
1. Explicitly titled as PREVIOUS completed day
2. Price action summary: % change, close, highs/lows, volume, BTC dominance
3. Forward bias section: Bullish/Bearish/Neutral with key levels
4. News filtered to previous day only
5. Professional, no-hype, honest tone
6. Graceful fallback if AI generation fails

EXPORTS (required by main.py):
- daily_callback_fixed
- build_previous_day_summary
- fetch_crypto_news_summary
- generate_direct_recap

Scheduled: 00:05 UTC (right after daily candle close)
"""
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp

from bot.config import CHAT_ID, SYMBOLS
from bot.utils import send_throttled, format_price
from bot.data_fetcher import fetch_ohlcv, fetch_ticker_batch

# Try to import optional modules
try:
    from bot.claude_api import query_claude_recap
    CLAUDE_RECAP_AVAILABLE = True
except ImportError:
    CLAUDE_RECAP_AVAILABLE = False

try:
    from bot.grok_api import query_grok_daily_recap
    GROK_RECAP_AVAILABLE = True
except ImportError:
    GROK_RECAP_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================
MAJOR_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
NEWS_API_TIMEOUT = 10
MAX_NEWS_ITEMS = 5


# ============================================================================
# PREVIOUS DAY DATA FETCHER
# ============================================================================
async def get_previous_day_data(symbol: str) -> Dict:
    """
    Get price action data for the PREVIOUS completed day.
    
    Returns:
        Dict with open, high, low, close, volume, change_pct, bullish
    """
    try:
        # Fetch last 3 daily candles to get previous day
        df = await fetch_ohlcv(symbol, '1d', 3)
        
        if len(df) < 2:
            return {}
        
        # Previous day is second-to-last candle
        prev_day = df.iloc[-2]
        day_before = df.iloc[-3] if len(df) >= 3 else df.iloc[-2]
        
        change_pct = ((prev_day['close'] - day_before['close']) / day_before['close']) * 100
        
        return {
            'open': prev_day['open'],
            'high': prev_day['high'],
            'low': prev_day['low'],
            'close': prev_day['close'],
            'volume': prev_day.get('volume', 0),
            'change_pct': change_pct,
            'bullish': prev_day['close'] > prev_day['open']
        }
        
    except Exception as e:
        logging.error(f"Previous day data error for {symbol}: {e}")
        return {}


# ============================================================================
# v27.12.1: REQUIRED BY main.py - build_previous_day_summary
# ============================================================================
async def build_previous_day_summary() -> Dict:
    """
    Build a summary of the previous day's price action.
    REQUIRED BY main.py import.
    
    Returns:
        Dict with BTC, ETH, SOL data and overall market summary
    """
    try:
        btc_data = await get_previous_day_data('BTC/USDT')
        eth_data = await get_previous_day_data('ETH/USDT')
        sol_data = await get_previous_day_data('SOL/USDT')
        
        # Calculate overall market sentiment
        changes = [
            btc_data.get('change_pct', 0),
            eth_data.get('change_pct', 0),
            sol_data.get('change_pct', 0)
        ]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        if avg_change > 2:
            sentiment = 'bullish'
        elif avg_change < -2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'BTC/USDT': btc_data,
            'ETH/USDT': eth_data,
            'SOL/USDT': sol_data,
            'avg_change': avg_change,
            'sentiment': sentiment,
            'date': (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        logging.error(f"build_previous_day_summary error: {e}")
        return {}


# ============================================================================
# v27.12.1: REQUIRED BY main.py - fetch_crypto_news_summary
# ============================================================================
async def fetch_crypto_news_summary() -> List[Dict]:
    """
    Fetch crypto news summary from the previous day.
    REQUIRED BY main.py import. Alias for fetch_previous_day_news.
    
    Returns:
        List of news items with title, source, url
    """
    return await fetch_previous_day_news()


# ============================================================================
# v27.12.1: REQUIRED BY main.py - generate_direct_recap
# ============================================================================
async def generate_direct_recap() -> str:
    """
    Generate a direct recap message without sending it.
    REQUIRED BY main.py import.
    
    Returns:
        Formatted recap message string
    """
    try:
        now = datetime.now(timezone.utc)
        previous_day = now - timedelta(days=1)
        
        # Gather data
        btc_data = await get_previous_day_data('BTC/USDT')
        eth_data = await get_previous_day_data('ETH/USDT')
        sol_data = await get_previous_day_data('SOL/USDT')
        
        btc_dominance = await get_btc_dominance()
        total_mcap = await get_total_market_cap()
        news_items = await fetch_previous_day_news()
        
        price_data = {
            'BTC/USDT': btc_data,
            'ETH/USDT': eth_data,
            'SOL/USDT': sol_data
        }
        
        forward_bias = await calculate_forward_bias(btc_data, eth_data)
        
        # Format message
        message = format_daily_recap_message(
            previous_date=previous_day,
            price_data=price_data,
            forward_bias=forward_bias,
            news_items=news_items,
            btc_dominance=btc_dominance,
            total_mcap=total_mcap
        )
        
        return message
        
    except Exception as e:
        logging.error(f"generate_direct_recap error: {e}")
        return f"Error generating recap: {e}"


# ============================================================================
# BTC DOMINANCE FETCHER
# ============================================================================
async def get_btc_dominance() -> Optional[float]:
    """Fetch BTC dominance from CoinGecko."""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.coingecko.com/api/v3/global"
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('market_cap_percentage', {}).get('btc', None)
    except Exception as e:
        logging.debug(f"BTC dominance fetch error: {e}")
    return None


# ============================================================================
# MARKET CAP FETCHER
# ============================================================================
async def get_total_market_cap() -> Optional[float]:
    """Fetch total crypto market cap from CoinGecko."""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.coingecko.com/api/v3/global"
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    total_mcap = data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
                    return round(total_mcap / 1e9, 2)  # Convert to billions
    except Exception as e:
        logging.debug(f"Market cap fetch error: {e}")
    return None


# ============================================================================
# FORWARD BIAS CALCULATION
# ============================================================================
async def calculate_forward_bias(btc_data: Dict, eth_data: Dict) -> Dict:
    """
    Calculate forward bias for the coming day.
    
    Returns:
        Dict with bias, confidence, key_levels, reasoning
    """
    bias = "Neutral"
    confidence = 50
    reasoning = []
    key_levels = []
    
    if not btc_data:
        return {
            'bias': 'Neutral',
            'confidence': 50,
            'key_levels': [],
            'reasoning': ['Insufficient data for bias calculation']
        }
    
    btc_close = btc_data.get('close', 0)
    btc_change = btc_data.get('change_pct', 0)
    btc_bullish = btc_data.get('bullish', False)
    btc_high = btc_data.get('high', 0)
    btc_low = btc_data.get('low', 0)
    
    # Determine bias based on previous day
    if btc_change > 3:
        bias = "Bullish"
        reasoning.append(f"Strong bullish day (+{btc_change:.1f}%)")
        confidence += 15
    elif btc_change > 1:
        bias = "Cautiously Bullish"
        reasoning.append(f"Mild bullish day (+{btc_change:.1f}%)")
        confidence += 10
    elif btc_change < -3:
        bias = "Bearish"
        reasoning.append(f"Strong bearish day ({btc_change:.1f}%)")
        confidence += 15
    elif btc_change < -1:
        bias = "Cautiously Bearish"
        reasoning.append(f"Mild bearish day ({btc_change:.1f}%)")
        confidence += 10
    else:
        reasoning.append(f"Ranging/consolidation ({btc_change:+.1f}%)")
        bias = "Ranging"
    
    # Add key psychological levels
    psych_levels = [100000, 95000, 90000, 85000, 80000, 75000]
    for level in psych_levels:
        if btc_low * 0.95 <= level <= btc_high * 1.05:
            key_levels.append(('Psychological', level))
            break
    
    # Add previous day's high/low as key levels
    key_levels.append(('Yesterday High', btc_high))
    key_levels.append(('Yesterday Low', btc_low))
    
    # ETH correlation check
    if eth_data:
        eth_change = eth_data.get('change_pct', 0)
        if (btc_change > 0 and eth_change > 0) or (btc_change < 0 and eth_change < 0):
            reasoning.append("ETH confirms BTC direction")
            confidence += 5
        elif abs(btc_change - eth_change) > 3:
            reasoning.append("ETH/BTC divergence - watch for rotation")
    
    # Cap confidence
    confidence = min(confidence, 85)
    
    return {
        'bias': bias,
        'confidence': confidence,
        'key_levels': key_levels[:5],
        'reasoning': reasoning
    }


# ============================================================================
# NEWS FETCHER (PREVIOUS DAY ONLY)
# ============================================================================
async def fetch_previous_day_news() -> List[Dict]:
    """
    Fetch crypto news from the PREVIOUS day only.
    
    Returns:
        List of news items with title, source, url
    """
    news_items = []
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    # Try CryptoPanic API (free tier)
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': 'free',
                'public': 'true',
                'kind': 'news',
                'filter': 'important',
                'currencies': 'BTC,ETH,SOL'
            }
            
            async with session.get(url, params=params, timeout=NEWS_API_TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    
                    for item in results[:10]:
                        published = item.get('published_at', '')
                        if yesterday_str in published:
                            news_items.append({
                                'title': item.get('title', 'No title'),
                                'source': item.get('source', {}).get('title', 'Unknown'),
                                'url': item.get('url', '')
                            })
                            
                            if len(news_items) >= MAX_NEWS_ITEMS:
                                break
    except Exception as e:
        logging.debug(f"CryptoPanic news fetch error: {e}")
    
    if not news_items:
        logging.info("No news from CryptoPanic, using fallback")
    
    return news_items


# ============================================================================
# FORMAT TELEGRAM MESSAGE
# ============================================================================
def format_daily_recap_message(
    previous_date: datetime,
    price_data: Dict[str, Dict],
    forward_bias: Dict,
    news_items: List[Dict],
    btc_dominance: Optional[float],
    total_mcap: Optional[float]
) -> str:
    """Format the complete daily recap message for Telegram."""
    
    # Header
    msg = "\U0001F4CA **Daily Recap** \u2013 "
    msg += f"_{previous_date.strftime('%B %d, %Y')}_\n"
    msg += "\u2501" * 22 + "\n\n"
    
    # Previous Day Summary
    msg += "**Previous Day Summary:**\n"
    
    for symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
        data = price_data.get(symbol, {})
        if not data:
            continue
        
        symbol_short = symbol.replace('/USDT', '')
        change = data.get('change_pct', 0)
        close = data.get('close', 0)
        
        emoji = "\U0001F7E2" if change >= 0 else "\U0001F534"
        
        msg += f"{emoji} **{symbol_short}**: ${close:,.2f} ({change:+.2f}%)\n"
    
    # Market stats
    if btc_dominance:
        msg += f"\U0001F4B0 BTC Dominance: {btc_dominance:.1f}%\n"
    if total_mcap:
        msg += f"\U0001F30D Total MCap: ${total_mcap:.1f}B\n"
    
    msg += "\n"
    
    # Forward Bias
    bias = forward_bias.get('bias', 'Neutral')
    confidence = forward_bias.get('confidence', 50)
    
    bias_emoji = {
        'Bullish': '\U0001F7E2',
        'Cautiously Bullish': '\U0001F7E1',
        'Bearish': '\U0001F534',
        'Cautiously Bearish': '\U0001F7E0',
        'Neutral': '\u26AA',
        'Ranging': '\U0001F7E1'
    }.get(bias, '\u26AA')
    
    msg += "\u2501" * 22 + "\n"
    msg += f"\U0001F52E **Forward Bias**\n"
    msg += "\u2501" * 22 + "\n\n"
    msg += f"{bias_emoji} **{bias}** ({confidence}% confidence)\n\n"
    
    # Reasoning
    reasoning = forward_bias.get('reasoning', [])
    if reasoning:
        msg += "**Why:**\n"
        for reason in reasoning[:3]:
            msg += f"\u2022 {reason}\n"
        msg += "\n"
    
    # Key levels
    key_levels = forward_bias.get('key_levels', [])
    if key_levels:
        msg += "**Key Levels to Watch:**\n"
        for level_type, level_price in key_levels[:4]:
            msg += f"\u2022 {level_type}: ${level_price:,.0f}\n"
        msg += "\n"
    
    # News
    if news_items:
        msg += "\u2501" * 22 + "\n"
        msg += "\U0001F4F0 **Key News**\n"
        msg += "\u2501" * 22 + "\n\n"
        
        for i, news in enumerate(news_items[:3], 1):
            title = news.get('title', 'No title')
            source = news.get('source', 'Unknown')
            if len(title) > 80:
                title = title[:77] + "..."
            msg += f"{i}. {title}\n"
            msg += f"   _via {source}_\n"
        msg += "\n"
    
    # Footer
    msg += "\u2501" * 22 + "\n"
    msg += "_Trade safe. Quality > Quantity._\n"
    
    return msg


# ============================================================================
# MAIN DAILY RECAP CALLBACK
# ============================================================================
async def daily_callback_fixed(context, query_grok_func=None):
    """
    Main daily recap callback - runs at 00:05 UTC.
    """
    try:
        logging.info("=" * 50)
        logging.info("=== Starting Daily Recap v27.12.1 ===")
        logging.info("=" * 50)
        
        now = datetime.now(timezone.utc)
        previous_day = now - timedelta(days=1)
        
        logging.info(f"Generating recap for: {previous_day.strftime('%Y-%m-%d')}")
        
        # Gather data in parallel
        tasks = [
            get_previous_day_data('BTC/USDT'),
            get_previous_day_data('ETH/USDT'),
            get_previous_day_data('SOL/USDT'),
            get_btc_dominance(),
            get_total_market_cap(),
            fetch_previous_day_news()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract results
        btc_data = results[0] if not isinstance(results[0], Exception) else {}
        eth_data = results[1] if not isinstance(results[1], Exception) else {}
        sol_data = results[2] if not isinstance(results[2], Exception) else {}
        btc_dominance = results[3] if not isinstance(results[3], Exception) else None
        total_mcap = results[4] if not isinstance(results[4], Exception) else None
        news_items = results[5] if not isinstance(results[5], Exception) else []
        
        price_data = {
            'BTC/USDT': btc_data,
            'ETH/USDT': eth_data,
            'SOL/USDT': sol_data
        }
        
        logging.info(f"Data gathered: BTC={bool(btc_data)}, ETH={bool(eth_data)}, News={len(news_items)}")
        
        # Calculate forward bias
        forward_bias = await calculate_forward_bias(btc_data, eth_data)
        logging.info(f"Forward bias: {forward_bias.get('bias')} ({forward_bias.get('confidence')}%)")
        
        # Format and send message
        message = format_daily_recap_message(
            previous_date=previous_day,
            price_data=price_data,
            forward_bias=forward_bias,
            news_items=news_items,
            btc_dominance=btc_dominance,
            total_mcap=total_mcap
        )
        
        await send_throttled(CHAT_ID, message, parse_mode='Markdown')
        logging.info("Daily recap sent successfully")
        
        # Optional: AI-Enhanced Recap
        if query_grok_func and GROK_RECAP_AVAILABLE:
            try:
                market_summary = f"""
Previous Day ({previous_day.strftime('%Y-%m-%d')}):
- BTC: ${btc_data.get('close', 0):,.0f} ({btc_data.get('change_pct', 0):+.2f}%)
- ETH: ${eth_data.get('close', 0):,.0f} ({eth_data.get('change_pct', 0):+.2f}%)
- BTC Dominance: {btc_dominance or 'N/A'}%
- Forward Bias: {forward_bias.get('bias')} ({forward_bias.get('confidence')}%)
"""
                
                ai_recap = await query_grok_func(market_summary, "")
                
                if ai_recap and len(ai_recap) > 50:
                    ai_msg = f"\U0001F916 **AI Market Insight**\n\n{ai_recap}"
                    await asyncio.sleep(2)
                    await send_throttled(CHAT_ID, ai_msg, parse_mode='Markdown')
                    
            except Exception as e:
                logging.warning(f"AI recap failed: {e}")
        
    except Exception as e:
        logging.error(f"Daily recap error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Fallback: Minimal recap
        try:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            fallback_msg = f"\U0001F4CA **Daily Recap \u2013 {yesterday.strftime('%B %d, %Y')}**\n\n"
            fallback_msg += "_Detailed recap unavailable. Check markets directly._\n\n"
            
            prices = await fetch_ticker_batch()
            for symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
                price = prices.get(symbol)
                if price:
                    symbol_short = symbol.replace('/USDT', '')
                    fallback_msg += f"\u2022 {symbol_short}: ${price:,.2f}\n"
            
            await send_throttled(CHAT_ID, fallback_msg, parse_mode='Markdown')
            
        except Exception as fallback_error:
            logging.error(f"Fallback recap also failed: {fallback_error}")


# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================
async def daily_callback(context):
    """Legacy callback - redirects to fixed version."""
    await daily_callback_fixed(context, query_grok_daily_recap if GROK_RECAP_AVAILABLE else None)


async def send_daily_recap():
    """Standalone function to send daily recap."""
    await daily_callback_fixed(None, None)
