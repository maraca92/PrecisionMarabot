# roadmap.py - Grok Elite Signal Bot v27.12.13 - Dual Roadmap System
# -*- coding: utf-8 -*-
"""
v27.12.13: CRITICAL FIX - DIRECTION LOGIC & WHY SECTION

CHANGES:
1. FIXED: Direction logic was INVERTED for short trades!
   - For LONG: price should be ABOVE zone (waiting to drop to it)
   - For SHORT: price should be BELOW zone (waiting to rise to it)
   - Previously SHORT logic was backwards
2. FIXED: Why section now shows actual confluence data instead of generic text
3. Added HTF trend and regime info to Why section

v27.12.11: CRITICAL FIX - ROADMAP TO LIVE TRADE CONVERSION

CHANGES:
1. FIXED: monitor_roadmap_proximity() now calls convert_roadmap_to_live()
2. FIXED: Roadmap zones are now added to open_trades when triggered
3. Added print statements for Render log visibility
4. Improved logging throughout

v27.12.10: ROADMAP DISTANCE FILTER
- Added ROADMAP_MAX_DISTANCE_PCT import (7% max distance)
- Added filter_zones_by_distance() function

v27.12.3: GROK OPINION INTEGRATION FOR ROADMAPS
"""
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from bot.config import SYMBOLS, CHAT_ID

# Safe config imports with fallbacks
try:
    from bot.config import (
        TREND_ROADMAP_MAX_ZONES, TREND_ROADMAP_MIN_OB_STRENGTH,
        TREND_ROADMAP_MIN_CONFIDENCE, ROADMAP_ENTRY_PROXIMITY_PCT,
        STRUCTURAL_MAX_ZONES, STRUCTURAL_BOUNCE_ENABLED,
        ROADMAP_MIN_VOL_SURGE, ROADMAP_MIN_DEPTH_BTC, ROADMAP_MIN_DEPTH_ALT,
        RELAXED_MAX_ZONES_TREND, RELAXED_MAX_ZONES_STRUCTURAL,
        RELAXED_MIN_OB_STRENGTH, RELAXED_MIN_CONFIDENCE,
        GROK_OPINION_ENABLED, GROK_ROADMAP_OPINION_ENABLED
    )
except ImportError:
    TREND_ROADMAP_MAX_ZONES = 10
    TREND_ROADMAP_MIN_OB_STRENGTH = 1.8
    TREND_ROADMAP_MIN_CONFIDENCE = 60
    ROADMAP_ENTRY_PROXIMITY_PCT = 0.5
    STRUCTURAL_MAX_ZONES = 5
    STRUCTURAL_BOUNCE_ENABLED = True
    ROADMAP_MIN_VOL_SURGE = 1.2
    ROADMAP_MIN_DEPTH_BTC = 300000
    ROADMAP_MIN_DEPTH_ALT = 150000
    RELAXED_MAX_ZONES_TREND = 5
    RELAXED_MAX_ZONES_STRUCTURAL = 2
    RELAXED_MIN_OB_STRENGTH = 1.5
    RELAXED_MIN_CONFIDENCE = 55
    GROK_OPINION_ENABLED = True
    GROK_ROADMAP_OPINION_ENABLED = True

# v27.12.10: Import max distance config
try:
    from bot.config import ROADMAP_MAX_DISTANCE_PCT
except ImportError:
    ROADMAP_MAX_DISTANCE_PCT = 7.0

from bot.utils import send_throttled, format_price, calculate_zone_proximity
from bot.models import (
    load_roadmap_zones, save_roadmap_zones_async, clear_expired_roadmap_zones,
    load_trades, save_trades_async  # v27.12.11: Added for live trade conversion
)

from bot.data_fetcher import fetch_ohlcv, fetch_ticker_batch
from bot.indicators import add_institutional_indicators, detect_market_regime
from bot.order_blocks import find_unmitigated_order_blocks

# v27.12.3: Import Grok opinion functions
try:
    from bot.grok_api import get_grok_roadmap_opinion
except ImportError:
    async def get_grok_roadmap_opinion(*args, **kwargs):
        return {'opinion': 'neutral', 'confidence_adj': 0, 'reason': '', 'display': ''}

# Safe structural bounce import
try:
    from bot.structural_bounce import detect_structural_bounces_batch
except ImportError:
    async def detect_structural_bounces_batch(*args):
        return []

# ============================================================================
# CONSTANTS
# ============================================================================
ROADMAP_CONVERSION_TRIGGER_PCT = 1.0
ROADMAP_ALERT_PROXIMITY_PCT = 2.5
ROADMAP_ALERT_COOLDOWN_MINUTES = 120

# ============================================================================
# GLOBAL STATE
# ============================================================================
roadmap_zones: Dict[str, List[Dict]] = {}
data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}


# ============================================================================
# v27.12.10: FILTER ZONES BY DISTANCE
# ============================================================================
def filter_zones_by_distance(zones: List[Dict], prices: Dict, max_distance_pct: float = None) -> List[Dict]:
    """Filter zones to only include those within max_distance_pct of current price."""
    if max_distance_pct is None:
        max_distance_pct = ROADMAP_MAX_DISTANCE_PCT
    
    filtered = []
    rejected_count = 0
    
    for zone in zones:
        symbol = zone.get('symbol')
        price = prices.get(symbol, 0)
        
        if price <= 0:
            continue
        
        zone_low = zone.get('zone_low', zone.get('entry_low', 0))
        zone_high = zone.get('zone_high', zone.get('entry_high', 0))
        zone_mid = (zone_low + zone_high) / 2
        
        dist_pct = abs(price - zone_mid) / price * 100
        
        if dist_pct <= max_distance_pct:
            zone['dist_pct'] = dist_pct
            filtered.append(zone)
        else:
            rejected_count += 1
            logging.debug(f"{symbol}: Zone rejected - {dist_pct:.1f}% > {max_distance_pct}% max distance")
    
    if rejected_count > 0:
        logging.info(f"Distance filter: {rejected_count} zones rejected (>{max_distance_pct}%)")
    
    return filtered


# ============================================================================
# VALIDATION - REQUIRED BY main.py
# ============================================================================
def validate_roadmap_for_conversion(symbol: str, price: float, zone: Dict, 
                                     data_cache: Dict, stats: Dict) -> Dict:
    """Validate if roadmap zone should convert to live trade."""
    try:
        if zone.get('converted'):
            return {'valid': False, 'reason': "Already converted"}
        
        prox = calculate_zone_proximity(zone, price)
        
        if prox['inside'] or prox['dist_pct'] <= ROADMAP_CONVERSION_TRIGGER_PCT:
            return {'valid': True, 'reason': f"Price at {prox['dist_pct']:.2f}% from zone"}
        
        return {'valid': False, 'reason': f"Not close enough ({prox['dist_pct']:.2f}%)"}
        
    except Exception as e:
        logging.error(f"Validation error: {e}")
        return {'valid': False, 'reason': str(e)}


# ============================================================================
# PROXIMITY ALERT - DISABLED
# ============================================================================
async def send_proximity_alert(alert_data: Dict):
    """DISABLED - proximity alerts were spam."""
    return


# ============================================================================
# CONVERSION ALERT - WITH GROK OPINION
# ============================================================================
async def send_conversion_alert(symbol: str, zone: Dict, price: float, prox: Dict):
    """Send alert when roadmap zone converts to active signal."""
    try:
        symbol_short = symbol.replace('/USDT', '')
        dir_emoji = "🟢" if zone['direction'] == 'Long' else "🔴"
        
        if zone.get('type') == 'structural_bounce':
            type_label = "STRUCTURAL"
        else:
            type_label = "TREND"
        
        ob_strength = zone.get('ob_strength', zone.get('strength', 0))
        
        msg = f"🚨 **ROADMAP ACTIVATED** ({type_label})\n\n"
        msg += f"{dir_emoji} **{symbol_short} {zone['direction']}** ({zone['confidence']}%)\n"
        msg += f"📍 Current: **{format_price(price)}**\n"
        msg += f"🎯 Entry: {format_price(zone['entry_low'])} - {format_price(zone['entry_high'])}\n"
        msg += f"📊 OB Strength: {ob_strength:.1f}\n"
        msg += f"🛑 SL: {format_price(zone['sl'])}\n"
        msg += f"✅ TP1: {format_price(zone['tp1'])}\n"
        msg += f"🎯 TP2: {format_price(zone['tp2'])}\n\n"
        msg += f"💡 Why: {zone.get('confluence', 'N/A')}"
        
        # v27.12.3: Add Grok opinion if available
        grok_display = zone.get('grok_display', '')
        if grok_display:
            msg += f"\n\n{grok_display}"
        
        # v27.12.11: Add live trade confirmation
        msg += "\n\n✅ **Trade added to tracking system**"
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Conversion alert error: {e}")


# ============================================================================
# CONVERT ROADMAP TO LIVE - v27.12.11 FIXED
# ============================================================================
async def convert_roadmap_to_live(symbol: str, zone: Dict, open_trades: Dict, protected_trades: Dict) -> bool:
    """
    Convert a roadmap zone to a live trade entry.
    v27.12.11: This is now actually called from monitor_roadmap_proximity()
    """
    try:
        # Check if symbol already has a trade
        if symbol in open_trades or symbol in protected_trades:
            logging.info(f"{symbol}: Already has active trade, skipping conversion")
            print(f"[ROADMAP] {symbol}: Skipped - already has active trade", flush=True)
            return False
        
        # Check for roadmap-based trade key
        trade_key = f"{symbol}_roadmap"
        if trade_key in open_trades:
            logging.info(f"{symbol}: Already has roadmap trade, skipping")
            return False
        
        # Create the trade entry
        trade = {
            'symbol': symbol,
            'direction': zone['direction'],
            'entry_low': zone['entry_low'],
            'entry_high': zone['entry_high'],
            'entry_price': (zone['entry_low'] + zone['entry_high']) / 2,
            'sl': zone['sl'],
            'tp1': zone['tp1'],
            'tp2': zone['tp2'],
            'confidence': zone['confidence'],
            'confluence': zone.get('confluence', ''),
            'ob_strength': zone.get('ob_strength', 0),
            'timeframe': zone.get('timeframe', '4h'),
            'from_roadmap': True,
            'roadmap_type': zone.get('type', 'trend'),
            'entry_time': datetime.now(timezone.utc),
            'last_check': datetime.now(timezone.utc),
            'active': False,  # Will become active when price enters zone
            'processed': False,
            'tp1_exited': False,
            'trailing_sl': None,
            'use_tp2': True,
            'grade': 'B',  # Default grade for roadmap trades
            'grade_score': zone['confidence'],
            'grok_opinion': zone.get('grok_opinion', 'neutral'),
            'factors': ['OB', 'Roadmap', 'HTF Confluence']
        }
        
        open_trades[trade_key] = trade
        
        logging.info(f"{symbol}: ✅ Converted roadmap zone to LIVE TRADE")
        print(f"[ROADMAP] {symbol}: ✅ Converted to LIVE TRADE - {zone['direction']}", flush=True)
        
        return True
        
    except Exception as e:
        logging.error(f"Roadmap conversion error: {e}")
        print(f"[ROADMAP] {symbol}: ❌ Conversion error: {e}", flush=True)
        return False


# ============================================================================
# PROXIMITY MONITORING - v27.12.11 FIXED WITH LIVE CONVERSION
# ============================================================================
async def monitor_roadmap_proximity():
    """
    Monitor price proximity to roadmap zones.
    v27.12.11: FIXED - Now actually converts zones to live trades!
    """
    global roadmap_zones
    
    if not roadmap_zones:
        return
    
    prices = await fetch_ticker_batch()
    now = datetime.now(timezone.utc)
    
    # v27.12.11: Load current trades to check for duplicates and save new ones
    open_trades = load_trades()
    protected_trades = {}  # Load if you have protected trades persistence
    
    conversions = 0
    checked = 0
    
    for symbol, zones in list(roadmap_zones.items()):
        price = prices.get(symbol)
        if price is None:
            continue
        
        for zone in zones:
            if zone.get('converted'):
                continue
            
            checked += 1
            prox = calculate_zone_proximity(zone, price)
            
            # Check if price is inside or very close to zone
            if prox['inside'] or prox['dist_pct'] <= ROADMAP_CONVERSION_TRIGGER_PCT:
                logging.info(f"{symbol}: Roadmap zone conversion triggered (dist: {prox['dist_pct']:.2f}%)")
                print(f"[ROADMAP] {symbol}: Zone triggered at {prox['dist_pct']:.2f}% distance", flush=True)
                
                # Mark zone as converted
                zone['converted'] = True
                zone['converted_at'] = now
                await save_roadmap_zones_async(roadmap_zones)
                
                # v27.12.11: CRITICAL FIX - Actually convert to live trade!
                converted = await convert_roadmap_to_live(symbol, zone, open_trades, protected_trades)
                
                if converted:
                    # Save the updated trades
                    await save_trades_async(open_trades)
                    conversions += 1
                    
                    # Send alert AFTER successful conversion
                    await send_conversion_alert(symbol, zone, price, prox)
                else:
                    # Still send alert but note it wasn't added as trade
                    logging.warning(f"{symbol}: Roadmap alert sent but trade not added (may already exist)")
    
    if conversions > 0:
        logging.info(f"Roadmap monitor: {conversions} zones converted to live trades")
        print(f"[ROADMAP] Monitor complete: {conversions} zones converted to live trades", flush=True)
    elif checked > 0:
        logging.debug(f"Roadmap monitor: Checked {checked} zones, no conversions")


# ============================================================================
# v27.12.3: ROADMAP BATCH WITH GROK OPINIONS
# ============================================================================
async def send_roadmap_batch(zones: List[Dict], zone_type: str, prices: Dict, btc_trend: str = "Unknown"):
    """Send formatted batch of roadmap zones WITH Grok opinions."""
    if not zones:
        return
    
    type_emoji = "📈" if "TREND" in zone_type else "🎯"
    msg = f"{type_emoji} **{zone_type} ROADMAP** ({len(zones)} zones)\n"
    msg += f"_Generated: {datetime.now(timezone.utc).strftime('%H:%M UTC')}_\n\n"
    
    for zone in zones:
        symbol = zone['symbol']
        symbol_short = symbol.replace('/USDT', '')
        price = prices.get(symbol, 0)
        
        zone_low = zone.get('zone_low', zone.get('entry_low', 0))
        zone_high = zone.get('zone_high', zone.get('entry_high', 0))
        zone_mid = (zone_low + zone_high) / 2
        dist_pct = abs(price - zone_mid) / price * 100 if price > 0 else 0
        
        dir_emoji = "🟢" if zone['direction'] == 'Long' else "🔴"
        ob_strength = zone.get('ob_strength', zone.get('strength', 1.5))
        ob_label = f"OB {ob_strength:.1f}"
        
        # v27.12.13: Build detailed why text from actual confluence data
        why_parts = []
        
        # Add confluence from zone if available
        confluence = zone.get('confluence', '')
        if confluence:
            why_parts.append(confluence)
        else:
            why_parts.append(f"Strong {zone['direction'].lower()} zone")
        
        # Add distance info
        why_parts.append(f"{dist_pct:.1f}% from current price")
        
        # Add HTF trend context
        htf_trend = zone.get('htf_trend', '')
        if htf_trend:
            if htf_trend.lower() == 'bullish' and zone['direction'] == 'Long':
                why_parts.append("HTF trend aligned (bullish)")
            elif htf_trend.lower() == 'bearish' and zone['direction'] == 'Short':
                why_parts.append("HTF trend aligned (bearish)")
            else:
                why_parts.append(f"HTF: {htf_trend}")
        
        # Add regime context
        regime = zone.get('regime', '')
        if regime:
            why_parts.append(f"Market: {regime}")
        
        # Add timeframe
        tf = zone.get('timeframe', '4h')
        why_parts.append(f"Source: {tf.upper()}")
        
        why_text = " | ".join(why_parts)
        
        # v27.12.3: Get Grok opinion for this zone
        grok_display = zone.get('grok_display', '')
        if not grok_display and GROK_ROADMAP_OPINION_ENABLED:
            try:
                grok_result = await get_grok_roadmap_opinion(zone, symbol, price, btc_trend)
                grok_display = grok_result.get('display', '')
                zone['grok_opinion'] = grok_result.get('opinion', 'neutral')
                zone['grok_display'] = grok_display
            except Exception as e:
                logging.debug(f"Grok roadmap opinion error for {symbol}: {e}")
        
        msg += f"\n{dir_emoji} **{symbol_short} {zone['direction']}** ({zone['confidence']}%)\n"
        msg += f"📍 Current: **{format_price(price)}**\n"
        msg += f"🎯 Entry: {format_price(zone['entry_low'])} - {format_price(zone['entry_high'])} ({dist_pct:.1f}% away)\n"
        msg += f"🛑 SL: {format_price(zone['sl'])} | ✅ TP1: {format_price(zone['tp1'])} | 🎯 TP2: {format_price(zone['tp2'])}\n"
        msg += f"📊 {ob_label}\n"
        msg += f"💡 Why: {why_text}\n"
        
        # Add Grok opinion if available
        if grok_display:
            msg += f"{grok_display}\n"
    
    await send_throttled(CHAT_ID, msg, parse_mode='Markdown')


# ============================================================================
# TREND ROADMAP ZONE GENERATION - v27.12.10 Updated
# ============================================================================
async def generate_trend_roadmap_zones(
    symbol: str,
    df_1d: pd.DataFrame,
    df_4h: pd.DataFrame,
    price: float,
    btc_trend: str
) -> List[Dict]:
    """Generate trend-following roadmap zones for a symbol."""
    zones = []
    
    try:
        # Detect market regime
        regime = detect_market_regime(df_1d) if len(df_1d) > 0 else 'unknown'
        
        # Get HTF trend
        htf_trend = 'bullish'
        if len(df_1d) >= 20:
            ema20 = df_1d['close'].rolling(20).mean().iloc[-1]
            if price > ema20:
                htf_trend = 'bullish'
            else:
                htf_trend = 'bearish'
        
        # Find order blocks from 4h timeframe
        all_obs = []
        
        if len(df_4h) >= 50:
            obs_4h = await find_unmitigated_order_blocks(df_4h, lookback=50, tf='4h')
            for ob_type in ['bullish', 'bearish']:
                for ob in obs_4h.get(ob_type, []):
                    ob['source_tf'] = '4h'
                    all_obs.append(ob)
        
        if len(df_1d) >= 30:
            obs_1d = await find_unmitigated_order_blocks(df_1d, lookback=30, tf='1d')
            for ob_type in ['bullish', 'bearish']:
                for ob in obs_1d.get(ob_type, []):
                    ob['source_tf'] = '1d'
                    all_obs.append(ob)
        
        # Sort by strength
        all_obs = sorted(all_obs, key=lambda x: x.get('strength', 0), reverse=True)
        
        # Calculate ATR for SL/TP
        atr = df_4h['high'].rolling(14).max().iloc[-1] - df_4h['low'].rolling(14).min().iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = price * 0.02
        
        rejected = {'distance': 0, 'strength': 0, 'position': 0, 'confidence': 0}
        
        for ob in all_obs[:10]:
            ob_mid = (ob['high'] + ob['low']) / 2
            dist_pct = abs(price - ob_mid) / price * 100
            
            # Apply distance filter
            if dist_pct > ROADMAP_MAX_DISTANCE_PCT:
                rejected['distance'] += 1
                continue
            
            strength = ob.get('strength', 1.5)
            if strength < RELAXED_MIN_OB_STRENGTH:
                rejected['strength'] += 1
                continue
            
            # Determine direction based on OB type and position relative to price
            # v27.12.13: CRITICAL FIX - Direction logic was inverted for shorts!
            #
            # For LONG trades (bullish OB):
            #   - Price should be ABOVE the zone (price has dropped to the zone)
            #   - We wait for price to touch the zone from above and bounce up
            #   - So: price > ob_mid means we're positioned correctly to go long
            #
            # For SHORT trades (bearish OB):
            #   - Price should be BELOW the zone (price will rise to the zone)
            #   - We wait for price to touch the zone from below and reject down
            #   - So: price < ob_mid means we're positioned correctly to go short
            #
            is_bullish_ob = ob.get('type', '').lower() == 'bullish' or 'bullish' in str(ob).lower()
            
            if is_bullish_ob:
                # Bullish OB = support zone = go LONG when price drops to it
                if price < ob_mid:
                    # Price already below the zone - we missed the entry
                    rejected['position'] += 1
                    continue
                direction = 'Long'
            else:
                # Bearish OB = resistance zone = go SHORT when price rises to it
                if price > ob_mid:
                    # Price already above the zone - we missed the entry
                    rejected['position'] += 1
                    continue
                direction = 'Short'
            
            # Calculate confidence
            confidence = 55 + int(strength * 8) + (5 if ob.get('source_tf') == '1d' else 0)
            confidence = min(90, max(55, confidence))
            
            if confidence < RELAXED_MIN_CONFIDENCE:
                rejected['confidence'] += 1
                continue
            
            # Create zone
            zone = {
                'symbol': symbol,
                'direction': direction,
                'type': 'trend',
                'zone_low': ob['low'],
                'zone_high': ob['high'],
                'entry_low': ob['low'],
                'entry_high': ob['high'],
                'confidence': confidence,
                'strength': strength,
                'ob_strength': strength,
                'timeframe': ob.get('source_tf', '4h'),
                'created_at': datetime.now(timezone.utc),
                'converted': False,
                'htf_trend': htf_trend,
                'regime': regime
            }
            
            # Calculate SL/TP
            entry_mid = (ob['low'] + ob['high']) / 2
            
            if direction == 'Long':
                zone['sl'] = ob['low'] - (atr * 0.3)
                zone['tp1'] = entry_mid + (atr * 1.5)
                zone['tp2'] = entry_mid + (atr * 3.0)
            else:
                zone['sl'] = ob['high'] + (atr * 0.3)
                zone['tp1'] = entry_mid - (atr * 1.5)
                zone['tp2'] = entry_mid - (atr * 3.0)
            
            # Build confluence string
            confluence_parts = [f"Strong OB ({strength:.1f})"]
            confluence_parts.append(f"{ob.get('source_tf', '4h').upper()} TF")
            confluence_parts.append(f"HTF {htf_trend}")
            if dist_pct < 3:
                confluence_parts.append(f"Near zone ({dist_pct:.1f}%)")
            else:
                confluence_parts.append(f"Close ({dist_pct:.1f}%)")
            if ob.get('type') == 'Breaker':
                confluence_parts.append("Breaker")
            
            zone['confluence'] = " + ".join(confluence_parts)
            
            zones.append(zone)
        
        logging.info(f"{symbol}: Rejected OBs - Distance:{rejected['distance']} Strength:{rejected['strength']} Position:{rejected['position']} Confidence:{rejected['confidence']}")
        logging.info(f"{symbol}: Generated {len(zones)} roadmap zones from {len(all_obs)} OBs (max dist: {ROADMAP_MAX_DISTANCE_PCT}%)")
        
    except Exception as e:
        logging.error(f"{symbol}: Trend zone generation error: {e}")
        import traceback
        logging.debug(traceback.format_exc())
    
    return zones


# ============================================================================
# MAIN ROADMAP GENERATION CALLBACK - v27.12.10 Updated
# ============================================================================
async def roadmap_generation_callback(data_cache: Dict, btc_trend: str):
    """Generate roadmaps for all symbols."""
    global roadmap_zones
    
    print(f"[ROADMAP] Starting roadmap generation...", flush=True)
    
    prices = await fetch_ticker_batch()
    
    if not prices:
        logging.error("Could not fetch prices for roadmap generation")
        print(f"[ROADMAP] ERROR: Could not fetch prices", flush=True)
        return
    
    trend_zones_all = []
    
    for symbol in SYMBOLS:
        try:
            price = prices.get(symbol, 0)
            if price <= 0:
                continue
            
            symbol_cache = data_cache.get(symbol, {})
            df_1d = symbol_cache.get('1d', pd.DataFrame())
            df_4h = symbol_cache.get('4h', pd.DataFrame())
            
            if len(df_1d) < 50:
                df_1d = await fetch_ohlcv(symbol, '1d', 200)
                if len(df_1d) > 0:
                    df_1d = add_institutional_indicators(df_1d)
            
            if len(df_4h) < 50:
                df_4h = await fetch_ohlcv(symbol, '4h', 200)
                if len(df_4h) > 0:
                    df_4h = add_institutional_indicators(df_4h)
            
            if len(df_1d) < 50:
                continue
            
            zones = await generate_trend_roadmap_zones(symbol, df_1d, df_4h, price, btc_trend)
            
            if zones:
                trend_zones_all.extend(zones)
                logging.info(f"{symbol}: Found {len(zones)} trend zones")
            
            await asyncio.sleep(0.3)
            
        except Exception as e:
            logging.error(f"{symbol}: Trend roadmap error: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            continue
    
    # v27.12.10: Filter zones by distance BEFORE sorting and selection
    trend_zones_all = filter_zones_by_distance(trend_zones_all, prices, ROADMAP_MAX_DISTANCE_PCT)
    
    # Sort and select top zones
    trend_zones_all = sorted(trend_zones_all, key=lambda z: z['confidence'], reverse=True)
    
    selected_trend_zones = []
    symbol_counts = {}
    
    for zone in trend_zones_all:
        symbol = zone['symbol']
        count = symbol_counts.get(symbol, 0)
        
        if count >= 2:
            continue
        
        selected_trend_zones.append(zone)
        symbol_counts[symbol] = count + 1
        
        if len(selected_trend_zones) >= RELAXED_MAX_ZONES_TREND:
            break
    
    logging.info(f"Found {len(trend_zones_all)} total trend zones, selected {len(selected_trend_zones)} (max {RELAXED_MAX_ZONES_TREND})")
    
    # Generate structural roadmap
    structural_zones_all = []
    
    if STRUCTURAL_BOUNCE_ENABLED:
        try:
            structural_zones_all = await detect_structural_bounces_batch(data_cache, prices, btc_trend)
            logging.info(f"Structural bounce detection complete: {len(structural_zones_all)} zones found")
        except Exception as e:
            logging.error(f"Structural bounce detection error: {e}")
    
    # v27.12.10: Filter structural zones by distance as well
    structural_zones_all = filter_zones_by_distance(structural_zones_all, prices, ROADMAP_MAX_DISTANCE_PCT)
    
    logging.info(f"Found {len(structural_zones_all)} structural zones after {ROADMAP_MAX_DISTANCE_PCT}% filter")
    
    selected_structural_zones = structural_zones_all[:RELAXED_MAX_ZONES_STRUCTURAL]
    
    # Combine and save
    all_selected_zones = selected_trend_zones + selected_structural_zones
    
    new_zones = {}
    for zone in all_selected_zones:
        symbol = zone['symbol']
        if symbol not in new_zones:
            new_zones[symbol] = []
        new_zones[symbol].append(zone)
    
    new_zones = clear_expired_roadmap_zones(new_zones, max_age_hours=48)
    
    roadmap_zones = new_zones
    await save_roadmap_zones_async(roadmap_zones)
    
    total_zones = len(all_selected_zones)
    trend_count = len(selected_trend_zones)
    structural_count = len(selected_structural_zones)
    
    logging.info(f"Roadmap generation complete: {total_zones} zones ({trend_count} trend + {structural_count} structural) within {ROADMAP_MAX_DISTANCE_PCT}%")
    print(f"[ROADMAP] Generation complete: {total_zones} zones ({trend_count} trend + {structural_count} structural)", flush=True)
    
    # Send formatted messages with Grok opinions
    if total_zones > 0:
        if trend_count > 0:
            await send_roadmap_batch(selected_trend_zones, "TREND-FOLLOWING", prices, btc_trend)
            await asyncio.sleep(1)
        
        if structural_count > 0:
            await send_roadmap_batch(selected_structural_zones, "STRUCTURAL BOUNCE", prices, btc_trend)
    else:
        await send_throttled(
            CHAT_ID,
            f"⚠️ **No roadmap zones generated**\n_No zones found within {ROADMAP_MAX_DISTANCE_PCT}% of current prices_",
            parse_mode='Markdown'
        )


# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_roadmaps():
    """Load roadmaps from disk on startup."""
    global roadmap_zones
    roadmap_zones = load_roadmap_zones()
    roadmap_zones = clear_expired_roadmap_zones(roadmap_zones)
    
    total_zones = sum(len(zl) for zl in roadmap_zones.values())
    trend_count = sum(1 for zones in roadmap_zones.values() for z in zones if z.get('type') != 'structural_bounce')
    structural_count = sum(1 for zones in roadmap_zones.values() for z in zones if z.get('type') == 'structural_bounce')
    
    logging.info(f"Initialized roadmaps: {total_zones} zones ({trend_count} trend + {structural_count} structural)")
    print(f"[ROADMAP] Initialized: {total_zones} zones loaded", flush=True)


def get_roadmap_zones() -> Dict[str, List[Dict]]:
    """Get current roadmap zones."""
    return roadmap_zones
