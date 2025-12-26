# telegram_commands.py - Grok Elite Signal Bot v27.12.10 - Telegram Commands
# -*- coding: utf-8 -*-
"""
Telegram bot commands:
- /stats - Bot performance statistics
- /health - System health check
- /recap - Manual daily recap
- /backtest - Run improved backtest with roadmap simulation
- /backtest_all - Multi-symbol backtest
- /validate - Monte Carlo validation
- /dashboard - Detailed performance metrics
- /roadmap - Show current roadmap zones (both types)
- /structural - Show only structural bounce zones
- /zones - Alias for /roadmap
- /commands - Show all available commands

v27.12.10: Updated /commands and /health to show 7% distance filter
v27.6.0: Added /structural command, updated /roadmap to show both types
"""
import logging
import os
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import aiofiles
from telegram import Update
from telegram.ext import ContextTypes

from bot.config import (
    CHAT_ID, SYMBOLS, SIMULATED_CAPITAL, BACKTEST_FILE,
    TRADE_LOG_FILE, BOT_VERSION, PAPER_TRADING,
    ROADMAP_MIN_OB_STRENGTH, FEE_PCT, STRUCTURAL_TP1_PCT,
    STRUCTURAL_TP2_PCT, STRUCTURAL_EXPECTED_WIN_RATE,
    STRUCTURAL_EXPECTED_AVG_BOUNCE,
    RELAXED_MAX_ZONES_TREND, RELAXED_MAX_ZONES_STRUCTURAL
)

# v27.12.10: Import max distance config
try:
    from bot.config import ROADMAP_MAX_DISTANCE_PCT
except ImportError:
    ROADMAP_MAX_DISTANCE_PCT = 7.0
from bot.models import load_stats, save_stats_async, HISTORICAL_DATA
from bot.utils import get_clean_symbol, send_throttled, format_price
from bot.data_fetcher import fetch_ohlcv, fetch_ticker_batch
from bot.grok_api import query_grok_daily_recap

open_trades = {}
protected_trades = {}

def set_trade_dicts(ot: Dict, pt: Dict):
    """Called by main.py to share trade dicts"""
    global open_trades, protected_trades
    open_trades = ot
    protected_trades = pt

# ============================================================================
# /commands COMMAND
# ============================================================================
async def commands_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display all available commands"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    commands = """**📋 Available Commands**

**Trading & Stats:**
- `/stats` - Performance statistics
- `/factors` - Factor performance analysis

**Roadmap (v27.12.10):**
- `/roadmap` or `/zones` - View all roadmap zones
- `/structural` - View only structural bounce zones
- `/genroadmap` - Force roadmap generation

**Market Analysis (v27.12.15):**
- `/liquidity [SYM]` - Liquidity heatmap (e.g. /liquidity BTC)
- `/orderbook [SYM]` - Orderbook imbalance analysis
- `/market` - Market overview
- `/health` - Bot health check
- `/system_health` - Detailed system status

**Backtesting (v27.12.17):**
- `/backtest_real [SYM] [DAYS]` - ⭐ REALISTIC backtest
- `/backtest_pro [SYM] [DAYS]` - Quick backtest (less realistic)
- `/backtest` - Legacy 90-day backtest
- `/validate` - Monte Carlo validation
- `/dashboard` - Detailed metrics

**Other:**
- `/recap` - Daily market recap
- `/force` - Force signal check
- `/commands` - Show this list

**💡 v27.12.17:** Realistic backtest with confirmation, fill probability, costs"""
    
    await send_throttled(CHAT_ID, commands, parse_mode='Markdown')

# ============================================================================
# /stats COMMAND
# ============================================================================
async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display bot performance statistics"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        stats = load_stats()
        total = stats['wins'] + stats['losses']
        winrate = (stats['wins'] / total * 100) if total > 0 else 0
        
        active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        pending = len([t for t in open_trades.values() if not t.get('active')])
        
        unique_symbols = sorted(set(
            get_clean_symbol(k).replace('/USDT', '')
            for trades in [open_trades, protected_trades]
            for k in trades.keys()
        ))
        open_symbols = ', '.join(unique_symbols) if unique_symbols else 'None'
        
        tp1_hits = stats.get('tp1_hits', 0)
        tp1_rate = (tp1_hits / total * 100) if total > 0 else 0
        avg_pnl = stats['pnl'] / total if total > 0 else 0
        
        roadmap_conv = stats.get('roadmap_conversions', 0)
        roadmap_skip = stats.get('roadmap_skips', 0)
        
        msg = (
            f"**Bot Statistics v{BOT_VERSION}**\n\n"
            f"**Total Trades:** {total}\n"
            f"**Wins:** {stats['wins']} ({winrate:.1f}%)\n"
            f"**Losses:** {stats['losses']}\n"
            f"**TP1 Hits:** {tp1_hits} ({tp1_rate:.1f}%)\n\n"
            f"**Total PNL:** {stats['pnl']:+.2f}%\n"
            f"**Capital:** ${stats['capital']:.2f}\n"
            f"**Avg PnL/Trade:** {avg_pnl:+.2f}%\n"
            f"**Best:** +{stats.get('best_trade', 0):.2f}%\n"
            f"**Worst:** {stats.get('worst_trade', 0):+.2f}%\n\n"
            f"**Active:** {active} | **Pending:** {pending}\n"
            f"**Open:** {open_symbols}\n\n"
            f"**Roadmap:** {roadmap_conv} converted | {roadmap_skip} skipped"
        )
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"/stats error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')

# ============================================================================
# /health COMMAND
# ============================================================================
async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display bot health status"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        from bot.roadmap import roadmap_zones
        
        uptime = datetime.now(timezone.utc).isoformat()
        open_count = len(open_trades)
        protected_count = len(protected_trades)
        active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        pending = len([t for t in open_trades.values() if not t.get('active')])
        
        roadmap_count = sum(len(zones) for zones in roadmap_zones.values())
        trend_count = sum(1 for zones in roadmap_zones.values() for z in zones if z.get('type') != 'structural_bounce')
        structural_count = sum(1 for zones in roadmap_zones.values() for z in zones if z.get('type') == 'structural_bounce')
        
        msg = (
            f"**Grok Elite Bot v{BOT_VERSION} - Alive!**\n\n"
            f"**MODE**: {'📋 PAPER' if PAPER_TRADING else '💰 LIVE'}\n"
            f"**Uptime Check:** {uptime}\n"
            f"**Open Trades:** {open_count}\n"
            f"**Protected Trades:** {protected_count}\n"
            f"**Active:** {active} | **Pending:** {pending}\n"
            f"**Roadmap Zones:** {roadmap_count} ({trend_count} trend + {structural_count} structural)\n"
            f"**Zone Limits:** {RELAXED_MAX_ZONES_TREND} trend + {RELAXED_MAX_ZONES_STRUCTURAL} structural within {ROADMAP_MAX_DISTANCE_PCT}%\n"
            f"**Status:** All systems operational ✅"
        )
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"/health error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')

# ============================================================================
# /roadmap COMMAND (Updated for v27.6.0)
# ============================================================================
async def roadmap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display current roadmap zones (both types)"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        from bot.roadmap import roadmap_zones
        
        total_zones = sum(len(zones) for zones in roadmap_zones.values())
        
        if total_zones == 0:
            await send_throttled(CHAT_ID, "No roadmap zones active. Wait for next generation (00:05 or 15:00 UTC).", parse_mode='Markdown')
            return
        
        # Separate by type
        trend_zones = []
        structural_zones = []
        
        for symbol, zones in roadmap_zones.items():
            for zone in zones:
                if zone.get('converted'):
                    continue
                
                if zone.get('type') == 'structural_bounce':
                    structural_zones.append((symbol, zone))
                else:
                    trend_zones.append((symbol, zone))
        
        prices = await fetch_ticker_batch()
        
        msg = f"📋 **Active Roadmap Zones** ({total_zones})\n\n"
        
        # ====================================================================
        # TREND-FOLLOWING ZONES
        # ====================================================================
        if trend_zones:
            msg += f"✅ **TREND-FOLLOWING** ({len(trend_zones)}):\n\n"
            
            for symbol, zone in trend_zones:
                symbol_short = symbol.replace('/USDT', '')
                price = prices.get(symbol, 0)
                
                zone_mid = (zone['zone_low'] + zone['zone_high']) / 2
                dist_pct = abs(price - zone_mid) / price * 100 if price > 0 else 0
                
                bt_str = f" | Est: {zone.get('backtest_pnl', 0):+.1f}R" if zone.get('backtest_pnl') else ""
                
                status = "🟢" if dist_pct < 1.0 else "🟡" if dist_pct < 3.0 else "⚪"
                
                msg += f"{status} **{symbol_short} {zone['direction']}** ({zone['confidence']}%){bt_str}\n"
                msg += f"   Zone: {format_price(zone['zone_low'])} - {format_price(zone['zone_high'])}\n"
                msg += f"   Dist: {dist_pct:.1f}% | {zone['confluence']}\n"
                msg += f"   SL: {format_price(zone['sl'])} | TP1: {format_price(zone['tp1'])} | TP2: {format_price(zone['tp2'])}\n\n"
        
        # ====================================================================
        # STRUCTURAL BOUNCE ZONES
        # ====================================================================
        if structural_zones:
            msg += f"🎯 **STRUCTURAL BOUNCE** ({len(structural_zones)}):\n\n"
            
            for symbol, zone in structural_zones:
                symbol_short = symbol.replace('/USDT', '')
                price = prices.get(symbol, 0)
                
                zone_mid = (zone['zone_low'] + zone['zone_high']) / 2
                dist_pct = abs(price - zone_mid) / price * 100 if price > 0 else 0
                
                status = "🟢" if dist_pct < 1.0 else "🟡" if dist_pct < 3.0 else "⚪"
                
                counter_tag = " ⚡" if zone.get('is_counter_trend') else ""
                psych_level = zone.get('psychological_level', 0)
                
                msg += f"{status} **{symbol_short} {zone['direction']}**{counter_tag} ({zone['confidence']}%)\n"
                msg += f"   Level: ${psych_level:,} psychological\n"
                msg += f"   Zone: {format_price(zone['zone_low'])} - {format_price(zone['zone_high'])}\n"
                msg += f"   OB: {zone.get('ob_strength', 0):.1f} | Dist: {dist_pct:.1f}%\n"
                msg += f"   Target: +{STRUCTURAL_TP1_PCT:.1f}% (TP1) / +{STRUCTURAL_TP2_PCT:.1f}% (TP2)\n"
                msg += f"   {zone['confluence']}\n\n"
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"/roadmap error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')

# Alias for roadmap
async def zones_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Alias for /roadmap"""
    await roadmap_cmd(update, context)

# ============================================================================
# /structural COMMAND (NEW v27.6.0)
# ============================================================================
async def structural_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display only structural bounce zones"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        from bot.roadmap import roadmap_zones
        
        structural_zones = []
        
        for symbol, zones in roadmap_zones.items():
            for zone in zones:
                if zone.get('type') == 'structural_bounce' and not zone.get('converted'):
                    structural_zones.append((symbol, zone))
        
        if not structural_zones:
            await send_throttled(
                CHAT_ID, 
                "No structural bounce zones active.\n\n"
                f"Expected Win Rate: {STRUCTURAL_EXPECTED_WIN_RATE:.1f}%\n"
                f"Expected Bounce: +{STRUCTURAL_EXPECTED_AVG_BOUNCE:.1f}%",
                parse_mode='Markdown'
            )
            return
        
        prices = await fetch_ticker_batch()
        
        msg = f"🎯 **Structural Bounce Zones** ({len(structural_zones)})\n"
        msg += f"*Expected WR: {STRUCTURAL_EXPECTED_WIN_RATE:.1f}% | Avg Bounce: +{STRUCTURAL_EXPECTED_AVG_BOUNCE:.1f}%*\n\n"
        
        for symbol, zone in structural_zones:
            symbol_short = symbol.replace('/USDT', '')
            price = prices.get(symbol, 0)
            
            zone_mid = (zone['zone_low'] + zone['zone_high']) / 2
            dist_pct = abs(price - zone_mid) / price * 100 if price > 0 else 0
            
            psych_level = zone.get('psychological_level', 0)
            counter_tag = " ⚡COUNTER-TREND" if zone.get('is_counter_trend') else ""
            
            msg += f"**{symbol_short} {zone['direction']}**{counter_tag}\n"
            msg += f"   Level: ${psych_level:,}\n"
            msg += f"   Current: ${price:,.2f} ({dist_pct:.1f}% away)\n"
            msg += f"   OB Strength: {zone.get('ob_strength', 0):.1f}\n"
            msg += f"   Confidence: {zone['confidence']}%\n"
            msg += f"   Entry: {format_price(zone['zone_low'])} - {format_price(zone['zone_high'])}\n"
            msg += f"   Targets: TP1 +{STRUCTURAL_TP1_PCT:.1f}% | TP2 +{STRUCTURAL_TP2_PCT:.1f}%\n\n"
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"/structural error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')

# ============================================================================
# /recap COMMAND
# ============================================================================
async def recap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually trigger daily recap"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        now = datetime.now(timezone.utc)
        text = f"**Daily Recap - {now.strftime('%b %d, %Y')}**\n"
        
        prices = await fetch_ticker_batch()
        
        for sym in SYMBOLS:
            price = prices.get(sym)
            if price is None:
                continue
            
            df = await fetch_ohlcv(sym, '1d', 2)
            if len(df) < 2:
                continue
            
            change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            text += f"{sym}: {change:+.2f}% | {format_price(price)}\n"
            await asyncio.sleep(2)
        
        text += "\nInstitutional macro recap + next 48h whale bias."
        
        recap = await query_grok_daily_recap(text)
        
        if recap:
            import html
            escaped_recap = html.escape(recap)
            full_msg = f"**INSTITUTIONAL DAILY SUMMARY**\n\n{escaped_recap}"
            await send_throttled(CHAT_ID, full_msg, parse_mode='HTML')
        else:
            await send_throttled(CHAT_ID, "Recap generation failed", parse_mode='Markdown')
    
    except Exception as e:
        logging.error(f"/recap error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')

# ============================================================================
# /backtest COMMAND
# ============================================================================
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run improved backtest using actual OB detection"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        await update.message.reply_text("Running improved backtest (90 days)...")
        
        from bot.indicators import add_institutional_indicators
        from bot.order_blocks import find_unmitigated_order_blocks
        
        days = 90
        symbol = 'BTC/USDT'
        
        # Fetch historical data
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        df = await fetch_ohlcv(symbol, '1d', limit=days + 50, since=since)
        
        if len(df) < 50:
            await send_throttled(CHAT_ID, f"Insufficient data: {len(df)} bars", parse_mode='Markdown')
            return
        
        df = add_institutional_indicators(df)
        
        trades = []
        capital = SIMULATED_CAPITAL
        
        # Walk through history and find OB-based trades
        for i in range(50, len(df) - 10):
            # Get subset for OB detection
            df_subset = df.iloc[:i+1].copy()
            
            # Find OBs at this point in history
            obs = await find_unmitigated_order_blocks(df_subset, lookback=50, tf='1d')
            
            current_price = df_subset['close'].iloc[-1]
            atr = df_subset['atr'].iloc[-1] if 'atr' in df_subset.columns else current_price * 0.02
            
            # Look for valid setups
            for ob_type in ['bullish', 'bearish']:
                for ob in obs.get(ob_type, [])[:1]:  # Top 1 OB
                    if ob['strength'] < ROADMAP_MIN_OB_STRENGTH:
                        continue
                    
                    mid = (ob['low'] + ob['high']) / 2
                    dist_pct = abs(current_price - mid) / current_price * 100
                    
                    # Only trade if price is near OB
                    if dist_pct > 2.0:
                        continue
                    
                    direction = 'Long' if ob_type == 'bullish' else 'Short'
                    entry = mid
                    
                    if direction == 'Long':
                        sl = ob['low'] - atr * 0.3
                        tp1 = entry + atr * 1.5
                        tp2 = entry + atr * 3.0
                    else:
                        sl = ob['high'] + atr * 0.3
                        tp1 = entry - atr * 1.5
                        tp2 = entry - atr * 3.0
                    
                    risk = abs(sl - entry)
                    if risk == 0:
                        continue
                    
                    # Simulate forward
                    result = simulate_trade_forward(
                        df.iloc[i+1:i+11], direction, entry, sl, tp1, tp2
                    )
                    
                    if result:
                        trades.append(result)
            
            await asyncio.sleep(0.01)  # Prevent blocking
        
        # Calculate results
        if not trades:
            await send_throttled(CHAT_ID, "No trades generated in backtest period.", parse_mode='Markdown')
            return
        
        wins = len([t for t in trades if t['result'] == 'win'])
        losses = len([t for t in trades if t['result'] == 'loss'])
        total = wins + losses
        winrate = (wins / total * 100) if total > 0 else 0
        
        total_pnl_r = sum(t['pnl_r'] for t in trades)
        avg_win_r = np.mean([t['pnl_r'] for t in trades if t['result'] == 'win']) if wins > 0 else 0
        avg_loss_r = np.mean([t['pnl_r'] for t in trades if t['result'] == 'loss']) if losses > 0 else 0
        
        # Calculate equity curve
        equity = SIMULATED_CAPITAL
        max_equity = equity
        max_dd = 0
        
        for t in trades:
            pnl_usd = equity * 0.015 * t['pnl_r']  # 1.5% risk per trade
            equity += pnl_usd
            max_equity = max(max_equity, equity)
            dd = (max_equity - equity) / max_equity * 100
            max_dd = max(max_dd, dd)
        
        final_pnl_pct = (equity - SIMULATED_CAPITAL) / SIMULATED_CAPITAL * 100
        
        # Calculate expected hit rates
        tp1_hits = len([t for t in trades if t.get('hit_tp1')])
        tp2_hits = len([t for t in trades if t.get('hit_tp2')])
        tp1_rate = tp1_hits / total if total > 0 else 0
        tp2_rate = tp2_hits / total if total > 0 else 0
        
        # Save results
        bt_results = {
            'symbol': symbol,
            'days': days,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'winrate': winrate,
            'total_pnl_r': total_pnl_r,
            'total_pnl_pct': final_pnl_pct,
            'avg_win_r': avg_win_r,
            'avg_loss_r': avg_loss_r,
            'max_drawdown': max_dd,
            'tp1_hit_rate': tp1_rate,
            'tp2_hit_rate': tp2_rate,
            'date': datetime.now(timezone.utc).isoformat()
        }
        
        async with aiofiles.open(BACKTEST_FILE, 'w') as f:
            await f.write(json.dumps(bt_results, indent=2))
        
        msg = (
            f"**Backtest Results (90d {symbol})**\n\n"
            f"**Trades:** {total} ({wins}W / {losses}L)\n"
            f"**Win Rate:** {winrate:.1f}%\n"
            f"**Total PnL:** {total_pnl_r:+.1f}R ({final_pnl_pct:+.1f}%)\n"
            f"**Avg Win:** {avg_win_r:+.2f}R | **Avg Loss:** {avg_loss_r:.2f}R\n"
            f"**Max Drawdown:** {max_dd:.1f}%\n\n"
            f"**TP1 Hit Rate:** {tp1_rate*100:.0f}%\n"
            f"**TP2 Hit Rate:** {tp2_rate*100:.0f}%"
        )
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"/backtest error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')

def simulate_trade_forward(df: pd.DataFrame, direction: str, entry: float, 
                           sl: float, tp1: float, tp2: float) -> Optional[Dict]:
    """Simulate a trade forward through historical data"""
    if len(df) == 0:
        return None
    
    risk = abs(sl - entry)
    if risk == 0:
        return None
    
    hit_tp1 = False
    hit_tp2 = False
    hit_sl = False
    exit_price = entry
    
    for _, row in df.iterrows():
        if direction == 'Long':
            if row['low'] <= sl:
                hit_sl = True
                exit_price = sl
                break
            if row['high'] >= tp1 and not hit_tp1:
                hit_tp1 = True
            if row['high'] >= tp2:
                hit_tp2 = True
                exit_price = tp2
                break
        else:
            if row['high'] >= sl:
                hit_sl = True
                exit_price = sl
                break
            if row['low'] <= tp1 and not hit_tp1:
                hit_tp1 = True
            if row['low'] <= tp2:
                hit_tp2 = True
                exit_price = tp2
                break
    
    # Calculate PnL
    if hit_tp2:
        pnl_r = abs(tp2 - entry) / risk
        result = 'win'
    elif hit_tp1 and hit_sl:
        # Partial win then stopped at breakeven
        pnl_r = abs(tp1 - entry) / risk * 0.5
        result = 'win'
    elif hit_sl:
        pnl_r = -1.0
        result = 'loss'
    else:
        # No exit - neutral
        pnl_r = 0
        result = 'neutral'
        return None  # Don't count incomplete trades
    
    return {
        'direction': direction,
        'entry': entry,
        'exit': exit_price,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'result': result,
        'pnl_r': pnl_r,
        'hit_tp1': hit_tp1,
        'hit_tp2': hit_tp2
    }

# ============================================================================
# /backtest_all COMMAND
# ============================================================================
async def backtest_all_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run backtest on all symbols"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    await update.message.reply_text("Running multi-symbol backtest (may take a while)...")
    
    from bot.indicators import add_institutional_indicators
    from bot.order_blocks import find_unmitigated_order_blocks
    
    summary_msg = "**Multi-Symbol Backtest (90d)**\n\n"
    total_trades_all = 0
    total_wins_all = 0
    
    for symbol in SYMBOLS[:4]:  # Limit to 4 for speed
        try:
            days = 90
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
            df = await fetch_ohlcv(symbol, '1d', limit=days + 50, since=since)
            
            if len(df) < 50:
                summary_msg += f"**{symbol.replace('/USDT','')}**: Insufficient data\n"
                continue
            
            df = add_institutional_indicators(df)
            
            trades = []
            
            for i in range(50, len(df) - 10, 5):  # Step by 5 for speed
                df_subset = df.iloc[:i+1].copy()
                obs = await find_unmitigated_order_blocks(df_subset, lookback=50, tf='1d')
                
                current_price = df_subset['close'].iloc[-1]
                atr = df_subset['atr'].iloc[-1] if 'atr' in df_subset.columns else current_price * 0.02
                
                for ob_type in ['bullish', 'bearish']:
                    for ob in obs.get(ob_type, [])[:1]:
                        if ob['strength'] < ROADMAP_MIN_OB_STRENGTH:
                            continue
                        
                        mid = (ob['low'] + ob['high']) / 2
                        dist_pct = abs(current_price - mid) / current_price * 100
                        
                        if dist_pct > 2.0:
                            continue
                        
                        direction = 'Long' if ob_type == 'bullish' else 'Short'
                        entry = mid
                        
                        if direction == 'Long':
                            sl = ob['low'] - atr * 0.3
                            tp1 = entry + atr * 1.5
                            tp2 = entry + atr * 3.0
                        else:
                            sl = ob['high'] + atr * 0.3
                            tp1 = entry - atr * 1.5
                            tp2 = entry - atr * 3.0
                        
                        result = simulate_trade_forward(
                            df.iloc[i+1:i+11], direction, entry, sl, tp1, tp2
                        )
                        
                        if result:
                            trades.append(result)
            
            if trades:
                wins = len([t for t in trades if t['result'] == 'win'])
                total = len(trades)
                winrate = (wins / total * 100) if total > 0 else 0
                total_pnl_r = sum(t['pnl_r'] for t in trades)
                
                total_trades_all += total
                total_wins_all += wins
                
                summary_msg += f"**{symbol.replace('/USDT','')}**: {wins}/{total} ({winrate:.0f}%) | {total_pnl_r:+.1f}R\n"
            else:
                summary_msg += f"**{symbol.replace('/USDT','')}**: No trades\n"
            
            await asyncio.sleep(2)
        
        except Exception as e:
            summary_msg += f"**{symbol.replace('/USDT','')}**: Error\n"
            logging.error(f"Backtest error for {symbol}: {e}")
    
    # Overall summary
    overall_wr = (total_wins_all / total_trades_all * 100) if total_trades_all > 0 else 0
    summary_msg += f"\n**Overall:** {total_wins_all}/{total_trades_all} ({overall_wr:.0f}%)"
    
    await send_throttled(CHAT_ID, summary_msg, parse_mode='Markdown')

# ============================================================================
# /validate COMMAND
# ============================================================================
async def validate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run Monte Carlo validation"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    if not os.path.exists(BACKTEST_FILE):
        await update.message.reply_text("No backtest found. Run /backtest first.")
        return
    
    try:
        # Load backtest results
        with open(BACKTEST_FILE, 'r') as f:
            bt = json.load(f)
        
        winrate = bt.get('winrate', 50) / 100
        avg_win = bt.get('avg_win_r', 2.0)
        avg_loss = abs(bt.get('avg_loss_r', -1.0))
        num_trades = bt.get('total_trades', 50)
        
        # Monte Carlo simulation
        num_simulations = 1000
        simulated_pnls = []
        
        for _ in range(num_simulations):
            pnl = 0
            for _ in range(num_trades):
                if np.random.random() < winrate:
                    pnl += avg_win * np.random.uniform(0.8, 1.2)
                else:
                    pnl -= avg_loss * np.random.uniform(0.8, 1.2)
            simulated_pnls.append(pnl)
        
        actual_pnl = bt.get('total_pnl_r', 0)
        
        # Calculate statistics
        percentile_5 = np.percentile(simulated_pnls, 5)
        percentile_50 = np.percentile(simulated_pnls, 50)
        percentile_95 = np.percentile(simulated_pnls, 95)
        
        better_count = sum(1 for sim_pnl in simulated_pnls if sim_pnl >= actual_pnl)
        p_value = better_count / num_simulations
        
        # Expected value
        ev = (winrate * avg_win) - ((1 - winrate) * avg_loss)
        
        msg = (
            f"**Monte Carlo Validation ({num_simulations} sims)**\n\n"
            f"**Actual PnL**: {actual_pnl:+.2f}R\n"
            f"**Simulated 5th %ile**: {percentile_5:+.2f}R\n"
            f"**Simulated Median**: {percentile_50:+.2f}R\n"
            f"**Simulated 95th %ile**: {percentile_95:+.2f}R\n\n"
            f"**P-value**: {p_value:.4f}\n"
            f"**Expected Value**: {ev:+.3f}R per trade\n"
            f"**Confidence**: {'High (EV > 0.3)' if ev > 0.3 else 'Moderate' if ev > 0 else 'Low (negative EV)'}"
        )
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Validation error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')

# ============================================================================
# /dashboard COMMAND
# ============================================================================
async def dashboard_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed performance dashboard"""
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        if not Path(TRADE_LOG_FILE).exists():
            await update.message.reply_text("No trade history yet.")
            return
        
        trades_df = pd.read_csv(TRADE_LOG_FILE)
        
        if len(trades_df) == 0:
            await update.message.reply_text("No closed trades yet.")
            return
        
        wins = len(trades_df[trades_df['result'] == 'WIN'])
        losses = len(trades_df[trades_df['result'] == 'LOSS'])
        total = wins + losses
        winrate = (wins / total * 100) if total > 0 else 0
        
        avg_win = trades_df[trades_df['result'] == 'WIN']['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['result'] == 'LOSS']['pnl_pct'].mean() if losses > 0 else 0
        
        expectancy = (winrate/100 * avg_win) + ((100-winrate)/100 * avg_loss)
        
        # From roadmap stats
        from_roadmap = len(trades_df[trades_df.get('from_roadmap', False) == True]) if 'from_roadmap' in trades_df.columns else 0
        
        msg = f"**📊 Performance Dashboard**\n\n"
        msg += f"**Overall**: {wins}W / {losses}L ({winrate:.1f}%)\n"
        msg += f"**Avg Win**: +{avg_win:.2f}% | **Avg Loss**: {avg_loss:.2f}%\n"
        msg += f"**Expectancy**: {expectancy:+.2f}%\n"
        
        if from_roadmap > 0:
            msg += f"**From Roadmap**: {from_roadmap} trades\n"
        
        msg += "\n"
        
        if 'symbol' in trades_df.columns:
            symbol_stats = trades_df.groupby('symbol').agg({
                'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
                'pnl_pct': 'sum'
            }).round(1)
            
            msg += "**By Symbol**:\n"
            for symbol, row in symbol_stats.iterrows():
                msg += f"  {symbol}: {row['result']:.0f}% WR | {row['pnl_pct']:+.1f}%\n"
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}", parse_mode='Markdown')


# ============================================================================
# v27.12.13: FACTOR ANALYSIS COMMAND
# ============================================================================

async def factor_analysis_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Show which confluence factors produce the best win rates.
    
    Usage: /factors
    """
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        from bot.models import get_signal_tracker
        
        tracker = get_signal_tracker()
        rankings = tracker.get_factor_rankings(min_signals=3)
        overall = tracker.get_overall_stats()
        
        if not rankings:
            await update.message.reply_text(
                "📊 **Factor Analysis**\n\n"
                "Not enough data yet. Need at least 3 signals per factor.\n"
                f"Total signals tracked: {overall['total_signals']}"
            )
            return
        
        msg = "📊 **Factor Performance Analysis**\n\n"
        msg += f"**Overall**: {overall['wins']}W / {overall['losses']}L ({overall['win_rate']:.1f}%)\n"
        msg += f"**Factors Tracked**: {overall['factors_tracked']}\n\n"
        msg += "**Top Performing Factors:**\n"
        
        for i, factor_data in enumerate(rankings[:10], 1):
            wr = factor_data['win_rate']
            
            # Win rate emoji
            if wr >= 70:
                emoji = "🟢"
            elif wr >= 55:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            factor_name = factor_data['factor'][:20]  # Truncate long names
            msg += f"{emoji} {i}. **{factor_name}**\n"
            msg += f"    {factor_data['signals']} signals | {wr:.0f}% WR | TP2: {factor_data['tp2_rate']:.0f}%\n"
        
        if len(rankings) > 10:
            msg += f"\n_...and {len(rankings) - 10} more factors_"
        
        msg += "\n\n💡 **Tip**: Focus on factors with >60% win rate"
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Factor analysis error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}")


# ============================================================================
# v27.12.13: SYSTEM HEALTH COMMAND (ENHANCED)
# ============================================================================

async def system_health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Comprehensive system health check.
    
    Usage: /system_health
    """
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        from bot.config import (
            BLOFIN_API_KEY, AUTO_TRADE_ENABLED, BLOFIN_DEMO_MODE,
            DYNAMIC_TP_ENABLED, MTF_WEIGHTING_ENABLED, 
            SIGNAL_TRACKING_ENABLED, LIQUIDITY_ANALYSIS_ENABLED,
            SESSION_TRADING_ENABLED, CORRELATION_SIZING_ENABLED
        )
        
        msg = "🏥 **System Health Dashboard**\n\n"
        
        # === API Status ===
        msg += "**📡 API Status:**\n"
        
        # Claude API
        try:
            from bot.claude_api import anthropic_client
            claude_ok = anthropic_client is not None
            msg += f"  Claude: {'✅ OK' if claude_ok else '❌ Not configured'}\n"
        except Exception as e:
            msg += f"  Claude: ❌ Error ({str(e)[:30]})\n"
        
        # Bybit API
        try:
            from bot.data_fetcher import exchange
            bybit_ok = exchange is not None
            msg += f"  Bybit: {'✅ OK' if bybit_ok else '❌ Not configured'}\n"
        except Exception as e:
            msg += f"  Bybit: ❌ Error ({str(e)[:30]})\n"
        
        # Blofin API
        if BLOFIN_API_KEY:
            mode = "DEMO" if BLOFIN_DEMO_MODE else "LIVE"
            status = "ON" if AUTO_TRADE_ENABLED else "OFF"
            msg += f"  Blofin: ✅ {mode} ({status})\n"
        else:
            msg += f"  Blofin: ⚪ Not configured\n"
        
        # === Feature Status ===
        msg += "\n**⚙️ Features:**\n"
        
        features = [
            ("Dynamic TPs", DYNAMIC_TP_ENABLED),
            ("MTF Weighting", MTF_WEIGHTING_ENABLED),
            ("Signal Tracking", SIGNAL_TRACKING_ENABLED),
            ("Liquidity Analysis", LIQUIDITY_ANALYSIS_ENABLED),
            ("Session Trading", SESSION_TRADING_ENABLED),
            ("Correlation Sizing", CORRELATION_SIZING_ENABLED),
        ]
        
        for name, enabled in features:
            emoji = "✅" if enabled else "⚪"
            msg += f"  {emoji} {name}\n"
        
        # === Memory & Trades ===
        msg += "\n**📊 Current State:**\n"
        msg += f"  Open Trades: {len(open_trades)}\n"
        msg += f"  Protected: {len(protected_trades)}\n"
        
        # Stats
        stats = load_stats()
        msg += f"  Capital: ${stats.get('capital', SIMULATED_CAPITAL):,.2f}\n"
        msg += f"  W/L: {stats.get('wins', 0)}/{stats.get('losses', 0)}\n"
        
        # === Session Info ===
        try:
            from bot.blofin_trader import get_current_session
            session_name, session_mult = get_current_session()
            msg += f"\n**🕐 Session:** {session_name} ({session_mult}x)\n"
        except Exception:
            pass  # Session info is optional, no need to log
        
        # === Version ===
        msg += f"\n**📦 Version:** v{BOT_VERSION}\n"
        msg += f"**⏰ Check Time:** {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        
        await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"System health error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}")


# ============================================================================
# v27.12.15: LIQUIDITY COMMAND
# ============================================================================
async def liquidity_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Show liquidity heatmap for a symbol.
    Usage: /liquidity BTC or /liquidity BTCUSDT
    """
    try:
        # Parse symbol from command
        if context.args and len(context.args) > 0:
            symbol_input = context.args[0].upper()
            # Normalize symbol
            if not symbol_input.endswith('USDT'):
                symbol_input += 'USDT'
            if '/' not in symbol_input:
                symbol_input = symbol_input.replace('USDT', '/USDT')
        else:
            symbol_input = 'BTC/USDT'
        
        await send_throttled(CHAT_ID, f"🔍 Analyzing liquidity for {symbol_input}...")
        
        # Fetch data
        df = await fetch_ohlcv(symbol_input, '4h', limit=100)
        if df is None or len(df) < 50:
            await send_throttled(CHAT_ID, f"❌ Insufficient data for {symbol_input}")
            return
        
        prices = await fetch_ticker_batch()
        current_price = prices.get(symbol_input, df['close'].iloc[-1])
        
        # Generate liquidity map
        try:
            from bot.liquidity_map import generate_liquidity_map, format_liquidity_report
            liquidity_map = generate_liquidity_map(df, symbol_input, current_price)
            
            if liquidity_map:
                report = format_liquidity_report(liquidity_map)
                await send_throttled(CHAT_ID, report, parse_mode='Markdown')
            else:
                await send_throttled(CHAT_ID, f"❌ Could not generate liquidity map for {symbol_input}")
        except ImportError:
            await send_throttled(CHAT_ID, "❌ Liquidity map module not available")
        except Exception as e:
            await send_throttled(CHAT_ID, f"❌ Liquidity analysis error: {str(e)[:100]}")
            
    except Exception as e:
        logging.error(f"Liquidity command error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}")


# ============================================================================
# v27.12.15: ORDERBOOK COMMAND
# ============================================================================
async def orderbook_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Show orderbook imbalance for a symbol.
    Usage: /orderbook BTC or /orderbook BTCUSDT
    """
    try:
        # Parse symbol from command
        if context.args and len(context.args) > 0:
            symbol_input = context.args[0].upper()
            if not symbol_input.endswith('USDT'):
                symbol_input += 'USDT'
            if '/' not in symbol_input:
                symbol_input = symbol_input.replace('USDT', '/USDT')
        else:
            symbol_input = 'BTC/USDT'
        
        await send_throttled(CHAT_ID, f"📊 Fetching orderbook for {symbol_input}...")
        
        # Fetch orderbook
        try:
            from bot.data_fetcher import fetch_order_flow_batch, analyze_orderbook_imbalance
            
            order_books = await fetch_order_flow_batch()
            
            if symbol_input not in order_books:
                await send_throttled(CHAT_ID, f"❌ No orderbook data for {symbol_input}")
                return
            
            orderbook = order_books[symbol_input]
            analysis = analyze_orderbook_imbalance(orderbook)
            
            # Format response
            signal = analysis.get('signal', 'NEUTRAL')
            ratio = analysis.get('imbalance_ratio', 1.0)
            confidence = analysis.get('confidence', 0)
            bid_vol = analysis.get('bid_volume', 0)
            ask_vol = analysis.get('ask_volume', 0)
            bid_wall = analysis.get('bid_wall')
            ask_wall = analysis.get('ask_wall')
            
            # Signal emoji
            if signal == 'LONG':
                signal_emoji = "🟢"
            elif signal == 'SHORT':
                signal_emoji = "🔴"
            else:
                signal_emoji = "⚪"
            
            msg = f"📊 **Orderbook Analysis: {symbol_input}**\n\n"
            msg += f"**Signal:** {signal_emoji} {signal}\n"
            msg += f"**Confidence:** {confidence}%\n"
            msg += f"**Imbalance Ratio:** {ratio:.2f}\n\n"
            msg += f"**Volume:**\n"
            msg += f"  • Bids: {bid_vol:,.0f}\n"
            msg += f"  • Asks: {ask_vol:,.0f}\n"
            
            if bid_wall:
                msg += f"\n**🧱 Bid Wall:** ${bid_wall['price']:,.2f} ({bid_wall['volume']:,.0f})\n"
            if ask_wall:
                msg += f"**🧱 Ask Wall:** ${ask_wall['price']:,.2f} ({ask_wall['volume']:,.0f})\n"
            
            # Interpretation
            msg += f"\n**Interpretation:**\n"
            if ratio > 1.5:
                msg += "Strong buying pressure - more bids than asks\n"
            elif ratio < 0.67:
                msg += "Strong selling pressure - more asks than bids\n"
            else:
                msg += "Balanced orderbook - no clear direction\n"
            
            await send_throttled(CHAT_ID, msg, parse_mode='Markdown')
            
        except ImportError as e:
            await send_throttled(CHAT_ID, f"❌ Orderbook module not available: {e}")
        except Exception as e:
            await send_throttled(CHAT_ID, f"❌ Orderbook analysis error: {str(e)[:100]}")
            
    except Exception as e:
        logging.error(f"Orderbook command error: {e}")
        await send_throttled(CHAT_ID, f"Error: {str(e)}")


# ============================================================================
# v27.12.15: PROFESSIONAL BACKTEST COMMAND
# ============================================================================
async def backtest_pro_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Run professional-grade backtest with proper metrics.
    
    Usage: 
    - /backtest_pro - Run on all symbols (365 days)
    - /backtest_pro BTC - Run on single symbol
    - /backtest_pro 180 - Run with custom days
    """
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        # Parse arguments
        days = 365
        symbols = SYMBOLS.copy()
        
        if context.args:
            for arg in context.args:
                if arg.isdigit():
                    days = int(arg)
                else:
                    # Treat as symbol
                    sym = arg.upper()
                    if not sym.endswith('USDT'):
                        sym += 'USDT'
                    if '/' not in sym:
                        sym = sym.replace('USDT', '/USDT')
                    symbols = [sym]
        
        await send_throttled(
            CHAT_ID, 
            f"🔬 **Starting Professional Backtest**\n\n"
            f"• Symbols: {len(symbols)}\n"
            f"• Period: {days} days\n"
            f"• Features: Monte Carlo, Proper Fees, Leverage-Adjusted PnL\n\n"
            f"_This may take a few minutes..._",
            parse_mode='Markdown'
        )
        
        # Import and run backtest
        from bot.backtest_pro import BacktestEngine, BacktestConfig, format_backtest_report
        
        config = BacktestConfig(days=days)
        engine = BacktestEngine(config)
        
        async def progress_update(msg):
            await send_throttled(CHAT_ID, f"⏳ {msg}")
        
        result = await engine.run_backtest(symbols, progress_callback=progress_update)
        
        # Format and send results
        report = format_backtest_report(result)
        await send_throttled(CHAT_ID, report, parse_mode='Markdown')
        
        # Save results to file
        results_file = 'data/backtest_pro_results.json'
        os.makedirs('data', exist_ok=True)
        
        save_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': {
                'days': config.days,
                'symbols': symbols,
                'min_ob_strength': config.min_ob_strength,
                'max_entry_distance': config.max_entry_distance_pct
            },
            'results': {
                'total_trades': result.total_trades,
                'wins': result.wins,
                'losses': result.losses,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_pnl_pct': result.total_pnl_pct,
                'max_drawdown': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'mc_prob_profit': result.mc_probability_profit
            },
            'factor_stats': result.factor_stats
        }
        
        async with aiofiles.open(results_file, 'w') as f:
            await f.write(json.dumps(save_data, indent=2))
        
        await send_throttled(CHAT_ID, f"📁 Results saved to `{results_file}`", parse_mode='Markdown')
        
    except ImportError as e:
        await send_throttled(CHAT_ID, f"❌ Backtest module not available: {e}")
    except Exception as e:
        logging.error(f"Professional backtest error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        await send_throttled(CHAT_ID, f"❌ Error: {str(e)[:200]}")


# ============================================================================
# v27.12.17: REALISTIC BACKTEST COMMAND
# ============================================================================
async def backtest_real_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Run realistic backtest with proper methodology.
    
    This backtest includes:
    - No look-ahead bias (bar-by-bar simulation)
    - Confirmation requirements (candle close + bounce)
    - Fill probability modeling (not every order fills)
    - Trade frequency limits (max 2/week per symbol)
    - Realistic costs (fees + slippage + funding)
    - Statistical validation
    
    Usage: 
    - /backtest_real - Run on all symbols (365 days)
    - /backtest_real BTC - Run on single symbol
    - /backtest_real 180 - Run with custom days
    """
    if str(update.effective_user.id) != CHAT_ID:
        await update.message.reply_text("Unauthorized")
        return
    
    try:
        # Parse arguments
        days = 365
        symbols = SYMBOLS.copy()
        
        if context.args:
            for arg in context.args:
                if arg.isdigit():
                    days = int(arg)
                else:
                    sym = arg.upper()
                    if not sym.endswith('USDT'):
                        sym += 'USDT'
                    if '/' not in sym:
                        sym = sym.replace('USDT', '/USDT')
                    symbols = [sym]
        
        await send_throttled(
            CHAT_ID, 
            f"🔬 **Starting REALISTIC Backtest**\n\n"
            f"• Symbols: {len(symbols)}\n"
            f"• Period: {days} days\n\n"
            f"**Methodology:**\n"
            f"• ✅ No look-ahead bias\n"
            f"• ✅ Bounce confirmation required\n"
            f"• ✅ Fill probability modeling\n"
            f"• ✅ Trade frequency limits\n"
            f"• ✅ Realistic slippage & fees\n"
            f"• ✅ Monte Carlo validation\n\n"
            f"_This produces REALISTIC results. Expect:\n"
            f"Win Rate: 45-55% | PF: 1.3-2.0 | Trades: 50-150/year_\n\n"
            f"⏳ Processing...",
            parse_mode='Markdown'
        )
        
        # Import and run backtest
        from bot.backtest_realistic import (
            RealisticBacktestEngine, 
            RealisticBacktestConfig, 
            format_realistic_report
        )
        
        config = RealisticBacktestConfig(days=days)
        engine = RealisticBacktestEngine(config)
        
        async def progress_update(msg):
            await send_throttled(CHAT_ID, f"⏳ {msg}")
        
        result = await engine.run_backtest(symbols, progress_callback=progress_update)
        
        # Format and send results
        report = format_realistic_report(result)
        await send_throttled(CHAT_ID, report, parse_mode='Markdown')
        
        # Save results
        results_file = 'data/backtest_realistic_results.json'
        os.makedirs('data', exist_ok=True)
        
        save_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'methodology': 'realistic_v27.12.17',
            'config': {
                'days': config.days,
                'symbols': symbols,
                'min_ob_strength': config.min_ob_strength,
                'min_confluence': config.min_confluence_factors,
                'require_confirmation': config.require_bounce_confirm,
                'base_fill_probability': config.base_fill_probability,
                'slippage_major': config.slippage_major_pct,
                'slippage_midcap': config.slippage_midcap_pct
            },
            'funnel': {
                'signals_detected': result.total_signals,
                'signals_confirmed': result.signals_confirmed,
                'orders_placed': result.orders_placed,
                'orders_filled': result.orders_filled
            },
            'results': {
                'total_trades': len(result.trades),
                'wins': result.wins,
                'losses': result.losses,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_pnl_pct': result.total_pnl_pct,
                'max_drawdown': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio
            },
            'costs': {
                'total_fees': result.total_fees_pct,
                'total_slippage': result.total_slippage_pct,
                'total_funding': result.total_funding_pct
            },
            'monte_carlo': {
                'median_pnl': result.mc_median_pnl,
                'lower_95': result.mc_95_lower,
                'upper_95': result.mc_95_upper,
                'prob_profit': result.mc_prob_profit
            },
            'validation': {
                'is_valid': result.is_statistically_valid,
                'warnings': result.validation_warnings
            },
            'factor_performance': result.factor_performance
        }
        
        async with aiofiles.open(results_file, 'w') as f:
            await f.write(json.dumps(save_data, indent=2))
        
        await send_throttled(CHAT_ID, f"📁 Results saved to `{results_file}`", parse_mode='Markdown')
        
    except ImportError as e:
        await send_throttled(CHAT_ID, f"❌ Realistic backtest module not available: {e}")
    except Exception as e:
        logging.error(f"Realistic backtest error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        await send_throttled(CHAT_ID, f"❌ Error: {str(e)[:200]}")
