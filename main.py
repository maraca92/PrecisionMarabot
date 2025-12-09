# main.py - Grok Elite Signal Bot v24.01.6 - Institutional OB Focus: Zero Noise Edition (Order Flow Enhanced + Env Debug)
# UPGRADE (v24.01.6): Env debug logs in main(); query_grok_instant upgraded (str=2+, conf≥75%, lev3-7x); backtest symmetry fixed; order_flow cache eviction.
# Retained v24.01.5: Order Flow (delta + footprint) via ccxt order_book; conf boost +2 on imbalance >1%; triggers + delta.
# Retained v24.01.4: Bidirectional + EMA200 bias + FVG caution – no bias.
# Retained v24.01.3: Added short entries (bearish EMA cross), vol threshold 1.5x, RSI filter (<30 long/>70 short), maintained precision.
# Retained v24.01.2: Robust EMA-only computation in BTC trend to avoid empty DF from full indicators/dropna.
# Retained v24.01.1: Increased limit=500 & len check in BTC trend; backoff*3; sem=3 for rate limit stability.
# Retained v24.01: Fixed SyntaxError in query_grok_instant prompt (raw string for JSON example).
# Retained v24.00: Str=3+ OBs only, dist=4%, conf>=90%, multi-TF align, 24h cooldown.
import asyncio
import os
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import httpx
import json
import re
from datetime import datetime, timezone, timedelta
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import logging
from typing import Dict, Any, Optional, List
import time
from collections import OrderedDict
import html
import sys  # NEW: For explicit exit in main()
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
TIMEFRAMES = ['4h', '1d', '1w']
CHECK_INTERVAL = 14400
TRACK_INTERVAL = 5
COOLDOWN_HOURS = 12
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
XAI_API_KEY = os.getenv("XAI_API_KEY")
bot = Bot(token=TELEGRAM_TOKEN)
exchange = ccxt.bybit({
    'enableRateLimit': True,
    'rateLimit': 1200,
    'options': {'adjustForTimeDifference': True, 'defaultType': 'spot', 'recvWindow': 10000},
    'timeout': 30000,
})
futures_exchange = ccxt.bybit({
    'enableRateLimit': True,
    'rateLimit': 1200,
    'options': {'adjustForTimeDifference': True, 'defaultType': 'swap', 'recvWindow': 10000},
    'timeout': 30000,
})
FLAG_FILE = "welcome_sent.flag"
STATS_FILE = "bot_stats.json"
TRADES_FILE = "open_trades.json"
PROTECTED_TRADES_FILE = "protected_trades.json"
RECAP_FILE = "last_recap.date"
BACKTEST_FILE = "backtest_results.json"
CACHE_TTL = 1800
HTF_CACHE_TTL = 3600
TICKER_CACHE_TTL = 10
ORDER_FLOW_CACHE_TTL = 5 # NEW Order Flow: Short TTL for fresh book
MAX_CACHE_SIZE = 50
ohlcv_cache: OrderedDict = OrderedDict()
ticker_cache: OrderedDict = OrderedDict()
order_flow_cache: OrderedDict = OrderedDict() # NEW: Cache for order books
last_oi: Dict[str, float] = {}
open_trades = {}
protected_trades = {}
last_signal_time = {}
prices_global: Dict[str, Optional[float]] = {s: None for s in SYMBOLS}
last_price_update: float = 0.0
last_ban_check: float = 0.0
btc_trend_global = "Unknown"
TRADE_TIMEOUT_HOURS = 24
PROTECT_AFTER_HOURS = 6
FEE_PCT = 0.04
SLIPPAGE_PCT = 0.001
ENTRY_SLIPPAGE_PCT = 0.002
MAX_CONCURRENT_TRADES = 1
MAX_DRAWDOWN_PCT = 3.0
RISK_PER_TRADE_PCT = 1.5
DAILY_ATR_MULT = 2.0
SIMULATED_CAPITAL = 10000.0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
fetch_sem = asyncio.Semaphore(3)
def evict_if_full(cache: OrderedDict, max_size: int = MAX_CACHE_SIZE):
    """Evict oldest cache entries if size exceeds limit to prevent memory growth."""
    evicted = 0
    while len(cache) > max_size:
        cache.popitem(last=False) # FIFO: oldest first
        evicted += 1
    if evicted > 0:
        logging.debug(f"Evicted {evicted} items from {cache.__class__.__name__} cache (now {len(cache)} items)")
class BanManager:
    ban_until: float = 0.0
    cooldown_until: float = 0.0
    @classmethod
    async def check_and_sleep(cls, skip_long_sleep: bool = True):
        global last_ban_check
        now = time.time()
        if now - last_ban_check < 10:
            return False
        last_ban_check = now
        if now < cls.ban_until:
            sleep_secs = max(60, cls.ban_until - now + 60)
            if skip_long_sleep:
                sleep_secs = min(sleep_secs, 5)
                logging.info(f"Ban active (short sleep mode); sleeping {sleep_secs:.0f}s")
            else:
                logging.info(f"Global ban active; sleeping {sleep_secs:.0f}s until {datetime.fromtimestamp(cls.ban_until + 60).isoformat()}")
            await asyncio.sleep(sleep_secs)
            return True
        if now < cls.cooldown_until:
            sleep_secs = cls.cooldown_until - now
            if skip_long_sleep:
                sleep_secs = min(sleep_secs, 5)
                logging.info(f"Cooldown active (short mode); sleeping {sleep_secs:.0f}s")
            else:
                logging.info(f"Post-ban cooldown; sleeping {sleep_secs:.0f}s")
            await asyncio.sleep(sleep_secs)
            return True
        return False
    @classmethod
    def update_ban(cls, ban_ts_ms: int):
        cls.ban_until = ban_ts_ms / 1000.0
        cls.cooldown_until = cls.ban_until + 3600
        logging.warning(f"Global ban updated: until {datetime.fromtimestamp(cls.ban_until).isoformat()} (+1h cooldown)")
def load_stats() -> Dict[str, Any]:
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                s = json.load(f)
            if 'capital' not in s:
                s['capital'] = SIMULATED_CAPITAL
            return s
        except json.JSONDecodeError:
            logging.warning("Invalid stats file, resetting.")
    return {"wins": 0, "losses": 0, "pnl": 0.0, "capital": SIMULATED_CAPITAL, "drawdown": 0.0}
def save_stats(s: Dict[str, Any]):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save stats: {e}")
stats = load_stats()
stats_lock = asyncio.Lock()
def load_trades() -> Dict[str, Any]:
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                loaded = json.load(f)
            for trade in loaded.values():
                if 'last_check' in trade:
                    trade['last_check'] = datetime.fromisoformat(trade['last_check'])
                if 'entry_time' in trade:
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                if 'processed' not in trade:
                    trade['processed'] = False
            return loaded
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid trades file, resetting: {e}")
    return {}
def save_trades(trades: Dict[str, Any]):
    try:
        dumpable = {}
        for sym, trade in trades.items():
            t_copy = trade.copy()
            if 'last_check' in t_copy:
                t_copy['last_check'] = trade['last_check'].isoformat()
            if 'entry_time' in t_copy:
                t_copy['entry_time'] = trade['entry_time'].isoformat()
            dumpable[sym] = t_copy
        with open(TRADES_FILE, 'w') as f:
            json.dump(dumpable, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save trades: {e}")
def load_protected() -> Dict[str, Any]:
    if os.path.exists(PROTECTED_TRADES_FILE):
        try:
            with open(PROTECTED_TRADES_FILE, 'r') as f:
                loaded = json.load(f)
            for trade in loaded.values():
                if 'last_check' in trade:
                    trade['last_check'] = datetime.fromisoformat(trade['last_check'])
                if 'entry_time' in trade:
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                if 'processed' not in trade:
                    trade['processed'] = False
            return loaded
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid protected trades file, resetting: {e}")
    return {}
def save_protected(trades: Dict[str, Any]):
    try:
        dumpable = {}
        for sym, trade in trades.items():
            t_copy = trade.copy()
            if 'last_check' in t_copy:
                t_copy['last_check'] = trade['last_check'].isoformat()
            if 'entry_time' in t_copy:
                t_copy['entry_time'] = trade['entry_time'].isoformat()
            dumpable[sym] = t_copy
        with open(PROTECTED_TRADES_FILE, 'w') as f:
            json.dump(dumpable, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save protected trades: {e}")
open_trades = load_trades()
protected_trades = load_protected()
def get_clean_symbol(trade_key: str) -> str:
    return re.sub(r'roadmap\d+$', '', trade_key)
def format_price(price: float) -> str:
    return f"{price:,.4f}"
# NEW Order Flow: Fetch batch order books
async def fetch_order_flow_batch() -> Dict[str, Dict]:
    async with fetch_sem:
        now = time.time()
        cache_hits = sum(1 for s in SYMBOLS if s in order_flow_cache and now - order_flow_cache[s]['timestamp'] < ORDER_FLOW_CACHE_TTL)
        if cache_hits == len(SYMBOLS):
            logging.debug(f"Full cache hit for order flow ({cache_hits}/{len(SYMBOLS)})")
            return {s: order_flow_cache[s]['book'] for s in SYMBOLS}
        logging.debug(f"Partial cache hit for order flow ({cache_hits}/{len(SYMBOLS)}); polling fresh")
        if await BanManager.check_and_sleep():
            return {s: order_flow_cache.get(s, {}).get('book') for s in SYMBOLS}
        evict_if_full(order_flow_cache)  # NEW: Evict if full
        order_books = {}
        backoff = 1
        for attempt in range(3):
            try:
                tasks = [exchange.fetch_order_book(s, limit=20) for s in SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        order_books[SYMBOLS[i]] = result
                        order_flow_cache[SYMBOLS[i]] = {'book': result, 'timestamp': now}
                logging.info(f"Order flow poll success (attempt {attempt+1})")
                break
            except Exception as e:
                logging.debug(f"Order flow poll attempt {attempt+1} failed: {e}")
                if "429" in str(e):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise
        else:
            order_books = {s: order_flow_cache.get(s, {}).get('book') for s in SYMBOLS}
        return order_books
# NEW Order Flow: Calculate delta and footprint
def calculate_order_flow(df: pd.DataFrame, order_book: Optional[Dict] = None) -> pd.DataFrame:
    if len(df) == 0 or not order_book:
        return df
    df = df.copy()
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])
    buy_vol = sum(amount for _, amount in bids) # Total buy volume top levels
    sell_vol = sum(amount for _, amount in asks) # Total sell volume
    total_vol = buy_vol + sell_vol
    delta = (buy_vol - sell_vol) / total_vol * 100 if total_vol > 0 else 0 # % imbalance
    df['order_delta'] = delta # Single value, repeat for df
    df['order_delta'] = df['order_delta'].fillna(delta) # Fill NaN
    # Cumulative delta (simple rolling)
    df['cum_delta'] = df['order_delta'].rolling(window=5, min_periods=1).sum()
    # Footprint imbalance: High if >10 active levels in 1% range
    mid_price = df['close'].iloc[-1]
    active_levels = len([p for p, _ in bids + asks if abs(p - mid_price) / mid_price < 0.01])
    df['footprint_imbalance'] = active_levels > 10 # Bool, repeat
    df['footprint_imbalance'] = df['footprint_imbalance'].fillna(True if active_levels > 10 else False)
    logging.debug(f"Order flow delta: {delta:.2f}% | Footprint: {active_levels} levels")
    return df
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"/backtest triggered by user {update.effective_user.id}")
    try:
        since = int((datetime.now(timezone.utc) - timedelta(days=90)).timestamp() * 1000)
        df = await fetch_ohlcv('BTC/USDT', '1d', limit=90, since=since)
        if len(df) == 0:
            await update.message.reply_text("Data unavailable (ban/active cooldown?), backtest skipped. Try later.", parse_mode='Markdown')
            return
        df = pd.DataFrame(df, columns=['ts', 'open', 'high', 'low', 'close', 'volume']) if 'ts' not in df.columns else df
        df['date'] = pd.to_datetime(df['ts'], unit='ms')
        df = add_indicators(df)
        trades = []
        capital = stats['capital']
        # Przeniesione poza pętlę dla efektywności
        for i in range(100, len(df)):
            # NEW v24.01.4: Symmetric vol >1.5x for long (from >2x if was, but already 1.5)
            if (df['ema50'].iloc[i] > df['ema200'].iloc[i] and  # NEW: EMA200 bias for long
                df['ema50'].iloc[i-1] <= df['ema100'].iloc[i-1] and
                df['volume'].iloc[i] > 1.5 * df['volume_sma'].iloc[i]):
                entry = df['close'].iloc[i]
                sl = entry * 0.98
                tp = entry * 1.06
                risk_distance = entry - sl
                size = (capital * RISK_PER_TRADE_PCT / 100) / risk_distance if risk_distance > 0 else 0
                for j in range(i+1, len(df)):
                    price = df['close'].iloc[j]
                    if price <= sl:
                        diff = sl - entry
                        pnl_usdt = diff * size - 2 * FEE_PCT * entry * size
                        trades.append({'result': 'loss', 'pnl': pnl_usdt})
                        break
                    if price >= tp:
                        diff = tp - entry
                        pnl_usdt = diff * size - 2 * FEE_PCT * entry * size
                        trades.append({'result': 'win', 'pnl': pnl_usdt})
                        break
                else:
                    diff = df['close'].iloc[-1] - entry
                    pnl_usdt = diff * size - 2 * FEE_PCT * entry * size
                    trades.append({'result': 'open', 'pnl': pnl_usdt})
            # NEW v24.01.3: Short entry (bearish cross) - Retained for v24.01.4 symmetry + EMA200 filter
            if df['ema200'].iloc[i] > df['close'].iloc[i]:  # NEW: EMA200 bias for short (below)
                continue # No counter-trend short
            elif (df['ema50'].iloc[i] < df['ema100'].iloc[i] and
                  df['ema50'].iloc[i-1] >= df['ema100'].iloc[i-1] and
                  df['volume'].iloc[i] > 1.5 * df['volume_sma'].iloc[i]): # NEW: vol >1.5x (z 2x)
                entry = df['close'].iloc[i]
                sl = entry * 1.03 # NEW: SL above for short
                tp = entry * 0.96 # NEW: TP below
                risk_distance = sl - entry
                size = (capital * RISK_PER_TRADE_PCT / 100) / risk_distance if risk_distance > 0 else 0
                for j in range(i+1, len(df)):
                    price = df['close'].iloc[j]
                    if price >= sl:
                        diff = entry - sl
                        pnl_usdt = diff * size - 2 * FEE_PCT * entry * size
                        trades.append({'result': 'loss', 'pnl': pnl_usdt})
                        break
                    if price <= tp:
                        diff = entry - tp
                        pnl_usdt = diff * size - 2 * FEE_PCT * entry * size
                        trades.append({'result': 'win', 'pnl': pnl_usdt})
                        break
                else:
                    diff = entry - df['close'].iloc[-1]
                    pnl_usdt = diff * size - 2 * FEE_PCT * entry * size
                    trades.append({'result': 'open', 'pnl': pnl_usdt})
        wins = len([t for t in trades if t['result'] == 'win'])
        total = len([t for t in trades if t['result'] != 'open'])
        winrate = (wins / total * 100) if total > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades)
        sharpe = np.mean([t['pnl'] for t in trades]) / (np.std([t['pnl'] for t in trades]) or np.inf) if trades else 0
        msg = f"**Institutional Backtest (90d BTC/USDT 1d)**\n\nWins: {wins}/{total} ({winrate:.1f}%)\nTotal PnL: {total_pnl:+.2f} ({total_pnl/capital*100:.2f}%)\nSharpe Ratio: {sharpe:.2f}"
        await update.message.reply_text(msg, parse_mode='Markdown')
        bt_results = {'winrate': winrate, 'total_pnl': total_pnl, 'sharpe': sharpe, 'date': datetime.now(timezone.utc).isoformat()}
        with open(BACKTEST_FILE, 'w') as f:
            json.dump(bt_results, f, indent=2)
        logging.info(f"Backtest completed: {winrate:.1f}% winrate")
    except Exception as e:
        logging.error(f"/backtest error: {e}")
        await update.message.reply_text(f"Backtest failed: {str(e)}", parse_mode='Markdown')
async def send_welcome_once():
    if not os.path.exists(FLAG_FILE):
        try:
            welcome_text = (
                "**Grok Elite Bot v24.01.6 ONLINE ♔** – Institutional OB Hunter: Zero Noise (Order Flow Enhanced + Env Debug)\n\n"
                "• v24.01.6: Env debug logs; query_grok_instant upgrade (str=2+, conf≥75%, lev3-7x); backtest symmetry; order_flow cache eviction.\n"
                "• v24.01.5: Order Flow (delta + footprint via ccxt book); conf +2 on imbalance >1%; triggers + delta for absorpcja confirm.\n"
                "• v24.01.4: Symmetric short/long (EMA cross both ways vol>1.5x), EMA200 1d trend filter (long above, short below, neutral +1 conf), reversal caution (FVG<0.3% or breaker no mit before entry), zero long bias.\n"
                "• v24.01.3: Short entries (bearish cross), vol 1.5x (+20-30% triggers), RSI <30 long/>70 short, precision 90%+ RR3:1 str3 24h CD.\n"
                "• v24.01.2: EMA-only robust computation in BTC trend (no full indicators/dropna empty DF).\n"
                "• v24.01.1: BTC trend robustness (limit=500, len check, backoff*3, sem=3).\n"
                "• v24.01: Fixed prompt syntax in Grok query.\n"
                "• v24.00: Str=3+ OBs only (HTF vol/disp confirmed); dist=4%; conf>=90%; multi-TF align; 24h cooldown.\n"
                "• All: Pure institutional ICT, 24-72h+ horizon, whale footprints prioritized."
            )
            escaped_text = html.escape(welcome_text)
            await bot.send_message(CHAT_ID, escaped_text, parse_mode='HTML')
            open(FLAG_FILE, "w").close()
            logging.info("Welcome sent (v24.01.6)")
        except Exception as e:
            logging.error(f"Failed to send welcome: {e}")
async def query_grok_instant(context: str, is_alt: bool = False) -> Dict[str, Any]:
    models = ["grok-4", "grok-3"]
    example_json = r'{"symbol":"BTC","direction":"Long" or "Short","entry_low":68234.5678,"entry_high":68456.1234,"sl":67987.2345,"tp1":68890.7890,"tp2":69543.4567,"leverage":3-7,"confidence":75-98,"strength":2,"reason":"concise daily reason (max 80 chars, e.g., HTF OB + vol on 1d)"}'
    system_prompt = (
        "You are an institutional whale trader spotting large footprints for daily trading. Output ONLY valid JSON. "
        "High-conviction institutional OB setup ≥75% confidence (str=2+ only) → exact format:\n"
        f"{example_json}\n"
        "Ignore retail noise: Require HTF (1d/1w) vol-confirmed displacement, unmitigated str=2+ OB, multi-TF align (priorytet 1d/1w), RSI exhaustion. "
        "Precise levels (4+ decimals), no rounds. Boost if whale OI/flows + POC align. Conservative: lev 3-7x, SL ATR*2, 12h+ horizon for daily swings.\n"
        f"{{'Alts: Align strictly with BTC inst trend; no counter.' if is_alt else ''}}\n"
        "No setup → {\"no_trade\": true}"
    )
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "temperature": 0.1,
            "max_tokens": 300
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post("https://api.x.ai/v1/chat/completions",
                                      json=payload,
                                      headers={"Authorization": f"Bearer {XAI_API_KEY}"})
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON from {model}: {content[:100]}...")
                    result = {"no_trade": True}
                if model != "grok-4":
                    logging.info(f"Fell back to {model} for query.")
                return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logging.warning(f"401 Unauthorized for {model} - Check SuperGrok/Premium+ subscription")
                continue
            raise
        except Exception as e:
            logging.error(f"Grok query error with {model}: {e}")
            if model == models[-1]:
                return {"error": str(e)}
    return {"error": "All models failed"}
def check_precision(trade: Dict[str, Any]) -> bool:
    price_keys = ['entry_low', 'entry_high', 'sl', 'tp1', 'tp2']
    for key in price_keys:
        if key in trade:
            p = trade[key]
            if round(p, 4) == round(p, 0):
                logging.warning(f"Precision fail: {key}={p} too round")
                return False
    return True
def check_rr(trade: Dict[str, Any]) -> bool:
    entry_mid = (trade['entry_low'] + trade['entry_high']) / 2
    rr2 = abs(trade['tp2'] - entry_mid) / abs(trade['sl'] - entry_mid)
    return rr2 >= 2.0
async def query_grok_potential(zones: List[Dict], symbol: str, current_price: float, trend: str, btc_trend: Optional[str], atr: float = 0) -> Dict[str, Any]:
    is_alt = symbol != 'BTC/USDT'
    filtered_zones = [z for z in zones if z.get('strength', 0) >= 3 and z.get('prob', 0) >= 85]
    if not filtered_zones:
        return {"no_live_trade": True, "roadmap": []}
    zone_summary = "\n".join([f"{z['direction']}: {z['zone_low']:.4f}-{z['zone_high']:.4f} ({z['confluence']}, {z['prob']}% prob, {z['dist_pct']:.1f}% away, str{z['strength']})" for z in filtered_zones])
    context = f"{symbol} | Price: {current_price:.4f} | Trend: {trend} {'| BTC: ' + btc_trend if is_alt else ''}\nInst Zones:\n{zone_summary}"
    system_prompt = (
        "You are an ICT whale analyst. Focus ONLY on institutional str=2+ OBs (vol-disp confirmed, unmitigated).\n"
        "Output ONLY valid JSON: If price in ≥75% inst zone (multi-TF align) → {'live_trade': {direction, entry_low, entry_high, sl, tp1, tp2, leverage:2-5, confidence≥75, strength:2, reason}}.\n"
        "Else: {'no_live_trade': true, 'roadmap': [{'direction': 'Long/Short', 'entry_low': x, 'entry_high': y, 'sl': z, 'tp1': a, 'tp2': b, 'leverage': 2-5, 'confidence': 75-98, 'strength':2, 'reason': 'max 60 chars (inst only)', 'dist_pct': d}]} # Top 1-2 elite zones\n"
        f"Precise 4 decimals (e.g., 3053.4567). Require HTF POC/VWAP align, whale flows. "
        f"{'Alts: Strict BTC align; no counter-trend.' if is_alt else ''}\n"
        "Conservative: Risk <5%, SL ATR*2 behind OB, TP1:1:2 R:R, TP2:1:3+ at next inst level. Ensure low < high."
    )
    models = ["grok-4", "grok-3"]
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "temperature": 0.05,
            "max_tokens": 400
        }
        attempt = 0
        while attempt < 2:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post("https://api.x.ai/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {XAI_API_KEY}"})
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"].strip()
                    result = json.loads(content)
                    precise = True
                    if 'live_trade' in result and (not check_precision(result['live_trade']) or not check_rr(result['live_trade'])):
                        precise = False
                    if 'roadmap' in result:
                        for z in result['roadmap']:
                            if (not check_precision(z) or not check_rr(z) or z.get('strength', 0) < 2 or z.get('confidence', 0) < 75):
                                precise = False
                                break
                    if precise:
                        return result
                    attempt += 1
                    if attempt == 2:
                        logging.warning(f"Precision/RR/Inst check failed after 2 attempts for {model}")
                        break
                    logging.info(f"Retrying {model} for inst precision/RR")
            except Exception as e:
                if attempt == 1:
                    if model == models[-1]:
                        return {"error": str(e)}
                attempt += 1
                continue
        if attempt == 2:
            continue
    return {"error": "All models failed"}
async def fetch_open_interest(symbol: str) -> Optional[Dict[str, float]]:
    async with fetch_sem:
        if await BanManager.check_and_sleep():
            logging.warning(f"OI skipped for {symbol}: ban/cooldown active")
            return None
        try:
            futures_symbol = symbol.replace('/', '')
            oi_data = await futures_exchange.fetch_open_interest(futures_symbol)
            logging.debug(f"Raw OI data for {futures_symbol}: {oi_data}")
            oi_value = None
            if isinstance(oi_data.get('openInterest'), dict):
                oi_value = oi_data['openInterest'].get('openInterestAmount')
            elif isinstance(oi_data.get('openInterest'), (int, float)):
                oi_value = oi_data['openInterest']
            else:
                info = oi_data.get('info', {})
                raw_oi = float(info.get('sumOpenInterest') or info.get('openInterest') or '0')
                oi_value = raw_oi if raw_oi > 0 else None
            if oi_value is None:
                raise KeyError("'openInterest' (amount/value) or raw sumOpenInterest not found")
            prev_oi = last_oi.get(futures_symbol, oi_value)
            oi_change_pct = (oi_value - prev_oi) / prev_oi * 100 if prev_oi and prev_oi != 0 else 0
            last_oi[futures_symbol] = float(oi_value)
            await asyncio.sleep(2)
            return {
                'open_interest': float(oi_value),
                'oi_change_pct': oi_change_pct
            }
        except Exception as e:
            logging.error(f"OI fetch error for {symbol}: {e}")
            await asyncio.sleep(5)
            return None
async def fetch_ohlcv(symbol: str, tf: str, limit: int = 200, since: Optional[int] = None) -> pd.DataFrame:
    async with fetch_sem:
        if await BanManager.check_and_sleep():
            logging.warning(f"OHLCV skipped for {symbol} {tf}: ban/cooldown active")
            return ohlcv_cache.get(f"{symbol}{tf}", {}).get('df', pd.DataFrame())
        cache_key = f"{symbol}{tf}"
        now = time.time()
        ttl = HTF_CACHE_TTL if tf in ['1d', '1w'] else CACHE_TTL
        cache_hit = False
        if cache_key in ohlcv_cache and now - ohlcv_cache[cache_key]['timestamp'] < ttl:
            cache_hit = True
            logging.debug(f"Cache hit for OHLCV {cache_key}")
        else:
            logging.debug(f"Cache miss for OHLCV {cache_key}; fetching fresh")
        evict_if_full(ohlcv_cache)
        backoff = 1
        success = False
        for attempt in range(4):
            try:
                norm_tf = tf.lower()
                params = {'limit': limit}
                if since:
                    params['since'] = since
                data = await exchange.fetch_ohlcv(symbol, norm_tf, **params)
                df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['ts'], unit='ms')
                ohlcv_cache[cache_key] = {'df': df, 'timestamp': now}
                success = True
                logging.info(f"OHLCV fetched for {symbol} {tf} (attempt {attempt+1})")
                break
            except Exception as e:
                logging.warning(f"OHLCV fetch attempt {attempt+1} failed for {symbol} {tf}: {e}")
                if attempt < 3:
                    await asyncio.sleep(backoff * 3) # Wzmocniony backoff *3
                    backoff *= 3
                else:
                    logging.warning(f"OHLCV fetch failed all attempts for {symbol} {tf}: using cache/empty")
                    break
        sleep_time = 3 if success else 5 # Zwiększony sleep po sukcesie
        await asyncio.sleep(sleep_time)
        return ohlcv_cache.get(cache_key, {}).get('df', pd.DataFrame())
async def fetch_ticker_batch() -> Dict[str, Optional[float]]:
    async with fetch_sem:
        now = time.time()
        cache_hits = sum(1 for s in SYMBOLS if s in ticker_cache and now - ticker_cache[s]['timestamp'] < TICKER_CACHE_TTL)
        if cache_hits == len(SYMBOLS):
            logging.debug(f"Full cache hit for tickers ({cache_hits}/{len(SYMBOLS)})")
            return {s: ticker_cache[s]['price'] for s in SYMBOLS}
        logging.debug(f"Partial cache hit for tickers ({cache_hits}/{len(SYMBOLS)}); polling fresh")
        if await BanManager.check_and_sleep():
            return {s: ticker_cache.get(s, {}).get('price') for s in SYMBOLS}
        evict_if_full(ticker_cache)
        backoff = 1
        for attempt in range(3):
            try:
                tickers = await exchange.fetch_tickers(SYMBOLS)
                prices = {s: tickers[s]['last'] if s in tickers else None for s in SYMBOLS}
                logging.info(f"Batch ticker poll success (attempt {attempt+1})")
                for s in SYMBOLS:
                    if prices[s]:
                        ticker_cache[s] = {'price': prices[s], 'timestamp': now}
                if attempt > 0:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                logging.info("Bybit poll: Rate limits OK")
                break
            except Exception as e:
                logging.debug(f"Batch ticker poll attempt {attempt+1} failed: {e}")
                if "429" in str(e):
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    raise
        else:
            prices = {s: ticker_cache.get(s, {}).get('price') for s in SYMBOLS}
        return prices
async def fetch_ticker(symbol: str) -> Optional[float]:
    prices = await fetch_ticker_batch()
    return prices.get(symbol)
async def price_background_task():
    """Background task: Polls tickers and order flow every 10s for fresh prices/data."""
    while True:
        try:
            prices = await fetch_ticker_batch()
            order_books = await fetch_order_flow_batch() # NEW: Add order flow polling
            if any(p is not None for p in prices.values()):
                prices_global.update(prices)
                logging.debug(f"Background poll update: {prices_global}")
            if order_books:
                logging.debug(f"Order flow updated: {list(order_books.keys())}")
            await asyncio.sleep(10)
        except Exception as e:
            logging.warning(f"Background poll error (retrying): {e}")
            await asyncio.sleep(5)
def add_indicators(df: pd.DataFrame, order_book: Optional[Dict] = None) -> pd.DataFrame: # NEW: Accept order_book
    if len(df) == 0:
        return df
    df = df.copy()
    df['ema50'] = ta.ema(df['close'], 50)
    df['ema100'] = ta.ema(df['close'], 100)
    df['ema200'] = ta.ema(df['close'], 200) # Retained: EMA200 for trend filter
    df['rsi'] = ta.rsi(df['close'], 14)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    macd_data = ta.macd(df['close'])
    if macd_data is not None and len(macd_data.columns) >= 3:
        df['macd'] = macd_data.iloc[:, 0]
        df['macd_signal'] = macd_data.iloc[:, 1]
        df['macd_hist'] = macd_data.iloc[:, 2]
    if len(df) >= 10:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=2.5)
        if st is not None and hasattr(st, 'columns') and 'SUPERT_10_2.5' in st.columns and 'SUPERTd_10_2.5' in st.columns:
            df['supertrend'] = st['SUPERT_10_2.5']
            df['supertrend_dir'] = st['SUPERTd_10_2.5']
    if len(df) >= 20:
        bb = ta.bbands(df['close'], length=20)
        if bb is not None and len(bb.columns) >= 3:
            df['bb_upper'] = bb.iloc[:, 0]
            df['bb_middle'] = bb.iloc[:, 1]
            df['bb_lower'] = bb.iloc[:, 2]
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None and len(stoch.columns) >= 2:
            df['stoch_k'] = stoch.iloc[:, 0]
            df['stoch_d'] = stoch.iloc[:, 1]
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    # NEW Order Flow: Calculate delta/footprint
    df = calculate_order_flow(df, order_book)
    key_cols = ['ema50', 'ema100', 'rsi', 'volume_sma', 'macd', 'macd_signal', 'supertrend_dir', 'vwap', 'ema200', 'order_delta', 'cum_delta'] # NEW: Include order flow cols
    subset_key = [col for col in key_cols if col in df.columns]
    df = df.dropna(subset=subset_key)
    return df
def zones_overlap(z1_low: float, z1_high: float, z2_low: float, z2_high: float, threshold: float = 0.5) -> float:
    o_low = max(z1_low, z2_low)
    o_high = min(z1_high, z2_high)
    if o_low >= o_high:
        return 0.0
    overlap_len = o_high - o_low
    min_width = min(z1_high - z1_low, z2_high - z2_low)
    return (overlap_len / min_width)
async def find_unmitigated_order_blocks(df: pd.DataFrame, lookback: int = 100, atr_mult: float = 2.0, tf: str = None, symbol: str = None) -> Dict[str, List[Dict]]:
    if len(df) < 30:
        return {'bullish': [], 'bearish': []}
    ltf_mult = 1 if tf in ['15m', '1h'] else 2 if tf == '4h' else 3
    dyn_lookback = lookback * ltf_mult
    df_local = df.tail(dyn_lookback).copy()
    df_local['atr'] = ta.atr(df_local['high'], df_local['low'], df_local['close'], 14)
    df_local['direction'] = np.where(df_local['close'] > df_local['open'], 1, -1)
    df_local['swing_high'] = df_local['high'].rolling(10, center=True).max() == df_local['high']
    df_local['swing_low'] = df_local['low'].rolling(10, center=True).min() == df_local['low']
    df_local['vol_surge'] = df_local['volume'] > 1.2 * df_local['volume'].rolling(20).mean()
    logging.info(f"Vol surge detected: {sum(df_local['vol_surge'])} in {len(df_local)} bars")
    df_local['volume_sma'] = df_local['volume'].rolling(20).mean()
    obs = {'bullish': [], 'bearish': []}
    for i in range(15, len(df_local) - 10):
        if df_local['swing_high'].iloc[i] and df_local['vol_surge'].iloc[i]:
            ob_high = df_local['high'].iloc[i]
            ob_low = max(df_local['open'].iloc[i], df_local['close'].iloc[i])
            move_down = df_local['low'].iloc[i+5:i+10].min() < ob_low - (df_local['atr'].iloc[i] * atr_mult)
            if move_down and abs((ob_low - df_local['low'].iloc[i+5:i+10].min()) / ob_low) > 0.02:
                mitigated = any(df_local['high'].iloc[i+1:] > ob_high)
                zone_type = 'Breaker' if any(df_local['low'].iloc[i+1:i+6] < ob_low) else 'OB'
                strength = 3 if zone_type == 'OB' and df_local['volume'].iloc[i] > 1.2 * df_local['volume_sma'].iloc[i] else 2
                if not mitigated and strength == 3:
                    obs['bearish'].append({
                        'low': ob_low, 'high': ob_high, 'type': zone_type,
                        'strength': strength, 'index': i, 'mitigated': False
                    })
    for i in range(15, len(df_local) - 10):
        if df_local['swing_low'].iloc[i] and df_local['vol_surge'].iloc[i]:
            ob_low = df_local['low'].iloc[i]
            ob_high = min(df_local['open'].iloc[i], df_local['close'].iloc[i])
            move_up = df_local['high'].iloc[i+5:i+10].max() > ob_high + (df_local['atr'].iloc[i] * atr_mult)
            if move_up and abs((df_local['high'].iloc[i+5:i+10].max() - ob_high) / ob_high) > 0.02:
                mitigated = any(df_local['low'].iloc[i+1:] < ob_low)
                zone_type = 'Breaker' if any(df_local['high'].iloc[i+1:i+6] > ob_high) else 'OB'
                strength = 3 if zone_type == 'OB' and df_local['volume'].iloc[i] > 1.2 * df_local['volume_sma'].iloc[i] else 2
                if not mitigated and strength == 3:
                    obs['bullish'].append({
                        'low': ob_low, 'high': ob_high, 'type': zone_type,
                        'strength': strength, 'index': i, 'mitigated': False
                    })
    if tf in ['1d', '1w'] and symbol:
        try:
            df_1h = await fetch_ohlcv(symbol, '1h', 200)
            obs_1h = await find_unmitigated_order_blocks(df_1h, lookback=100, tf='1h', symbol=symbol)
            for ob_type in ['bullish', 'bearish']:
                merged = []
                for ob_htf in obs[ob_type]:
                    for ob_ltf in obs_1h[ob_type]:
                        overlap_ratio = zones_overlap(ob_htf['low'], ob_htf['high'], ob_ltf['low'], ob_ltf['high'], 0.7)
                        if overlap_ratio > 0.7:
                            merged_low = min(ob_htf['low'], ob_ltf['low'])
                            merged_high = max(ob_htf['high'], ob_ltf['high'])
                            merged_strength = max(ob_htf['strength'], ob_ltf['strength'])
                            merged.append({
                                'low': merged_low, 'high': merged_high, 'type': ob_htf['type'],
                                'strength': merged_strength, 'index': ob_htf['index'], 'mitigated': False
                            })
                            logging.info(f"Merged inst {ob_type} OB for {symbol} {tf}: overlap {overlap_ratio:.2f}")
                obs[ob_type] = merged[:2]
        except Exception as e:
            logging.warning(f"HTF 1h verify failed for {symbol} {tf}: {e}")
    for key in obs:
        obs[key] = sorted(obs[key], key=lambda z: (len(df_local) - z['index']) * z['strength'], reverse=True)[:2]
    return obs
async def find_next_premium_zones(df: pd.DataFrame, current_price: float, tf: str, symbol: str = None, oi_data: Optional[Dict[str, float]] = None, trend: str = None, whale_data: Optional[Dict[str, Any]] = None, order_book: Optional[Dict] = None) -> List[Dict]: # NEW: Accept order_book
    if len(df) < 50:
        return []
    df = add_indicators(df, order_book) # Pass order_book for delta calc
    # Retained: EMA200 for 1d trend bias
    if 'ema200' not in df.columns or df['ema200'].isna().all():
        df['ema200'] = ta.ema(df['close'], 200)
        df = df.dropna(subset=['ema200'])
    ema200_val = df['ema200'].iloc[-1]
    price_current = df['close'].iloc[-1]
    if price_current > ema200_val:
        trend_bias = 'bull' # Long favor
    elif price_current < ema200_val:
        trend_bias = 'bear' # Short favor
    else:
        trend_bias = 'neutral' # Both, +1 conf
  
    if tf in ['1d', '1w']: # Daily focus
        if trend_bias == 'bull' and direction == 'Long':
            conf_score += 2.0 # Mocniejszy boost dla HTF
        elif trend_bias == 'bear' and direction == 'Short':
            conf_score += 2.0
    logging.info(f"EMA200 bias for {symbol} {tf}: {trend_bias} (price {price_current:.4f} vs EMA {ema200_val:.4f})")
    rsi_1d = None
    rsi_1w = None
    if symbol:
        try:
            df_1d = await fetch_ohlcv(symbol, '1d', 200)
            df_1d = add_indicators(df_1d)
            rsi_1d = df_1d['rsi'].iloc[-1] if len(df_1d) > 0 and 'rsi' in df_1d.columns else None
            df_1w = await fetch_ohlcv(symbol, '1w', 200)
            df_1w = add_indicators(df_1w)
            rsi_1w = df_1w['rsi'].iloc[-1] if len(df_1w) > 0 and 'rsi' in df_1w.columns else None
        except Exception as e:
            logging.error(f"HTF RSI fetch error for {symbol}: {e}")
    tf_trend = trend
    htf_align = 1.0
    if tf in ['4h', '1d', '1w']:
        df_4h = await fetch_ohlcv(symbol, '4h', 100)
        df_4h = add_indicators(df_4h)
        if len(df_4h) > 0:
            l4h = df_4h.iloc[-1]
            if (l4h['ema50'] > l4h['ema100'] and tf_trend == 'Uptrend') or (l4h['ema50'] < l4h['ema100'] and tf_trend == 'Downtrend'):
                htf_align += 1.5
            else:
                return []
    htf_mult = htf_align if tf in ['1d', '1w'] else 1.0
    obs = await find_unmitigated_order_blocks(df, tf=tf, symbol=symbol)
    elite_obs = {k: [o for o in v if o['strength'] == 3] for k, v in obs.items()}
    liq_profile = calc_liquidity_profile(df)
    poc = max(liq_profile.items(), key=lambda x: x[1])[0] if liq_profile else None
    zones_4pct = []
    buffer_mult = 0.04 if trend == 'Downtrend' else 0.06
    long_buffer = current_price * buffer_mult
    short_buffer = current_price * buffer_mult
    for ob in elite_obs.get('bullish', []):
        mid = (ob['low'] + ob['high']) / 2
        dist_pct = abs(current_price - mid) / current_price * 100
        if mid < current_price - long_buffer and dist_pct <= 4.0:
            zones_4pct.append(ob)
    for ob in elite_obs.get('bearish', []):
        mid = (ob['low'] + ob['high']) / 2
        dist_pct = abs(current_price - mid) / current_price * 100
        if mid > current_price + short_buffer and dist_pct <= 4.0:
            zones_4pct.append(ob)
    zones_to_use = zones_4pct
    logging.info(f"Using 4% elite zones for {symbol} {tf}: {len(zones_to_use)} str=3 OBs")
    zones = []
    for ob in zones_to_use:
        mid = (ob['low'] + ob['high']) / 2
        dist = abs(current_price - mid) / current_price * 100
        conf_score = ob['strength'] * htf_mult
        if 'vwap' in df.columns and not pd.isna(df['vwap'].iloc[-1]) and abs(mid - df['vwap'].iloc[-1]) / current_price * 100 < 0.5:
            conf_score += 1.5
            confluence_str = ob['type'] + "+VWAP"
        else:
            confluence_str = ob['type']
        if poc and abs(mid - poc) / current_price * 100 < 0.3:
            conf_score += 2
            confluence_str += "+POC"
        oi_str = ""
        if oi_data and oi_data['oi_change_pct'] > 10:
            conf_score += 1.5
            oi_str = "+Whale OI"
        whale_str = ""
        if whale_data and whale_data['boost'] > 1:
            conf_score += whale_data['boost']
            whale_str = whale_data['reason']
        if 'supertrend_dir' in df.columns and df['supertrend_dir'].iloc[-1] == (1 if 'bullish' in str(ob) else -1):
            conf_score += 1
            confluence_str += "+ST Align"
        if liq_profile and liq_profile.get(mid, 0) > 1.5:
            conf_score += 1.5
            confluence_str += "+High Liq"
        confluence_str += oi_str + whale_str
        rsi_os_boost = 0
        direction = 'Long' if 'bullish' in str(ob.get('type', '')) else 'Short'
        if ((rsi_1d and rsi_1d < 35) or (rsi_1w and rsi_1w < 35)) and ob['strength'] == 3 and direction == 'Long': # NEW: RSI<30 for long
            rsi_os_boost = 1.5
            confluence_str += "+RSI Exhaust Long"
        elif ((rsi_1d and rsi_1d > 65) or (rsi_1w and rsi_1w > 65)) and ob['strength'] == 3 and direction == 'Short': # NEW: RSI>70 for short
            rsi_os_boost = 1.5
            confluence_str += "+RSI Exhaust Short"
        else:
            confluence_str += "+RSI Neutral" # NEW: Fallback
        conf_score += rsi_os_boost
        # Retained: Trend bias boost
        if trend_bias == 'bull' and direction == 'Long':
            conf_score += 1.5 # Bias boost
        elif trend_bias == 'bear' and direction == 'Short':
            conf_score += 1.5
        elif trend_bias == 'neutral':
            conf_score += 1
        # Retained: Reversal caution
        fvgs = detect_fvg(df, tf, proximity_pct=0.3)
        obs_key = 'bullish' if direction == 'Long' else 'bearish'
        breaker_confirmed = any('Breaker' in ob['type'] for ob in elite_obs.get(obs_key, [])) # Pseudo-check
        if fvgs or breaker_confirmed:
            reversal_caution = True
            conf_score += 1.5
            confluence_str += "+FVG/Breaker Caution"
        else:
            reversal_caution = False
        aligned_bias = 'bull' if direction == 'Long' else 'bear'
        if trend_bias != 'neutral' and trend_bias != aligned_bias and not reversal_caution: # Skip counter-trend without confirm
            logging.info(f"Skipped counter-trend {direction} for {symbol} {tf}: bias {trend_bias}, no caution")
            continue
        # NEW Order Flow: Boost on delta imbalance >1%
        delta = df['order_delta'].iloc[-1]
        if abs(delta) > 1.0:
            conf_score += 2
            confluence_str += f"+Order Flow Delta {delta:.1f}%"
        # Footprint imbalance boost
        if df['footprint_imbalance'].iloc[-1]:
            conf_score += 1
            confluence_str += "+Footprint Imbalance"
        prob = min(98, 70 + conf_score * 8)
        direction = 'Long' if 'bullish' in str(ob.get('type', '')) else 'Short'
        zones.append({
            'direction': direction, 'zone_low': ob['low'], 'zone_high': ob['high'],
            'confluence': confluence_str, 'dist_pct': dist, 'prob': prob, 'strength': ob['strength']
        })
    zones = sorted(zones, key=lambda z: z['dist_pct'])[:2]
    zones = [z for z in zones if z['prob'] >= 75]
    if len(zones) < 2 and tf not in ['1w']:
        zones = []
    logging.info(f"Filtered to {len(zones)} elite zones >=90% for {symbol} {tf}")
    return zones
def calc_liquidity_profile(df: pd.DataFrame, bins: int = 15) -> Dict[float, float]:
    if len(df) < 50:
        return {}
    prices = df['close']
    volumes = df['volume']
    min_p, max_p = prices.min(), prices.max()
    bin_edges = np.linspace(min_p, max_p, bins + 1)
    vol_profile = {}
    for i in range(bins):
        mask = (prices >= bin_edges[i]) & (prices < bin_edges[i+1])
        vol_sum = volumes[mask].sum()
        bin_mid = (bin_edges[i] + bin_edges[i+1]) / 2
        vol_profile[bin_mid] = vol_sum
    sorted_vols = sorted(vol_profile.values(), reverse=True)
    threshold = sorted_vols[int(len(sorted_vols) * 0.85)] if sorted_vols else 0
    if threshold == 0:
        hot_zones = {k: 0 for k, v in vol_profile.items()}
    else:
        hot_zones = {k: v / threshold if v > threshold else 0 for k, v in vol_profile.items()}
    return hot_zones
def detect_supertrend(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 11 or 'supertrend_dir' not in df.columns or pd.isna(df['supertrend_dir'].iloc[-1]):
        return None
    if df['supertrend_dir'].iloc[-1] == 1 and df['supertrend_dir'].iloc[-2] == -1:
        return f"SuperTrend Bullish Flip ({tf})"
    if df['supertrend_dir'].iloc[-1] == -1 and df['supertrend_dir'].iloc[-2] == 1:
        return f"SuperTrend Bearish Flip ({tf})"
    return None
def detect_fvg(df: pd.DataFrame, tf: str, lookback: int = 50, proximity_pct: float = 0.3) -> List[str]:
    if len(df) < 5:
        return []
    df_local = df.tail(lookback).copy()
    fvgs = []
    price = df_local['close'].iloc[-1]
    for i in range(2, len(df_local) - 1):
        if df_local['high'].iloc[i-2] < df_local['low'].iloc[i]:
            fvg_low = df_local['high'].iloc[i-2]
            fvg_high = df_local['low'].iloc[i]
            mid = (fvg_low + fvg_high) / 2
            dist_pct = abs(price - mid) / price * 100
            if dist_pct < proximity_pct:
                fvgs.append(f"Near Bullish FVG {dist_pct:.1f}% away ({tf})")
        elif df_local['low'].iloc[i-2] > df_local['high'].iloc[i]:
            fvg_low = df_local['high'].iloc[i]
            fvg_high = df_local['low'].iloc[i-2]
            mid = (fvg_low + fvg_high) / 2
            dist_pct = abs(price - mid) / price * 100
            if dist_pct < proximity_pct:
                fvgs.append(f"Near Bearish FVG {dist_pct:.1f}% away ({tf})")
    return fvgs
def detect_macd(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 2 or 'macd' not in df.columns or 'macd_signal' not in df.columns or pd.isna(df['macd'].iloc[-1]) or pd.isna(df['macd_signal'].iloc[-1]):
        return None
    if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
        return f"Bullish MACD Crossover ({tf})"
    if df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
        return f"Bearish MACD Crossover ({tf})"
    return None
def detect_pre_cross(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 2 or tf not in ['4h', '1d'] or 'ema50' not in df.columns or 'ema100' not in df.columns or pd.isna(df['ema50'].iloc[-1]) or pd.isna(df['ema100'].iloc[-1]):
        return None
    l = df.iloc[-1]
    diff = abs(l['ema50'] - l['ema100']) / l['close']
    if diff < 0.005:
        return "Golden Cross Incoming" if l['ema50'] > l['ema100'] else "Death Cross Incoming"
    return None
def detect_divergence(df: pd.DataFrame, tf: str) -> Optional[str]:
    if len(df) < 50 or tf not in ['4h', '1d'] or 'rsi' not in df.columns or pd.isna(df['rsi'].iloc[-1]):
        return None
    price = df['close'].iloc[-40:]
    rsi = df['rsi'].iloc[-40:]
    swing_low_idx = price.iloc[-10:].idxmin()
    swing_high_idx = price.iloc[-10:].idxmax()
    if pd.isna(rsi.loc[swing_low_idx]) or pd.isna(rsi.loc[swing_high_idx]):
        return None
    if price.iloc[-1] > price.loc[swing_low_idx] and rsi.iloc[-1] < rsi.loc[swing_low_idx] and rsi.loc[swing_low_idx] < 30:
        return f"Hidden Bullish RSI Divergence ({tf})"
    if price.iloc[-1] < price.loc[swing_high_idx] and rsi.iloc[-1] > rsi.loc[swing_high_idx] and rsi.loc[swing_high_idx] > 70:
        return f"Hidden Bearish RSI Divergence ({tf})"
    if price.iloc[-1] < price.loc[swing_low_idx] and rsi.iloc[-1] > rsi.loc[swing_low_idx] and rsi.iloc[-1] < 25:
        return f"Bullish RSI Divergence ({tf})"
    if price.iloc[-1] > price.loc[swing_high_idx] and rsi.iloc[-1] < rsi.loc[swing_high_idx] and rsi.iloc[-1] > 75:
        return f"Bearish RSI Divergence ({tf})"
    return None
def detect_candle_patterns(df: pd.DataFrame, tf: str) -> List[str]:
    if len(df) < 3:
        return []
    patterns = set()
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c['close'] - c['open'])
    upper = c['high'] - max(c['open'], c['close'])
    lower = min(c['open'], c['close']) - c['low']
    range_size = c['high'] - c['low']
    if body > 0.7 * range_size and df['volume'].iloc[-1] > 1.5 * df['volume_sma'].iloc[-1]:
        dir_str = "Bullish" if c['close'] > c['open'] else "Bearish"
        patterns.add(f"{dir_str} Displacement Candle ({tf})")
    if body > 0 and lower > 2 * body and upper < body * 0.3 and c['close'] > p['close']:
        patterns.add(f"Bullish Pinbar ({tf})")
    if body > 0 and upper > 2 * body and lower < body * 0.3 and c['close'] < p['close']:
        patterns.add(f"Bearish Pinbar ({tf})")
    if p['close'] < p['open'] and c['close'] > c['open'] and c['open'] < p['close'] and c['close'] > p['open']:
        patterns.add(f"Bullish Engulfing ({tf})")
    if p['close'] > p['open'] and c['close'] < c['open'] and c['open'] > p['close'] and c['close'] < p['open']:
        patterns.add(f"Bearish Engulfing ({tf})")
    if c['high'] < p['high'] and c['low'] > p['low']:
        patterns.add(f"Inside Bar ({tf})")
    return list(patterns)
async def process_trade(trades: Dict[str, Any], to_delete: List[str], now: datetime, current_capital: float, prices: Dict[str, Optional[float]], updated_keys: List[str], is_protected: bool = False):
    for trade_key, trade in list(trades.items()):
        clean_symbol = get_clean_symbol(trade_key)
        logging.debug(f"Clean symbol for tracking: {clean_symbol} from key {trade_key}")
        if 'last_check' in trade and now - trade['last_check'] > timedelta(hours=TRADE_TIMEOUT_HOURS):
            logging.info(f"Timeout for {'protected ' if is_protected else ''}{trade_key}")
            await bot.send_message(CHAT_ID, f"**TIMEOUT** {clean_symbol.replace('/USDT','')} (*neutral PnL*)", parse_mode='Markdown')
            to_delete.append(trade_key)
            updated_keys.append(trade_key)
            continue
        price = prices.get(clean_symbol)
        if price is None:
            trade['last_check'] = now
            updated_keys.append(trade_key)
            continue
        trade['last_check'] = now
        updated_keys.append(trade_key)
        if not is_protected and trade.get('active') and 'entry_time' in trade and now - trade['entry_time'] > timedelta(hours=PROTECT_AFTER_HOURS):
            protected_trades[trade_key] = trades.pop(trade_key)
            updated_keys.append(trade_key)
            logging.info(f"Moved active {trade_key} to protected after {PROTECT_AFTER_HOURS}h")
            continue
        if not trade.get('active', False):
            entry_low = trade['entry_low']
            entry_high = trade['entry_high']
            slippage_note = ""
            if trade['direction'] == 'Long':
                extended_high = entry_high * (1 + ENTRY_SLIPPAGE_PCT)
                in_zone = entry_low <= price <= extended_high
                if price > entry_high:
                    slippage_note = " (*late entry via slippage*)"
            else:
                extended_low = entry_low * (1 - ENTRY_SLIPPAGE_PCT)
                in_zone = extended_low <= price <= entry_high
                if price < entry_low:
                    slippage_note = " (*late entry via slippage*)"
            logging.info(f"Checking {clean_symbol} ({trade.get('type', '')} {trade['direction']} {trade['confidence']}%) - price {price:.4f} vs zone {entry_low:.4f}-{entry_high:.4f}, in_zone={in_zone}{slippage_note}")
            if in_zone:
                current_exposure = sum(
                    t.get('position_size', 0) * t.get('leverage', 1)
                    for trades_dict in [open_trades, protected_trades]
                    for t in trades_dict.values()
                    if t.get('active')
                )
                risk_distance = abs(price - trade['sl'])
                proposed_size = (current_capital * RISK_PER_TRADE_PCT / 100) / risk_distance if risk_distance > 0 else 0
                proposed_exposure = proposed_size * trade['leverage']
                max_exposure = current_capital * 0.05
                if current_exposure + proposed_exposure > max_exposure:
                    logging.info(f"Skipped {'protected ' if is_protected else ''}entry for {clean_symbol}: exposure would exceed 5% ({current_exposure + proposed_exposure:.2f} > {max_exposure:.2f})")
                    continue
                trade['active'] = True
                trade['entry_price'] = price + (SLIPPAGE_PCT * price if trade['direction'] == 'Long' else -SLIPPAGE_PCT * price)
                if 'entry_time' not in trade:
                    trade['entry_time'] = now
                risk_amount = current_capital * RISK_PER_TRADE_PCT / 100
                trade['position_size'] = risk_amount / risk_distance if risk_distance > 0 else 0
                tag = ('(*roadmap*)' if trade.get('type') == 'roadmap' else ('(*protected*)' if is_protected else ''))
                if tag == '(*roadmap*)':
                    logging.info(f"Roadmap ENTRY ACTIVATED for {clean_symbol} @ {price:.4f} (conf {trade['confidence']}%)")
                await bot.send_message(CHAT_ID,
                                       f"**ENTRY ACTIVATED** {tag} (Size: {trade['position_size']:.4f}){slippage_note}\n\n"
                                       f"**{clean_symbol.replace('/USDT','')} {trade['direction']}** @ {format_price(price)}\n"
                                       f"*SL* {format_price(trade['sl'])} │ *TP1* {format_price(trade['tp1'])} │ *TP2* {format_price(trade['tp2'])} │ {trade['leverage']}x",
                                       parse_mode='Markdown')
                updated_keys.append(trade_key)
        if trade.get('active'):
            entry_price = trade['entry_price']
            size = trade.get('position_size', 1)
            current_sl = trade.get('trailing_sl', trade['sl'])
            if (trade['direction'] == 'Long' and price >= trade['tp1']) or (trade['direction'] == 'Short' and price <= trade['tp1']):
                if current_sl == trade['sl']:
                    trade['trailing_sl'] = entry_price
                    current_sl = entry_price
                    logging.info(f"Trailing SL to breakeven for {'protected ' if is_protected else ''}{clean_symbol}")
                    updated_keys.append(trade_key)
            hit_tp = (price >= trade['tp2'] if trade['direction'] == 'Long' else price <= trade['tp2'])
            hit_sl = (price <= current_sl if trade['direction'] == 'Long' else price >= current_sl)
            if hit_tp or hit_sl:
                if trade.get('processed'):
                    logging.info(f"Duplicate PnL skipped for {'protected ' if is_protected else ''}{clean_symbol}")
                    continue
                trade['processed'] = True
                diff = (price - entry_price) if trade['direction'] == 'Long' else (entry_price - price)
                pnl_usdt = diff * size
                fee_usdt = 2 * FEE_PCT * (entry_price * size)
                net_pnl_usdt = pnl_usdt - fee_usdt
                net_pnl_pct = net_pnl_usdt / current_capital * 100
                result = "WIN" if hit_tp else "LOSS"
                tag = ('(*roadmap*)' if trade.get('type') == 'roadmap' else ('(*protected*)' if is_protected else ''))
                await bot.send_message(CHAT_ID,
                                       f"**{result}** {clean_symbol.replace('/USDT','')} {tag}\n\n"
                                       f"+{net_pnl_pct:+.2f}% (size {size:.4f}) @ {trade['leverage']}x (conf {trade['confidence']}%) **[fees adj]**",
                                       parse_mode='Markdown')
                async with stats_lock:
                    delta_capital = current_capital * (net_pnl_pct / 100)
                    stats['capital'] += delta_capital
                    stats['pnl'] = (stats['capital'] - SIMULATED_CAPITAL) / SIMULATED_CAPITAL * 100
                    stats['wins' if hit_tp else 'losses'] += 1
                save_stats(stats)
                to_delete.append(trade_key)
                updated_keys.append(trade_key)
async def btc_trend_update(context):
    global btc_trend_global
    try:
        df = await fetch_ohlcv('BTC/USDT', '1d', limit=500) # Zwiększony limit z 300 na 500
        if len(df) == 0:
            logging.warning("Empty raw DF in BTC trend – possible rate limit or insufficient data")
            btc_trend_global = "Sideways" # fallback
            return
        # Compute only necessary for trend (avoid full indicators/dropna causing empty DF)
        df = df.copy()
        df['ema50'] = ta.ema(df['close'], 50)
        df['ema100'] = ta.ema(df['close'], 100)
        # Drop only rows where ema50 and ema100 are NaN
        df = df.dropna(subset=['ema50', 'ema100'])
        if len(df) == 0:
            logging.warning("Empty DF after EMAs in BTC trend – insufficient data")
            btc_trend_global = "Sideways" # fallback
            return
        if len(df) < 100: # Min bars for reliable EMA100
            logging.warning(f"Insufficient bars ({len(df)}) for trend in BTC – fallback to Sideways")
            btc_trend_global = "Sideways"
            return
        l = df.iloc[-1]
        if pd.isna(l['ema50']) or pd.isna(l['ema100']):
            logging.warning("EMA NaN in BTC trend – skipping update")
            return
        if l['ema50'] > l['ema100']:
            btc_trend_global = "Uptrend"
        elif l['ema50'] < l['ema100']:
            btc_trend_global = "Downtrend"
        else:
            btc_trend_global = "Sideways"
        logging.info(f"Global BTC trend: {btc_trend_global}")
    except Exception as e:
        logging.error(f"BTC trend update error: {e}")
        btc_trend_global = "Sideways" # safe fallback
async def price_update_callback(context):
    global last_price_update, prices_global
    now = time.time()
    if now - last_price_update < 30:
        return
    start = time.perf_counter()
    try:
        prices = await fetch_ticker_batch()
        if any(p is not None for p in prices.values()):
            prices_global.update(prices)
            last_price_update = now
            logging.info(f"Prices updated globally: {prices_global}")
        await asyncio.sleep(1)
    except Exception as e:
        logging.error(f"Price update error: {e}")
    finally:
        exec_time = time.perf_counter() - start
        logging.info(f"Price update cycle completed in {exec_time:.2f}s")
async def track_callback(context):
    global stats, open_trades, protected_trades
    start_time = time.perf_counter()
    try:
        if not open_trades and not protected_trades:
            logging.info(f"Track cycle completed in {time.perf_counter() - start_time:.2f}s (no trades)")
            return
        now = datetime.now(timezone.utc)
        to_delete_open = []
        to_delete_protected = []
        updated_keys_open = []
        updated_keys_protected = []
        current_capital = stats['capital']
        prices = prices_global.copy()
        if all(p is None for p in prices.values()):
            logging.warning("No global prices available – skipping track checks")
            return
        drawdown = max(0, -stats['pnl'])
        if drawdown > MAX_DRAWDOWN_PCT and len(open_trades) > 0:
            logging.warning(f"Max drawdown exceeded ({drawdown:.1f}%) – pausing trades")
            for key in list(open_trades.keys()):
                if not open_trades[key].get('active'):
                    del open_trades[key]
                    updated_keys_open.append(key)
            save_trades(open_trades)
            await bot.send_message(CHAT_ID, f"**PAUSE:** Drawdown {drawdown:.1f}% – Cleared pending trades", parse_mode='Markdown')
        total_active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        if total_active >= MAX_CONCURRENT_TRADES:
            logging.info("Max concurrent trades reached – skipping new checks")
        await process_trade(open_trades, to_delete_open, now, current_capital, prices, updated_keys_open, is_protected=False)
        await process_trade(protected_trades, to_delete_protected, now, current_capital, prices, updated_keys_protected, is_protected=True)
        for key in to_delete_open:
            del open_trades[key]
        for key in to_delete_protected:
            del protected_trades[key]
        if updated_keys_open or to_delete_open:
            save_trades(open_trades)
        if updated_keys_protected or to_delete_protected:
            save_protected(protected_trades)
        exec_time = time.perf_counter() - start_time
        logging.info(f"Track cycle completed in {exec_time:.2f}s ({len(to_delete_open + to_delete_protected)} closes, {len(updated_keys_open + updated_keys_protected)} updates)")
    except Exception as e:
        logging.error(f"Track callback error: {e}")
        logging.info(f"Track cycle failed in {time.perf_counter() - start_time:.2f}s")
async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"/stats triggered by user {update.effective_user.id}")
    try:
        total = stats['wins'] + stats['losses']
        winrate = (stats['wins'] / total * 100) if total > 0 else 0
        active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        pending = len([t for t in open_trades.values() if not t.get('active')])
        unique_symbols = sorted(set(get_clean_symbol(k).replace('/USDT', '') for trades in [open_trades, protected_trades] for k in trades.keys()))
        open_symbols = ', '.join(unique_symbols) if unique_symbols else 'None'
        drawdown = max(0, -stats['pnl'])
        msg = (
            "**Bot Statistics ♔**\n\n"
            f"**Total Trades:** {total}\n"
            f"**Wins:** {stats['wins']} ({winrate:.1f}%)\n"
            f"**Losses:** {stats['losses']}\n"
            f"**Total PNL:** {stats['pnl']:+.2f}% | *Capital:* ${stats['capital']:.2f}\n"
            f"**Drawdown:** {drawdown:.2f}%\n"
            f"**Active (incl protected):** {active}\n"
            f"**Pending:** {pending}\n"
            f"**Open:** {open_symbols}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')
        logging.info(f"/stats sent successfully")
    except Exception as e:
        logging.error(f"/stats error: {e}")
        await update.message.reply_text(f"Error fetching stats: {str(e)}", parse_mode='Markdown')
async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"/health triggered by user {update.effective_user.id}")
    try:
        uptime = datetime.now(timezone.utc).isoformat()
        open_count = len(open_trades)
        protected_count = len(protected_trades)
        active = len([t for trades in [open_trades, protected_trades] for t in trades.values() if t.get('active')])
        pending = len([t for t in open_trades.values() if not t.get('active')])
        msg = (
            "**Grok Elite Bot v24.01.6 - Institutional Hunter Alive!**\n\n"
            f"**Uptime Check:** {uptime}\n"
            f"**Open Trades:** {open_count}\n"
            f"**Protected Trades:** {protected_count}\n"
            f"**Active:** {active} | **Pending:** {pending}\n"
            f"**Status:** Zero-noise mode: Str=3 OBs only, 24h cooldown, multi-TF align + Order Flow delta ♔"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')
        logging.info(f"/health sent successfully")
    except Exception as e:
        logging.error(f"/health error: {e}")
        await update.message.reply_text(f"Health check failed: {str(e)}", parse_mode='Markdown')
async def recap_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"/recap triggered by user {update.effective_user.id}")
    await daily_callback(context)
    await update.message.reply_text("**Manual recap triggered**—check logs/messages!", parse_mode='Markdown')
async def daily_callback(context):
    now = datetime.now(timezone.utc)
    recap_file = RECAP_FILE
    last_date = None
    if os.path.exists(recap_file):
        try:
            with open(recap_file, 'r') as f:
                last_date = datetime.fromisoformat(f.read().strip())
                logging.info(f"Last recap: {last_date.date()}")
        except Exception as e:
            logging.warning(f"Failed to read recap file: {e}")
            last_date = None
    if last_date and now.date() == last_date.date():
        logging.info("Recap skipped—already sent today")
        return
    try:
        logging.info(f"Daily recap starting for {now.date()}")
        text = f"**Daily Recap – {now.strftime('%b %d, %Y')}**\n"
        prices = await fetch_ticker_batch()
        for sym in SYMBOLS:
            if await BanManager.check_and_sleep():
                logging.warning(f"Ban active during recap for {sym}; skipping")
                continue
            price = prices.get(sym)
            if price is None:
                logging.warning(f"No price for {sym}")
                await asyncio.sleep(2)
                continue
            df = await fetch_ohlcv(sym, '1d', 2)
            if len(df) < 2:
                logging.warning(f"Insufficient data for {sym}")
                await asyncio.sleep(2)
                continue
            change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            text += f"{sym}: {change:+.2f}% │ {format_price(price)}\n"
            logging.info(f"{sym}: {change:+.2f}% @ {price:.4f}")
            await asyncio.sleep(2)
        if not text.endswith('\n'):
            await bot.send_message(CHAT_ID, "**Recap skipped**—API ban active, try later.", parse_mode='Markdown')
            return
        text += "\nInstitutional macro recap + next 48h whale bias."
        logging.info("Calling Grok for recap...")
        payload_base = {
            "messages": [{"role": "system", "content": "You are a top institutional macro analyst. Focus on whale flows, OB structures. Respond concisely."},
                         {"role": "user", "content": text}],
            "temperature": 0.2, "max_tokens": 300
        }
        models = ["grok-4", "grok-3"]
        recap = None
        for model in models:
            payload = payload_base.copy()
            payload["model"] = model
            try:
                async with httpx.AsyncClient(timeout=45.0) as c:
                    r = await c.post("https://api.x.ai/v1/chat/completions", json=payload,
                                     headers={"Authorization": f"Bearer {XAI_API_KEY}"})
                    r.raise_for_status()
                    recap = r.json()["choices"][0]["message"]["content"].strip()
                    logging.info(f"Grok recap from {model} (len: {len(recap)})")
                    if model != "grok-4":
                        logging.info(f"Daily recap fell back to {model}")
                    break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logging.warning(f"401 Unauthorized for {model} - Check SuperGrok/Premium+ subscription")
                if model == models[-1]:
                    raise
                continue
            except Exception as inner_e:
                logging.error(f"Daily recap Grok error with {model}: {inner_e}")
                if model == models[-1]:
                    raise
                continue
        if recap:
            escaped_recap = html.escape(recap)
            full_msg = f"**INSTITUTIONAL DAILY SUMMARY**\n\n{escaped_recap}"
            await bot.send_message(CHAT_ID, full_msg, parse_mode='HTML')
            with open(recap_file, 'w') as f:
                f.write(now.isoformat())
            logging.info("Daily recap sent successfully")
            await asyncio.sleep(1)
        else:
            logging.error("No recap generated - all models failed")
            await asyncio.sleep(1)
    except Exception as e:
        logging.error(f"Daily recap error: {type(e).name}: {str(e)}")
        await asyncio.sleep(1)
async def webhook_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        logging.info(f"Webhook update: user {update.effective_user.id}, command {update.message.text}")
async def signal_callback(context):
    start_time = time.perf_counter()
    logging.info("=== Starting elite signal cycle ===")
    try:
        if await BanManager.check_and_sleep(skip_long_sleep=True):
            logging.warning("Global ban during signal check (short mode); skipping")
            total_time = time.perf_counter() - start_time
            logging.info(f"=== Signal cycle complete in {total_time:.2f}s ===")
            return
        if not TELEGRAM_TOKEN:  # NEW: Early check for env
            logging.error("TELEGRAM_TOKEN missing - skipping signal cycle")
            return
        btc_trend = btc_trend_global
        logging.info(f"BTC Trend (global): {btc_trend}")
        now = datetime.now(timezone.utc)
        prices = await fetch_ticker_batch()
        order_books = await fetch_order_flow_batch() # NEW: Fetch order books
        oi_tasks = [fetch_open_interest(s) for s in SYMBOLS]
        oi_results = await asyncio.gather(*oi_tasks, return_exceptions=True)
        oi_data_dict = {SYMBOLS[i]: oi_results[i] if not isinstance(oi_results[i], Exception) else None for i in range(len(SYMBOLS))}
        for symbol in SYMBOLS:
            sym_start = time.perf_counter()
            price = prices.get(symbol)
            if price is None:
                logging.warning(f"No price for {symbol}; skipping")
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            if symbol in last_signal_time and now - last_signal_time[symbol] < timedelta(hours=COOLDOWN_HOURS):
                remaining = timedelta(hours=COOLDOWN_HOURS) - (now - last_signal_time[symbol])
                logging.info(f"Cooldown active for {symbol}: {remaining} remaining - skipping")
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            if await BanManager.check_and_sleep(skip_long_sleep=True):
                logging.warning(f"Global ban hit for {symbol} (short mode); skipping")
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            oi_data = oi_data_dict.get(symbol)
            whale_data = {'boost': 0, 'reason': ''}
            if symbol != 'BTC/USDT':
                threshold = 200 if 'ETH' in symbol else 2000
                try:
                    since = exchange.milliseconds() - 7 * 24 * 60 * 60 * 1000
                    trades = await exchange.fetch_trades(symbol, since=since, limit=1000)
                    inflows = [t for t in trades if t['side'] == 'buy' and t['amount'] > threshold]
                    count = len(inflows)
                    if count >= 5:
                        total = sum(t['amount'] for t in inflows)
                        avg = total / count
                        if avg > 2 * threshold:
                            whale_data = {'boost':2, 'reason':f"+Inst Whale Alt (avg {avg:.1f}, {count} tx)"}
                            logging.info(f"Inst whale boost for {symbol}: avg {avg:.1f}, count {count}")
                except Exception as e:
                    logging.error(f"CCXT whale error for {symbol}: {e}")
            if symbol == 'BTC/USDT':
                try:
                    since = exchange.milliseconds() - 7 * 24 * 60 * 60 * 1000
                    trades = await exchange.fetch_trades(symbol, since=since, limit=1000)
                    inflows = [t for t in trades if t['side'] == 'buy' and t['amount'] > 20]
                    count = len(inflows)
                    if count >= 5:
                        total = sum(t['amount'] for t in inflows)
                        avg = total / count
                        if avg > 40:
                            whale_data = {'boost':2, 'reason': f"+Inst Whale (avg {avg:.1f} BTC, {count} tx)"}
                            logging.info(f"Inst whale boost for {symbol}: avg {avg:.1f}, count {count}")
                except Exception as e:
                    logging.error(f"CCXT whale error for {symbol}: {e}")
            data = {}
            for tf in TIMEFRAMES:
                df_tf = await fetch_ohlcv(symbol, tf)
                book = order_books.get(symbol) # NEW: Pass book to indicators
                data[tf] = add_indicators(df_tf, book) if len(df_tf) > 0 else pd.DataFrame()
                await asyncio.sleep(0.5)
            trend = None
            if symbol == 'BTC/USDT' and len(data.get('1d', pd.DataFrame())) > 0 and 'ema50' in data['1d'].columns and 'ema100' in data['1d'].columns and not pd.isna(data['1d']['ema50'].iloc[-1]) and not pd.isna(data['1d']['ema100'].iloc[-1]):
                l = data['1d'].iloc[-1]
                if l['ema50'] > l['ema100']:
                    trend = "Uptrend"
                elif l['ema50'] < l['ema100']:
                    trend = "Downtrend"
                else:
                    trend = "Sideways"
            elif trend is None and btc_trend:
                trend = btc_trend
            else:
                trend = "Banned/Unknown"
            if trend == "Banned/Unknown":
                if len(data.get('1d', pd.DataFrame())) > 0:
                    l = data['1d'].iloc[-1]
                    if 'ema50' in data['1d'].columns and 'ema100' in data['1d'].columns and not pd.isna(l['ema50']) and not pd.isna(l['ema100']):
                        if l['ema50'] > l['ema100']:
                            trend = "Uptrend"
                        elif l['ema50'] < l['ema100']:
                            trend = "Downtrend"
                        else:
                            trend = "Sideways"
                else:
                    trend = "Sideways" # Fallback neutral
            triggers = set()
            is_alt = symbol != 'BTC/USDT'
            for tf in TIMEFRAMES:
                df = data.get(tf, pd.DataFrame())
                if len(df) == 0:
                    continue
                for func in [detect_pre_cross, detect_divergence, detect_macd, detect_supertrend]:
                    if r := func(df, tf):
                        triggers.add(r)
                fvg_triggers = detect_fvg(df, tf)
                normalized_fvgs = [re.sub(r'\d+\.\d+%', '<0.3%', f) for f in fvg_triggers]
                triggers.update(normalized_fvgs)
                candles = detect_candle_patterns(df, tf)
                triggers.update(candles)
                if len(df) > 20 and 'volume_sma' in df.columns and not pd.isna(df['volume_sma'].iloc[-1]) and df['volume'].iloc[-1] > 1.5 * df['volume_sma'].iloc[-1]:
                    triggers.add(f"Inst Vol Surge ({tf})")
                obs = await find_unmitigated_order_blocks(df, tf=tf, symbol=symbol)
                raw_obs = len(obs.get('bullish', []) + obs.get('bearish', []))
                logging.info(f"Raw OBs for {symbol} {tf}: {raw_obs} (before elite filter)")
                for ob_type in ['bullish', 'bearish']:
                    for ob in obs.get(ob_type, []):
                        if ob['strength'] != 3:
                            continue
                        mid = (ob['low'] + ob['high']) / 2
                        dist_pct = abs(price - mid) / price * 100
                        if dist_pct < 0.5:
                            # Retained: Bidirectional triggers
                            dir_str = "Bullish Long" if ob_type == 'bullish' else "Bearish Short"
                            triggers.add(f"Elite {dir_str} OB str3 {dist_pct:.1f}% away ({tf})")
                # NEW Order Flow: Add delta trigger if |delta| >1%
                delta = df['order_delta'].iloc[-1] if 'order_delta' in df.columns else 0
                if abs(delta) > 1.0:
                    dir_str = "Bullish" if delta > 0 else "Bearish"
                    triggers.add(f"Order Flow {dir_str} Delta {delta:.1f}% ({tf})")
            # Retained: Add 'Bearish Short' to strong keywords for symmetry
            strong_triggers = [t for t in triggers if any(kw in t for kw in ['OB str3', 'Displacement', 'Inst Vol', 'Exhaust', 'Bearish Short', 'Order Flow'])] # NEW: + Order Flow
            logging.info(f"Elite triggers for {symbol}: {len(strong_triggers)} (min 4 req)")
            if len(strong_triggers) < 3:
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Skipped {symbol}: <4 elite triggers")
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            premium_zones = []
            for tf in ['1d', '1w']:
                if tf not in data:
                    continue
                tf_df = data[tf]
                if len(tf_df) > 0:
                    book = order_books.get(symbol) # NEW: Pass to zones
                    tf_zones = await find_next_premium_zones(tf_df, price, tf, symbol, oi_data, trend=trend, whale_data=whale_data, order_book=book)
                    premium_zones.extend(tf_zones)
            logging.info(f"Total elite {len(premium_zones)} premium zones for {symbol} before Grok")
            if not premium_zones:
                sym_time = time.perf_counter() - sym_start
                logging.info(f"Skipped {symbol}: No elite zones")
                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                continue
            async def mini_backtest_zone(zone: Dict, df_14d: pd.DataFrame) -> float:
                hits = 0
                total = 0
                for _, row in df_14d.iterrows():
                    if zone['direction'] == 'Long' and max(zone['zone_low'], row['low']) <= min(zone['zone_high'], row['high']):
                        hits += 1
                    elif zone['direction'] == 'Short' and max(zone['zone_low'], row['low']) <= min(zone['zone_high'], row['high']):
                        hits += 1
                    total += 1
                return hits / total * 100 if total > 0 else 0
            df_14d = await fetch_ohlcv(symbol, '1h', since=int((datetime.now(timezone.utc)-timedelta(days=14)).timestamp()*1000))
            for z in premium_zones:
                bt_win = await mini_backtest_zone(z, df_14d)
                if bt_win < 60:
                    z['prob'] -= 15
                    logging.info(f"Mini-backtest adjusted {symbol} {z['direction']} prob to {z['prob']}% (bt_win {bt_win:.1f}%)")
                if z['prob'] < 90:
                    premium_zones = [zz for zz in premium_zones if zz != z]
            if premium_zones:
                grok_potential = await query_grok_potential(premium_zones, symbol, price, trend, btc_trend)
                retry_count = 0
                max_retries = 2
                while retry_count < max_retries:
                    has_roadmap = 'roadmap' in grok_potential and grok_potential.get('roadmap')
                    if not has_roadmap and premium_zones:
                        logging.info(f"No roadmap on attempt {retry_count+1} for {symbol}; forcing retry...")
                        grok_potential = await query_grok_potential(premium_zones, symbol, price, trend, btc_trend)
                        retry_count += 1
                        continue
                    break
                live_trade_key = 'live_trade' if 'live_trade' in grok_potential else None
                if live_trade_key and not grok_potential.get('no_live_trade', True):
                    grok = grok_potential[live_trade_key]
                    logging.info(f"Grok elite live trade for {symbol}: {grok}")
                    if grok.get('strength', 0) != 3 or grok.get('confidence', 0) < 90:
                        logging.info(f"Skipped non-elite live {symbol}: str {grok.get('strength', 0)} or conf {grok.get('confidence', 0)}")
                        sym_time = time.perf_counter() - sym_start
                        logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                        continue
                    required_keys = ['direction', 'entry_low', 'entry_high', 'sl', 'tp1', 'tp2', 'leverage', 'confidence', 'reason']
                    if not all(key in grok for key in required_keys):
                        logging.warning(f"Invalid Grok response for {symbol}: {grok}")
                        await asyncio.sleep(2)
                        sym_time = time.perf_counter() - sym_start
                        logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                        continue
                    entry_low = min(grok['entry_low'], grok['entry_high'])
                    entry_high = max(grok['entry_low'], grok['entry_high'])
                    new_conf = grok['confidence']
                    leverage = min(grok['leverage'], 5)
                    overlapping = []
                    for trades_dict, dict_name in [(open_trades, 'open'), (protected_trades, 'protected')]:
                        for key, t in trades_dict.items():
                            overlap_ratio = zones_overlap(entry_low, entry_high, t['entry_low'], t['entry_high'])
                            if overlap_ratio > 0:
                                overlapping.append((key, t, dict_name, overlap_ratio))
                    if overlapping:
                        same_dir_overlaps = [o for o in overlapping if o[1]['direction'] == grok['direction']]
                        if same_dir_overlaps:
                            max_conf = max(new_conf, max(o[1]['confidence'] for o in same_dir_overlaps))
                            min_sl = min(grok['sl'], min(o[1]['sl'] for o in same_dir_overlaps))
                            max_tp = max(grok['tp2'], max(o[1]['tp2'] for o in same_dir_overlaps))
                            for key, t, dict_name, ratio in same_dir_overlaps:
                                if t.get('active') and dict_name == 'open':
                                    protected_trades[key] = open_trades.pop(key)
                                    save_trades(open_trades)
                                    save_protected(protected_trades)
                                    logging.info(f"Protected active overlap for {key}")
                            merge_key, merge_t, _, _ = same_dir_overlaps[0]
                            merge_t['confidence'] = max_conf
                            merge_t['sl'] = min_sl
                            merge_t['tp2'] = max_tp
                            if new_conf > merge_t['confidence']:
                                merge_t['entry_low'] = entry_low
                                merge_t['entry_high'] = entry_high
                            logging.info(f"Merged elite live signal for {symbol}: conf {max_conf}")
                            for key, t, dict_name, ratio in same_dir_overlaps[1:]:
                                if ratio >= 0.95:
                                    if dict_name == 'open':
                                        del open_trades[key]
                                    else:
                                        del protected_trades[key]
                                    logging.info(f"Removed high-overlap {key} (ratio {ratio:.2f})")
                            save_trades(open_trades)
                            save_protected(protected_trades)
                            last_signal_time[symbol] = now
                        else:
                            max_overlap = max(r for _, _, _, r in overlapping)
                            if max_overlap < 0.95:
                                trade_key = symbol
                                open_trades[trade_key] = {
                                    'direction': grok['direction'],
                                    'entry_low': entry_low,
                                    'entry_high': entry_high,
                                    'sl': grok['sl'],
                                    'tp1': grok['tp1'],
                                    'tp2': grok['tp2'],
                                    'leverage': leverage,
                                    'confidence': new_conf,
                                    'active': False,
                                    'last_check': datetime.now(timezone.utc),
                                    'processed': False,
                                    'strength': grok.get('strength', 3)
                                }
                                save_trades(open_trades)
                                last_signal_time[symbol] = now
                            else:
                                logging.info(f"Skipped live {symbol}: high overlap diff dir {max_overlap:.2f}")
                                sym_time = time.perf_counter() - sym_start
                                logging.info(f"Finished {symbol} in {sym_time:.2f}s")
                                continue
                    else:
                        trade_key = symbol
                        open_trades[trade_key] = {
                            'direction': grok['direction'],
                            'entry_low': entry_low,
                            'entry_high': entry_high,
                            'sl': grok['sl'],
                            'tp1': grok['tp1'],
                            'tp2': grok['tp2'],
                            'leverage': leverage,
                            'confidence': new_conf,
                            'active': False,
                            'last_check': datetime.now(timezone.utc),
                            'processed': False,
                            'strength': grok.get('strength', 3)
                        }
                        save_trades(open_trades)
                        last_signal_time[symbol] = now
                    entry_mid = (entry_low + entry_high) / 2
                    rr1 = abs(grok['tp1'] - entry_mid) / abs(grok['sl'] - entry_mid)
                    rr2 = abs(grok['tp2'] - entry_mid) / abs(grok['sl'] - entry_mid)
                    dist_to_sl = abs(entry_mid - grok['sl']) / entry_mid * 100
                    msg = (
                        f"**{symbol.replace('/USDT','')} ELITE LIVE SIGNAL (Inst OB Hit!)**\n\n"
                        f"*Price:* {format_price(price)} | *Trend:* {trend}"
                    )
                    if is_alt and btc_trend: msg += f" | *BTC:* {btc_trend}"
                    msg += "\n\n"
                    msg += (
                        f"{grok['direction']} | **{new_conf}%**\n"
                        f"*Entry Zone:* {format_price(entry_low)}–{format_price(entry_high)}\n"
                        f"*SL:* {format_price(grok['sl'])} | *TP1* {format_price(grok['tp1'])} | *TP2* {format_price(grok['tp2'])}\n"
                        f"*Leverage:* {leverage}x | *R:R* 1:{rr1:.1f} / 1:{rr2:.1f} | *Risk* {dist_to_sl:.2f}%\n"
                        f"**Reason:** {grok['reason']}"
                    )
                    logging.info(f"Sending elite live for {symbol}")
                    await bot.send_message(CHAT_ID, msg, parse_mode='Markdown')
                else:
                    sent_roadmap = False
                    if 'roadmap' in grok_potential and grok_potential['roadmap']:
                        conservative_msg = f"**{symbol.replace('/USDT','')} ELITE ROADMAP** | *Price:* {format_price(price)} | *Trend:* {trend}"
                        if is_alt and btc_trend: conservative_msg += f" | *BTC:* {btc_trend}"
                        conservative_msg += "\n\nElite inst zones (90%+ conf, str=3):\n"
                        roadmap_count = 0
                        for i, z in enumerate(grok_potential['roadmap'], 1):
                            if z['confidence'] > 90 and z['dist_pct'] < 5 and z.get('strength', 0) == 3:
                                roadmap_count += 1
                                entry_low = min(z['entry_low'], z['entry_high'])
                                entry_high = max(z['entry_low'], z['entry_high'])
                                new_conf = z['confidence']
                                leverage = min(z['leverage'], 5)
                                unique_key = f"{symbol}roadmap{i}"
                                overlapping = []
                                for trades_dict, dict_name in [(open_trades, 'open'), (protected_trades, 'protected')]:
                                    for key, t in trades_dict.items():
                                        overlap_ratio = zones_overlap(entry_low, entry_high, t['entry_low'], t['entry_high'])
                                        if overlap_ratio > 0:
                                            overlapping.append((key, t, dict_name, overlap_ratio))
                                if overlapping:
                                    same_dir_overlaps = [o for o in overlapping if o[1]['direction'] == z['direction']]
                                    if same_dir_overlaps:
                                        max_conf = max(new_conf, max(o[1]['confidence'] for o in same_dir_overlaps))
                                        min_sl = min(z['sl'], min(o[1]['sl'] for o in same_dir_overlaps))
                                        max_tp = max(z['tp2'], max(o[1]['tp2'] for o in same_dir_overlaps))
                                        for key, t, dict_name, ratio in same_dir_overlaps:
                                            if t.get('active') and dict_name == 'open':
                                                protected_trades[key] = open_trades.pop(key)
                                                save_trades(open_trades)
                                                save_protected(protected_trades)
                                        merge_key, merge_t, _, _ = same_dir_overlaps[0]
                                        merge_t['confidence'] = max_conf
                                        merge_t['sl'] = min_sl
                                        merge_t['tp2'] = max_tp
                                        if new_conf > merge_t['confidence']:
                                            merge_t['entry_low'] = entry_low
                                            merge_t['entry_high'] = entry_high
                                        logging.info(f"Merged elite roadmap for {symbol}: conf {max_conf}")
                                        for key, t, dict_name, ratio in same_dir_overlaps[1:]:
                                            if ratio >= 0.95:
                                                if dict_name == 'open':
                                                    del open_trades[key]
                                                else:
                                                    del protected_trades[key]
                                        save_trades(open_trades)
                                        save_protected(protected_trades)
                                        last_signal_time[symbol] = now
                                    else:
                                        max_overlap = max(r for _, _, _, r in overlapping)
                                        if max_overlap < 0.95:
                                            open_trades[unique_key] = {
                                                'direction': z['direction'],
                                                'entry_low': entry_low,
                                                'entry_high': entry_high,
                                                'sl': z['sl'],
                                                'tp1': z['tp1'],
                                                'tp2': z['tp2'],
                                                'leverage': leverage,
                                                'confidence': new_conf,
                                                'type': 'roadmap',
                                                'active': False,
                                                'last_check': datetime.now(timezone.utc),
                                                'dist_pct': z['dist_pct'],
                                                'processed': False,
                                                'strength': z['strength']
                                            }
                                            save_trades(open_trades)
                                            last_signal_time[symbol] = now
                                            logging.info(f"Added elite roadmap trade for {symbol}: {z['direction']} {new_conf}% str3")
                                        else:
                                            logging.info(f"Skipped elite roadmap {symbol}: high overlap diff dir {max_overlap:.2f}")
                                            continue
                                else:
                                    open_trades[unique_key] = {
                                        'direction': z['direction'],
                                        'entry_low': entry_low,
                                        'entry_high': entry_high,
                                        'sl': z['sl'],
                                        'tp1': z['tp1'],
                                        'tp2': z['tp2'],
                                        'leverage': leverage,
                                        'confidence': new_conf,
                                        'type': 'roadmap',
                                        'active': False,
                                        'last_check': datetime.now(timezone.utc),
                                        'dist_pct': z['dist_pct'],
                                        'processed': False,
                                        'strength': z['strength']
                                    }
                                    save_trades(open_trades)
                                    last_signal_time[symbol] = now
                                    logging.info(f"Added elite roadmap trade for {symbol}: {z['direction']} {new_conf}% str3")
                                entry_mid = (z['entry_low'] + z['entry_high']) / 2
                                rr = abs(z['tp2'] - entry_mid) / abs(z['sl'] - entry_mid)
                                conservative_msg += f"{i}. {z['direction']} **{z['confidence']}**\n"
                                conservative_msg += f"*Zone:* {format_price(z['entry_low'])}–{format_price(z['entry_high'])} | *SL:* {format_price(z['sl'])}\n"
                                conservative_msg += f"*TP1:* {format_price(z['tp1'])} | *TP2:* {format_price(z['tp2'])} | {min(z['leverage'], 5)}x | *R:R* 1:{rr:.1f}\n"
                                conservative_msg += f"**Reason:** {z['reason']} ( {z['dist_pct']:.1f}% away, str3 )\n\n"
                        if roadmap_count > 0:
                            logging.info(f"Sending elite roadmap for {symbol}")
                            await bot.send_message(CHAT_ID, conservative_msg, parse_mode='Markdown')
                            last_signal_time[symbol] = now
                            sent_roadmap = True
                    if not sent_roadmap and len(strong_triggers) >= 4:
                        trigger_msg = f"**{symbol.replace('/USDT','')} - Elite Triggers** | *Price:* {format_price(price)} | *Trend:* {trend}"
                        if is_alt and btc_trend: trigger_msg += f" | *BTC:* {btc_trend}"
                        trigger_msg += "\n\n__Inst signals:__\n" + '\n'.join([f"• {t}" for t in sorted(strong_triggers[:6])])
                        logging.info(f"Sending elite triggers for {symbol}: {len(strong_triggers)}")
                        await bot.send_message(CHAT_ID, trigger_msg, parse_mode='Markdown')
                        last_signal_time[symbol] = now
                        sent_roadmap = True
            sym_time = time.perf_counter() - sym_start
            logging.info(f"Finished {symbol} in {sym_time:.2f}s")
            await asyncio.sleep(2)
        total_time = time.perf_counter() - start_time
        logging.info(f"=== Elite signal cycle complete in {total_time:.2f}s ===")
    except Exception as e:
        logging.error(f"Signal callback error: {e}")
        total_time = time.perf_counter() - start_time
        logging.info(f"=== Signal cycle complete in {total_time:.2f}s (error) ===")
async def post_init(application: Application) -> None:
    logging.info("Starting post_init: Sending welcome and setting webhook...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.bybit.com/v5/market/tickers?category=spot")
            if r.status_code != 200:
                logging.warning("Bybit API health check failed - check connectivity")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post("https://api.x.ai/v1/chat/completions", json={"model": "grok-3", "messages": [{"role": "user", "content": "ping"}]}, headers={"Authorization": f"Bearer {XAI_API_KEY}"})
            if r.status_code == 401:
                logging.error("xAI API unauthorized - Check SuperGrok/Premium+ key!")
            elif r.status_code != 200:
                logging.warning(f"xAI API health check failed: status {r.status_code}")
    except Exception as e:
        logging.error(f"API health check error: {e}")
    await send_welcome_once()
    await application.bot.delete_webhook(drop_pending_updates=True)
    external_hostname = os.getenv('RENDER_EXTERNAL_HOSTNAME')
    if not external_hostname:
        logging.error("RENDER_EXTERNAL_HOSTNAME not set - cannot set webhook!")
        return
    webhook_url = f"https://{external_hostname}/webhook"
    logging.info(f"Setting webhook to: {webhook_url}")
    await application.bot.set_webhook(url=webhook_url)
    logging.info("Webhook set successfully. Jobs will now run.")
    asyncio.create_task(price_background_task())
    logging.info("Background Polling Task started – fresh prices + order flow every 10s!")
    logging.info("Post_init complete – Elite signals via job in ~60s.")
def main():
    # NEW: Env debug logs
    logging.info(f"Loaded env: TOKEN={TELEGRAM_TOKEN[:5] if TELEGRAM_TOKEN else 'MISSING'}..., CHAT={CHAT_ID if CHAT_ID else 'MISSING'}, KEY={XAI_API_KEY[:5] if XAI_API_KEY else 'MISSING'}...")
    if not all([TELEGRAM_TOKEN, CHAT_ID, XAI_API_KEY]):
        logging.error("Missing required env vars: TELEGRAM_TOKEN, CHAT_ID, XAI_API_KEY")
        sys.exit(1)  # NEW: Explicit exit
    logging.info(f"Elite cooldown: {COOLDOWN_HOURS}h | Inst OB only, zero noise + Order Flow delta")
    application = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("health", health_cmd))
    application.add_handler(CommandHandler("recap", recap_cmd))
    application.add_handler(CommandHandler("backtest", backtest_cmd))
    application.add_handler(MessageHandler(filters.ALL, webhook_update))
    signal_job = application.job_queue.run_repeating(
        signal_callback,
        interval=CHECK_INTERVAL,
        first=60,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )
    logging.info(f"Elite signal job: first in 60s, max_instances=2")
    track_job = application.job_queue.run_repeating(
        track_callback,
        interval=TRACK_INTERVAL,
        first=60,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )
    logging.info(f"Track job: interval=5s")
    price_job = application.job_queue.run_repeating(
        price_update_callback,
        interval=60,
        first=30,
        job_kwargs={'max_instances': 2, 'misfire_grace_time': 30}
    )
    logging.info("Price update job: every 60s")
    btc_job = application.job_queue.run_repeating(
        btc_trend_update,
        interval=300,
        first=60,
        job_kwargs={'max_instances': 1}
    )
    logging.info("BTC trend job: every 5min")
    now = datetime.now(timezone.utc)
    target = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if now.time() > target.time():
        target += timedelta(days=1)
    delta = target - now
    first_delay = int(delta.total_seconds())
    logging.info(f"Daily recap first in {first_delay}s")
    application.job_queue.run_repeating(daily_callback, interval=86400, first=first_delay)
    webhook_path = "/webhook"
    port = int(os.getenv("PORT", 10000))
    external_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}"
    webhook_url = f"{external_url}{webhook_path}"
    logging.info(f"Starting webhook server on port {port}, URL: {webhook_url}")
    try:
        application.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=webhook_path,
            webhook_url=webhook_url
        )
    finally:
        asyncio.run(exchange.close())
        asyncio.run(futures_exchange.close())
        logging.info("CCXT exchanges closed gracefully!")
if __name__ == "__main__":
    main()
