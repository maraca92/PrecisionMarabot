import os
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import httpx
import json
import re
from collections import OrderedDict
from typing import Dict, Any, Optional, List
import asyncio
import time
from flask import Flask, request

# Konfiguracja
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')  # Opcjonalne dla demo
BINANCE_SECRET = os.getenv('BINANCE_SECRET')
GROK_API_KEY = os.getenv('GROK_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja exchange i Grok
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'sandbox': True,  # Demo mode
    'enableRateLimit': True,
})
client = OpenAI(
    api_key=GROK_API_KEY,
    base_url='https://api.x.ai/v1',
)

# Cache i Semaphore (z starego kodu)
CACHE_TTL = 1800
HTF_CACHE_TTL = 3600
TICKER_CACHE_TTL = 10
MAX_CACHE_SIZE = 50
ohlcv_cache: OrderedDict = OrderedDict()
ticker_cache: OrderedDict = OrderedDict()
fetch_sem = asyncio.Semaphore(4)

def evict_if_full(cache: OrderedDict, max_size: int = MAX_CACHE_SIZE):
    """Evict oldest cache entries if size exceeds limit."""
    evicted = 0
    while len(cache) > max_size:
        cache.popitem(last=False)  # FIFO
        evicted += 1
    if evicted > 0:
        logging.debug(f"Evicted {evicted} from {cache.__class__.__name__} cache")

# Inicjalizacja DB (ulepszona z pnl field)
conn = sqlite3.connect('trades.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT,
        timeframe TEXT,
        ob_type TEXT,
        entry_price REAL,
        sl_price REAL,
        tp_price REAL,
        outcome TEXT DEFAULT 'open',  -- 'win', 'loss', 'open'
        pnl REAL DEFAULT 0.0,
        timestamp TEXT,
        grok_confidence INTEGER
    )
''')
conn.commit()

def get_current_price(pair):
    """Pobiera aktualną cenę (z cache/batch)."""
    ticker = exchange.fetch_ticker(pair)
    return ticker['last']

def update_trade_outcome(trade_id):
    """Aktualizuje outcome i PnL (ulepszone z process_trade)."""
    cursor.execute("SELECT entry_price, sl_price, tp_price FROM trades WHERE id=?", (trade_id,))
    trade = cursor.fetchone()
    if not trade:
        return
    entry, sl, tp = trade
    current = get_current_price('BTC/USDT')
    if current >= tp:
        outcome = 'win'
        pnl = (tp - entry) * 1.0 - 0.0008 * 2 * entry  # Fee adj (0.04%)
    elif current <= sl:
        outcome = 'loss'
        pnl = (sl - entry) * 1.0 - 0.0008 * 2 * entry
    else:
        return  # Open
    cursor.execute("UPDATE trades SET outcome=?, pnl=? WHERE id=?", (outcome, pnl, trade_id))
    conn.commit()

def calculate_winrate():
    """Oblicza winrate z zamkniętych tradów (z PnL)."""
    cursor.execute("SELECT outcome, pnl FROM trades WHERE outcome != 'open'")
    closed = cursor.fetchall()
    if not closed:
        return 0, 0, 0.0
    wins = sum(1 for o, _ in closed if o == 'win')
    total = len(closed)
    total_pnl = sum(p for _, p in closed)
    return (wins / total * 100) if total > 0 else 0, total, total_pnl

# Dodane z starego: Indicators, OB detection, FVG, etc.
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df = df.copy()
    df['ema50'] = ta.ema(df['close'], 50)
    df['ema100'] = ta.ema(df['close'], 100)
    df['rsi'] = ta.rsi(df['close'], 14)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    macd_data = ta.macd(df['close'])
    if macd_data is not None and len(macd_data.columns) >= 3:
        df['macd'] = macd_data.iloc[:, 0]
        df['macd_signal'] = macd_data.iloc[:, 1]
        df['macd_hist'] = macd_data.iloc[:, 2]
    if len(df) >= 10:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        if st is not None and 'SUPERT_10_3.0' in st.columns:
            df['supertrend'] = st['SUPERT_10_3.0']
            df['supertrend_dir'] = st['SUPERTd_10_3.0']
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
    key_cols = ['ema50', 'ema100', 'rsi', 'volume_sma', 'macd', 'macd_signal', 'supertrend_dir']
    subset_key = [col for col in key_cols if col in df.columns]
    df = df.dropna(subset=subset_key)
    return df

async def find_unmitigated_order_blocks(df: pd.DataFrame, lookback: int = 100, atr_mult: float = 1.5, tf: str = None, symbol: str = None) -> Dict[str, List[Dict]]:
    if len(df) < 20:
        return {'bullish': [], 'bearish': []}
    ltf_mult = 1 if tf in ['15m', '1h'] else 2 if tf == '4h' else 3
    dyn_lookback = lookback * ltf_mult
    df_local = df.tail(dyn_lookback).copy()
    df_local['atr'] = ta.atr(df_local['high'], df_local['low'], df_local['close'], 14)
    df_local['direction'] = np.where(df_local['close'] > df_local['open'], 1, -1)
    df_local['swing_high'] = df_local['high'].rolling(5, center=True).max() == df_local['high']
    df_local['swing_low'] = df_local['low'].rolling(5, center=True).min() == df_local['low']
    obs = {'bullish': [], 'bearish': []}
    for i in range(10, len(df_local) - 5):
        if df_local['swing_high'].iloc[i]:
            ob_high = df_local['high'].iloc[i]
            ob_low = max(df_local['open'].iloc[i], df_local['close'].iloc[i])
            move_down = df_local['low'].iloc[i+3:i+6].min() < ob_low - (df_local['atr'].iloc[i] * atr_mult)
            if move_down:
                mitigated = any(df_local['high'].iloc[i+1:] > ob_high)
                zone_type = 'Breaker' if any(df_local['low'].iloc[i+1:i+5] < ob_low) else 'OB'
                strength = 3 if zone_type == 'OB' else 2
                if not mitigated:
                    obs['bearish'].append({
                        'low': ob_low, 'high': ob_high, 'type': zone_type,
                        'strength': strength, 'index': i, 'mitigated': False
                    })
    for i in range(10, len(df_local) - 5):
        if df_local['swing_low'].iloc[i]:
            ob_low = df_local['low'].iloc[i]
            ob_high = min(df_local['open'].iloc[i], df_local['close'].iloc[i])
            move_up = df_local['high'].iloc[i+3:i+6].max() > ob_high + (df_local['atr'].iloc[i] * atr_mult)
            if move_up:
                mitigated = any(df_local['low'].iloc[i+1:] < ob_low)
                zone_type = 'Breaker' if any(df_local['high'].iloc[i+1:i+5] > ob_high) else 'OB'
                strength = 3 if zone_type == 'OB' else 2
                if not mitigated:
                    obs['bullish'].append({
                        'low': ob_low, 'high': ob_high, 'type': zone_type,
                        'strength': strength, 'index': i, 'mitigated': False
                    })
    # Merger HTF/LTF (jak w starym)
    if tf in ['1d', '1w'] and symbol:
        try:
            df_1h = await fetch_ohlcv(symbol, '1h', 200)
            obs_1h = await find_unmitigated_order_blocks(df_1h, lookback=100, tf='1h', symbol=symbol)
            for ob_type in ['bullish', 'bearish']:
                for ob_htf in obs[ob_type]:
                    for ob_ltf in obs_1h[ob_type]:
                        overlap_ratio = zones_overlap(ob_htf['low'], ob_htf['high'], ob_ltf['low'], ob_ltf['high'], 0.5)
                        if overlap_ratio > 0.5:
                            merged_low = (ob_htf['low'] + ob_ltf['low']) / 2
                            merged_high = (ob_htf['high'] + ob_ltf['high']) / 2
                            merged_strength = max(ob_htf['strength'], ob_ltf['strength'])
                            ob_htf['low'] = merged_low
                            ob_htf['high'] = merged_high
                            ob_htf['strength'] = merged_strength
                            ob_htf['type'] = ob_htf['type'] if ob_htf['strength'] >= ob_ltf['strength'] else ob_ltf['type']
                            logging.info(f"Merged {ob_type} OB for {symbol} {tf}: overlap {overlap_ratio:.2f}")
        except Exception as e:
            logging.warning(f"HTF merge failed for {symbol} {tf}: {e}")
    for key in obs:
        obs[key] = sorted(obs[key], key=lambda z: (len(df_local) - z['index']) * z['strength'], reverse=True)[:3]
    return obs

def zones_overlap(z1_low: float, z1_high: float, z2_low: float, z2_high: float, threshold: float = 0.5) -> float:
    o_low = max(z1_low, z2_low)
    o_high = min(z1_high, z2_high)
    if o_low >= o_high:
        return 0.0
    overlap_len = o_high - o_low
    min_width = min(z1_high - z1_low, z2_high - z2_low)
    return (overlap_len / min_width)

def detect_fvg(df: pd.DataFrame, tf: str, proximity_pct: float = 0.5) -> List[str]:
    if len(df) < 5:
        return []
    df_local = df.tail(50).copy()
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

def calc_liquidity_profile(df: pd.DataFrame, bins: int = 20) -> Dict[float, float]:
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
    threshold = sorted_vols[int(len(sorted_vols) * 0.8)] if sorted_vols else 0
    hot_zones = {k: v / threshold if v > threshold else 0 for k, v in vol_profile.items()}
    return hot_zones

async def fetch_ohlcv(symbol: str, tf: str, limit: int = 200, since: Optional[int] = None) -> pd.DataFrame:
    async with fetch_sem:
        cache_key = f"{symbol}{tf}"
        now = time.time()
        ttl = HTF_CACHE_TTL if tf in ['1d', '1w'] else CACHE_TTL
        if cache_key in ohlcv_cache and now - ohlcv_cache[cache_key]['timestamp'] < ttl:
            return ohlcv_cache[cache_key]['df']
        evict_if_full(ohlcv_cache)
        try:
            params = {'limit': limit}
            if since:
                params['since'] = since
            data = exchange.fetch_ohlcv(symbol, tf, **params)
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['ts'], unit='ms')
            ohlcv_cache[cache_key] = {'df': df, 'timestamp': now}
            return df
        except Exception as e:
            logging.warning(f"OHLCV fetch failed for {symbol} {tf}: {e}")
            return pd.DataFrame()

# Whale detection (z CCXT trades, jak w starym)
async def detect_whale_boost(symbol: str) -> Dict[str, Any]:
    try:
        since = exchange.milliseconds() - 30 * 24 * 60 * 60 * 1000  # 30d
        trades = await exchange.fetch_trades(symbol, since=since, limit=1000)
        threshold = 10 if 'BTC' in symbol else 100
        inflows = [t for t in trades if t['side'] == 'buy' and t['amount'] > threshold]
        count = len(inflows)
        if count > 0:
            total = sum(t['amount'] for t in inflows)
            avg = total / count
            if avg > threshold:
                return {'boost': 1, 'reason': f"+Whale (avg {avg:.1f}, {count} tx)"}
    except Exception as e:
        logging.error(f"Whale detection error for {symbol}: {e}")
    return {'boost': 0, 'reason': ''}

# Trend update (global BTC, jak w starym)
btc_trend_global = "Unknown"
async def btc_trend_update():
    global btc_trend_global
    df = await fetch_ohlcv('BTC/USDT', '1d')
    if len(df) > 0:
        df = add_indicators(df)
        if 'ema50' in df.columns and not pd.isna(df['ema50'].iloc[-1]):
            l = df.iloc[-1]
            if l['ema50'] > l['ema100']:
                btc_trend_global = "Uptrend"
            elif l['ema50'] < l['ema100']:
                btc_trend_global = "Downtrend"
            else:
                btc_trend_global = "Sideways"
            logging.info(f"BTC trend: {btc_trend_global}")

# Ulepszony Grok prompt (z JSON, RR/precision check, whale/boost)
def check_precision(trade: Dict[str, Any]) -> bool:
    price_keys = ['entry_low', 'entry_high', 'sl', 'tp1', 'tp2']
    for key in price_keys:
        if key in trade:
            p = trade[key]
            if round(p, 2) == int(p):  # Avoid round numbers
                return False
    return True

def check_rr(trade: Dict[str, Any]) -> bool:
    entry_mid = (trade['entry_low'] + trade['entry_high']) / 2
    rr2 = abs(trade['tp2'] - entry_mid) / abs(trade['sl'] - entry_mid)
    return rr2 >= 2.5

async def query_grok_potential(zones: List[Dict], symbol: str, current_price: float, trend: str, btc_trend: Optional[str], whale_data: Dict[str, Any]) -> Dict[str, Any]:
    zone_summary = "\n".join([f"{z['direction']}: {z['zone_low']:.4f}-{z['zone_high']:.4f} ({z['confluence']}, {z['prob']}% prob, {z['dist_pct']:.1f}% away, str{z['strength']})" for z in zones])
    context = f"{symbol} | Price: {current_price:.4f} | Trend: {trend} {'| BTC: ' + btc_trend if symbol != 'BTC/USDT' else ''}\nNext Zones:\n{zone_summary}\nWhale: {whale_data['reason']}"
    system_prompt = (
        "You are an ICT institutional trader. Output ONLY valid JSON. "
        "High-conviction order-block setup ≥80% confidence → exact format:\n"
        '{"symbol":"BTC","direction":"Long" or "Short","entry_low":68234.5678,"entry_high":68456.1234,"'
        '"sl":67987.2345,"tp1":68890.7890,"tp2":69543.4567,"leverage":2-10,"confidence":80-100,"'
        '"strength":1-3,"reason":"concise reason (max 100 chars)"}\n'
        f"Consider SuperTrend regime, FVG presence, MACD, RSI, order blocks (prefer str>=3), liquidity heatmap, RSI 1D/1W exhaustion. "
        f"Use precise price levels with at least 4 decimal places (e.g., 68234.5678), avoiding round numbers unless exactly at key support/resistance."
        f"Boost reversal if RSI 1D/1W OS/OB + FVG fill, conservative in trend.\n"
        "Conservative: Max 10% risk, leverage 2-10x, reasonable SL (ATR*2.0 behind strong support/resistance), focus 4h+ TF wide 12-48h+ horizon no scalping. Always ensure entry_low < entry_high. No setup → {\"no_trade\": true}"
    )
    payload = {
        "model": "grok-beta",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}],
        "temperature": 0.2,
        "max_tokens": 400
    }
    response = client.chat.completions.create(**payload)
    content = response.choices[0].message.content.strip()
    try:
        result = json.loads(content)
        if 'live_trade' in result and not check_precision(result['live_trade']):
            result = {"no_trade": True}
        if 'roadmap' in result:
            for z in result['roadmap']:
                if not check_precision(z) or not check_rr(z):
                    z['confidence'] -= 10  # Penalize
        return result
    except json.JSONDecodeError:
        return {"no_trade": True}

# Background price task (jak v23.71)
prices_global = {'BTC/USDT': None}
async def price_background_task():
    while True:
        try:
            prices = exchange.fetch_tickers(['BTC/USDT'])
            prices_global['BTC/USDT'] = prices['BTC/USDT']['last']
            await asyncio.sleep(10)
        except Exception as e:
            logging.warning(f"Background price error: {e}")
            await asyncio.sleep(5)

# Process trade (z PnL, trailing, exposure – integruje z SQLite)
async def process_trade(trade_id: int, current_price: float):
    cursor.execute("SELECT entry_price, sl_price, tp_price, outcome FROM trades WHERE id=?", (trade_id,))
    trade = cursor.fetchone()
    if not trade or trade[3] != 'open':
        return
    entry, sl, tp, _ = trade
    # Trailing to BE after TP1 (simplified)
    if current_price >= tp * 0.5:  # Assume partial TP1 hit
        sl = max(sl, entry)  # BE
    hit_tp = current_price >= tp
    hit_sl = current_price <= sl
    if hit_tp or hit_sl:
        outcome = 'win' if hit_tp else 'loss'
        pnl = (tp - entry if hit_tp else sl - entry) * 1.0 - 0.0008 * 2 * entry  # Fee
        cursor.execute("UPDATE trades SET outcome=?, pnl=? WHERE id=?", (outcome, pnl, trade_id))
        conn.commit()

# Ulepszony scan_ob (z nowymi funkcjami, cooldown 10h, whale, indicators, FVG, liquidity)
last_signal_time = {'BTC/USDT': datetime.min}
COOLDOWN_HOURS = 10
async def scan_ob(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    symbol = 'BTC/USDT'
    now = datetime.now(timezone.utc)
    if now - last_signal_time[symbol] < timedelta(hours=COOLDOWN_HOURS):
        await context.bot.send_message(chat_id=chat_id, text=f"Cooldown: {COOLDOWN_HOURS}h do następnego skanu.")
        return
    await context.bot.send_message(chat_id=chat_id, text="Skanuję instytucyjne OB na BTC/USDT 4H...")

    # Fetch data
    ohlcv = await fetch_ohlcv(symbol, '4h', 200)
    ohlc = add_indicators(ohlcv)  # Dodane indicators
    if len(ohlc) < 50:
        await context.bot.send_message(chat_id=chat_id, text="Niewystarczające dane.")
        return

    # Detekcja OB (nowa funkcja, filtr vol >2x avg dla instytucjonalnych)
    avg_vol = ohlc['volume'].mean()
    obs = await find_unmitigated_order_blocks(ohlc, tf='4h', symbol=symbol)
    ob_df = []
    for ob_type in ['bullish', 'bearish']:
        for ob in obs.get(ob_type, []):
            if ob['high'] - ob['low'] > 0 and ohlc['volume'].iloc[ob['index']] > avg_vol * 2:  # Instytucjonalny filtr
                ob_df.append({**ob, 'OB': 1 if ob_type == 'bullish' else -1})
    if not ob_df:
        await context.bot.send_message(chat_id=chat_id, text="Brak silnych instytucyjnych OB.")
        return

    latest_ob = sorted(ob_df, key=lambda x: x['index'], reverse=True)[0]
    ob_type_str = "Bullish" if latest_ob['OB'] == 1 else "Bearish"
    top = latest_ob['high']
    bottom = latest_ob['low']
    strength = latest_ob['strength']

    # Confluence: FVG, Liquidity, Whale
    fvgs = detect_fvg(ohlc, '4h')
    liq_profile = calc_liquidity_profile(ohlc)
    whale_data = await detect_whale_boost(symbol)
    confluence = f"FVG: {len(fvgs)}, Liq: {any(v > 1.2 for v in liq_profile.values())}, Whale: {whale_data['reason']}"

    # Grok (ulepszony prompt z whale, confluence)
    prompt = f"""
    Oceń instytucjonalny Order Block na BTC/USDT 4H:
    Typ: {ob_type_str}, Top: {top:.4f}, Bottom: {bottom:.4f}, Moc str: {strength}, Vol >2x avg.
    Confluence: {confluence}. Dane OHLC: {ohlc.tail(5).to_json()}.
    Podaj JSON: {{"direction": "Long/Short", "entry_low": x. xxxx, "entry_high": y.yyyy, "sl": z.zzzz, "tp1": a.aaaa, "tp2": b.bbbb, "leverage": 2-10, "confidence": 80-100, "strength": 1-3, "reason": "max 100 chars"}}.
    Użyj SMC: retest bez ominięcia, RR >=2.5, precise 4 decimals.
    """
    grok_response = await query_grok_potential([{'direction': ob_type_str, 'zone_low': bottom, 'zone_high': top, 'confluence': confluence, 'prob': 85, 'dist_pct': 0, 'strength': strength}], symbol, get_current_price(symbol), btc_trend_global, btc_trend_global, whale_data)
    if 'no_trade' in grok_response:
        await context.bot.send_message(chat_id=chat_id, text="Grok: Brak high-conf setup.")
        return
    grok = grok_response.get('live_trade', grok_response.get('roadmap', [{}])[0])  # Prefer live, fallback roadmap
    entry = (top + bottom) / 2
    sl = grok.get('sl', entry * 0.98)
    tp = grok.get('tp2', entry * 1.06)
    confidence = grok.get('confidence', 85)

    # Zapisz do DB
    cursor.execute('''
        INSERT INTO trades (pair, timeframe, ob_type, entry_price, sl_price, tp_price, timestamp, grok_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, '4h', ob_type_str, entry, sl, tp, now.isoformat(), confidence))
    trade_id = cursor.lastrowid
    conn.commit()

    # Roadmap (z RR, leverage)
    rr = abs(tp - entry) / abs(sl - entry)
    roadmap = f"""
🗺️ **Roadmap Instytucyjny OB ({ob_type_str}) ID: {trade_id}**
- Strefa: {bottom:.4f} - {top:.4f}
- Entry: {entry:.4f} | SL: {sl:.4f} | TP: {tp:.4f}
- Moc: str{strength} (vol >2x, {confluence})
- Confidence: {confidence}% | Leverage: {grok.get('leverage', 5)}x | R:R 1:{rr:.1f}
- {grok.get('reason', 'SMC retest')}

Aktywacja po retest. Cooldown: {COOLDOWN_HOURS}h.
    """
    await context.bot.send_message(chat_id=chat_id, text=roadmap, parse_mode='Markdown')

    # Aktywacja i process (z process_trade)
    current_price = get_current_price(symbol)
    if bottom <= current_price <= top:
        await context.bot.send_message(chat_id=chat_id, text=f"🚨 **Active Trade #{trade_id} Aktywowany!**")
        await process_trade(trade_id, current_price)
    last_signal_time[symbol] = now

# Stats (ulepszone z PnL)
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    winrate, total, total_pnl = calculate_winrate()
    cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 5")
    recent = cursor.fetchall()
    
    stats_msg = f"📊 **Winrate: {winrate:.1f}%** (z {total} tradów, PnL: {total_pnl:.2f})\n\nOstatnie trady:\n"
    for trade in recent:
        outcome_emoji = "✅" if trade[7] == 'win' else "❌" if trade[7] == 'loss' else "⏳"
        stats_msg += f"{outcome_emoji} ID:{trade[0]} {trade[3]} | Entry:{trade[4]:.4f} | PnL:{trade[8]:.2f} | Outcome:{trade[7]}\n"
    
    await context.bot.send_message(chat_id=chat_id, text=stats_msg, parse_mode='Markdown')

# Backtest cmd (z starego, 30d hit rate)
async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        since = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
        df = await fetch_ohlcv('BTC/USDT', '1h', limit=720, since=since)
        if len(df) == 0:
            await context.bot.send_message(chat_id=chat_id, text="Dane niedostępne.")
            return
        df = add_indicators(df)
        hits = 0
        total = 0
        for i in range(100, len(df)):
            # Sim OB entry
            entry = df['close'].iloc[i]
            sl = entry * 0.98
            tp = entry * 1.03
            for j in range(i+1, len(df)):
                price = df['close'].iloc[j]
                if price <= sl or price >= tp:
                    hits += 1 if price >= tp else 0
                    break
            total += 1
        winrate = (hits / total * 100) if total > 0 else 0
        msg = f"**Backtest 30d BTC/USDT**\n\nHit Rate: {winrate:.1f}% ({hits}/{total})"
        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Backtest failed: {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Witaj! /scan - skan OB, /stats - winrate, /backtest - test historyczny.")

# Background tasks (w post_init)
async def post_init(application: Application):
    asyncio.create_task(price_background_task())
    asyncio.create_task(btc_trend_update())  # Initial
    application.job_queue.run_repeating(btc_trend_update, interval=300, first=60)
    logging.info("Background tasks started.")

# Flask webhook
flask_app = Flask(__name__)
telegram_app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("scan", scan_ob))
telegram_app.add_handler(CommandHandler("stats", stats))
telegram_app.add_handler(CommandHandler("backtest", backtest_cmd))

@flask_app.route('/webhook', methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    telegram_app.process_update(update)
    return 'OK'

def run_flask():
    port = int(os.environ.get('PORT', 5000))
    flask_app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    run_flask()
