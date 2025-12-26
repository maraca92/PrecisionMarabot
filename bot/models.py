# models.py - Grok Elite Signal Bot v27.2.0 - Data Models & Persistence
"""
Data persistence layer - handles loading/saving of:
- Bot statistics
- Open trades
- Protected trades
- Historical backtest data
- Roadmap zones (NEW v27.2.0)
"""
import os
import json
import logging
import aiofiles
from datetime import datetime, timezone
from typing import Dict, Any, List

from bot.config import (
    STATS_FILE, TRADES_FILE, PROTECTED_TRADES_FILE, 
    BACKTEST_FILE, SIMULATED_CAPITAL, ROADMAP_FILE
)
from bot.utils import DateTimeEncoder

# ============================================================================
# STATISTICS PERSISTENCE
# ============================================================================
def load_stats() -> Dict[str, Any]:
    """Load bot statistics from disk."""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                s = json.load(f)
            
            if 'capital' not in s:
                s['capital'] = SIMULATED_CAPITAL
            if 'best_trade' not in s:
                s['best_trade'] = 0
            if 'worst_trade' not in s:
                s['worst_trade'] = 0
            if 'tp1_hits' not in s:
                s['tp1_hits'] = 0
            if 'roadmap_conversions' not in s:
                s['roadmap_conversions'] = 0
            if 'roadmap_skips' not in s:
                s['roadmap_skips'] = 0
            
            logging.info(f"Loaded stats: {s['wins']}W/{s['losses']}L, Capital: ${s['capital']:.2f}")
            return s
            
        except json.JSONDecodeError:
            logging.warning("Invalid stats file, resetting to defaults")
    
    return {
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "capital": SIMULATED_CAPITAL,
        "drawdown": 0.0,
        "best_trade": 0,
        "worst_trade": 0,
        "tp1_hits": 0,
        "roadmap_conversions": 0,
        "roadmap_skips": 0
    }

async def save_stats_async(s: Dict[str, Any]):
    """Save bot statistics to disk (async)."""
    try:
        async with aiofiles.open(STATS_FILE, 'w') as f:
            await f.write(json.dumps(s, indent=2))
        logging.debug(f"Stats saved: Capital ${s['capital']:.2f}")
    except Exception as e:
        logging.error(f"Failed to save stats: {e}")

# ============================================================================
# TRADES PERSISTENCE
# ============================================================================
def load_trades() -> Dict[str, Any]:
    """Load open trades from disk."""
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                loaded = json.load(f)
            
            for trade in loaded.values():
                if 'last_check' in trade and trade['last_check']:
                    trade['last_check'] = datetime.fromisoformat(trade['last_check'])
                if 'entry_time' in trade and trade['entry_time']:
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                if 'processed' not in trade:
                    trade['processed'] = False
                if 'factors' not in trade:
                    trade['factors'] = []
                if 'from_roadmap' not in trade:
                    trade['from_roadmap'] = False
            
            logging.info(f"Loaded {len(loaded)} open trades")
            return loaded
            
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid trades file, resetting: {e}")
    
    return {}

async def save_trades_async(trades: Dict[str, Any]):
    """Save open trades to disk (async)."""
    try:
        dumpable = {}
        for sym, trade in trades.items():
            t_copy = trade.copy()
            if 'last_check' in t_copy and t_copy['last_check']:
                t_copy['last_check'] = trade['last_check'].isoformat() if isinstance(trade['last_check'], datetime) else trade['last_check']
            if 'entry_time' in t_copy and t_copy['entry_time']:
                t_copy['entry_time'] = trade['entry_time'].isoformat() if isinstance(trade['entry_time'], datetime) else trade['entry_time']
            dumpable[sym] = t_copy
        
        async with aiofiles.open(TRADES_FILE, 'w') as f:
            await f.write(json.dumps(dumpable, indent=2, cls=DateTimeEncoder))
        
        logging.debug(f"Saved {len(trades)} open trades")
    except Exception as e:
        logging.error(f"Failed to save trades: {e}")

# ============================================================================
# PROTECTED TRADES PERSISTENCE
# ============================================================================
def load_protected() -> Dict[str, Any]:
    """Load protected trades from disk."""
    if os.path.exists(PROTECTED_TRADES_FILE):
        try:
            with open(PROTECTED_TRADES_FILE, 'r') as f:
                loaded = json.load(f)
            
            for trade in loaded.values():
                if 'last_check' in trade and trade['last_check']:
                    trade['last_check'] = datetime.fromisoformat(trade['last_check'])
                if 'entry_time' in trade and trade['entry_time']:
                    trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                if 'processed' not in trade:
                    trade['processed'] = False
                if 'factors' not in trade:
                    trade['factors'] = []
            
            logging.info(f"Loaded {len(loaded)} protected trades")
            return loaded
            
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid protected trades file, resetting: {e}")
    
    return {}

async def save_protected_async(trades: Dict[str, Any]):
    """Save protected trades to disk (async)."""
    try:
        dumpable = {}
        for sym, trade in trades.items():
            t_copy = trade.copy()
            if 'last_check' in t_copy and t_copy['last_check']:
                t_copy['last_check'] = trade['last_check'].isoformat() if isinstance(trade['last_check'], datetime) else trade['last_check']
            if 'entry_time' in t_copy and t_copy['entry_time']:
                t_copy['entry_time'] = trade['entry_time'].isoformat() if isinstance(trade['entry_time'], datetime) else trade['entry_time']
            dumpable[sym] = t_copy
        
        async with aiofiles.open(PROTECTED_TRADES_FILE, 'w') as f:
            await f.write(json.dumps(dumpable, indent=2, cls=DateTimeEncoder))
        
        logging.debug(f"Saved {len(trades)} protected trades")
    except Exception as e:
        logging.error(f"Failed to save protected trades: {e}")

# ============================================================================
# ROADMAP ZONES PERSISTENCE (NEW v27.2.0)
# ============================================================================
def load_roadmap_zones() -> Dict[str, List[Dict]]:
    """
    Load roadmap zones from disk.
    
    Returns:
        Dict mapping symbol -> list of zone dicts
    """
    if os.path.exists(ROADMAP_FILE):
        try:
            with open(ROADMAP_FILE, 'r') as f:
                loaded = json.load(f)
            
            for symbol, zones in loaded.items():
                for zone in zones:
                    if 'created_at' in zone and zone['created_at']:
                        zone['created_at'] = datetime.fromisoformat(zone['created_at'])
                    if 'last_alert' in zone and zone['last_alert']:
                        zone['last_alert'] = datetime.fromisoformat(zone['last_alert'])
            
            total_zones = sum(len(zones) for zones in loaded.values())
            logging.info(f"Loaded {total_zones} roadmap zones for {len(loaded)} symbols")
            return loaded
            
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Invalid roadmap file, resetting: {e}")
    
    return {}

async def save_roadmap_zones_async(zones: Dict[str, List[Dict]]):
    """
    Save roadmap zones to disk (async).
    
    Args:
        zones: Dict mapping symbol -> list of zone dicts
    """
    try:
        dumpable = {}
        for symbol, zone_list in zones.items():
            dumpable[symbol] = []
            for zone in zone_list:
                z_copy = zone.copy()
                if 'created_at' in z_copy and z_copy['created_at']:
                    z_copy['created_at'] = zone['created_at'].isoformat() if isinstance(zone['created_at'], datetime) else zone['created_at']
                if 'last_alert' in z_copy and z_copy['last_alert']:
                    z_copy['last_alert'] = zone['last_alert'].isoformat() if isinstance(zone['last_alert'], datetime) else zone['last_alert']
                dumpable[symbol].append(z_copy)
        
        async with aiofiles.open(ROADMAP_FILE, 'w') as f:
            await f.write(json.dumps(dumpable, indent=2, cls=DateTimeEncoder))
        
        total_zones = sum(len(zl) for zl in zones.values())
        logging.debug(f"Saved {total_zones} roadmap zones")
    except Exception as e:
        logging.error(f"Failed to save roadmap zones: {e}")

def clear_expired_roadmap_zones(zones: Dict[str, List[Dict]], max_age_hours: int = 48) -> Dict[str, List[Dict]]:
    """
    Remove expired roadmap zones.
    
    Args:
        zones: Current roadmap zones
        max_age_hours: Maximum age before expiry
    
    Returns:
        Cleaned zones dict
    """
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    cleaned = {}
    
    for symbol, zone_list in zones.items():
        valid_zones = []
        for zone in zone_list:
            created_at = zone.get('created_at')
            if isinstance(created_at, datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                
                if now - created_at < timedelta(hours=max_age_hours):
                    valid_zones.append(zone)
        
        if valid_zones:
            cleaned[symbol] = valid_zones
    
    return cleaned


# ============================================================================
# v27.12.13: SIGNAL PERFORMANCE TRACKING
# ============================================================================

# Try to import factor performance file path
try:
    from bot.config import FACTOR_PERFORMANCE_FILE
except ImportError:
    FACTOR_PERFORMANCE_FILE = 'data/factor_performance.json'


class SignalTracker:
    """
    Track which confluence factors produce winning trades.
    
    Records signal outcomes and calculates win rate by factor
    to enable data-driven strategy improvements.
    """
    
    def __init__(self):
        self.performance_data = self._load_performance()
    
    def _load_performance(self) -> Dict:
        """Load factor performance data from disk."""
        if os.path.exists(FACTOR_PERFORMANCE_FILE):
            try:
                with open(FACTOR_PERFORMANCE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load factor performance: {e}")
        
        return {
            'factors': {},
            'total_signals': 0,
            'total_wins': 0,
            'total_losses': 0,
            'last_updated': None
        }
    
    async def save_performance(self):
        """Save factor performance data to disk."""
        try:
            self.performance_data['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(FACTOR_PERFORMANCE_FILE), exist_ok=True)
            
            async with aiofiles.open(FACTOR_PERFORMANCE_FILE, 'w') as f:
                await f.write(json.dumps(self.performance_data, indent=2))
        except Exception as e:
            logging.error(f"Failed to save factor performance: {e}")
    
    def record_signal(self, factors: List[str], outcome: str, pnl: float = 0):
        """
        Record a signal outcome for performance tracking.
        
        Args:
            factors: List of confluence factors present in the signal
            outcome: 'TP1', 'TP2', 'SL', 'TIMEOUT', 'BREAKEVEN'
            pnl: Profit/loss in percentage
        """
        is_win = outcome in ['TP1', 'TP2', 'BREAKEVEN']
        is_loss = outcome == 'SL'
        
        self.performance_data['total_signals'] += 1
        
        if is_win:
            self.performance_data['total_wins'] += 1
        elif is_loss:
            self.performance_data['total_losses'] += 1
        
        # Update each factor's statistics
        for factor in factors:
            if factor not in self.performance_data['factors']:
                self.performance_data['factors'][factor] = {
                    'signals': 0,
                    'wins': 0,
                    'losses': 0,
                    'tp1_hits': 0,
                    'tp2_hits': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            
            f_data = self.performance_data['factors'][factor]
            f_data['signals'] += 1
            f_data['total_pnl'] += pnl
            
            if outcome == 'TP1':
                f_data['wins'] += 1
                f_data['tp1_hits'] += 1
            elif outcome == 'TP2':
                f_data['wins'] += 1
                f_data['tp2_hits'] += 1
            elif outcome == 'BREAKEVEN':
                f_data['wins'] += 1
            elif is_loss:
                f_data['losses'] += 1
            
            # Calculate win rate
            if f_data['signals'] > 0:
                f_data['win_rate'] = f_data['wins'] / f_data['signals'] * 100
    
    def get_factor_rankings(self, min_signals: int = 5) -> List[Dict]:
        """
        Get factors ranked by win rate.
        
        Args:
            min_signals: Minimum signals required for ranking
        
        Returns:
            List of factors sorted by win rate
        """
        rankings = []
        
        for factor, data in self.performance_data['factors'].items():
            if data['signals'] >= min_signals:
                rankings.append({
                    'factor': factor,
                    'signals': data['signals'],
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'win_rate': data['win_rate'],
                    'avg_pnl': data['total_pnl'] / data['signals'] if data['signals'] > 0 else 0,
                    'tp2_rate': data['tp2_hits'] / data['signals'] * 100 if data['signals'] > 0 else 0
                })
        
        return sorted(rankings, key=lambda x: x['win_rate'], reverse=True)
    
    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics."""
        total = self.performance_data['total_signals']
        wins = self.performance_data['total_wins']
        losses = self.performance_data['total_losses']
        
        return {
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total * 100 if total > 0 else 0,
            'factors_tracked': len(self.performance_data['factors']),
            'last_updated': self.performance_data.get('last_updated')
        }


# Global signal tracker instance
_signal_tracker: SignalTracker = None


def get_signal_tracker() -> SignalTracker:
    """Get or create the signal tracker singleton."""
    global _signal_tracker
    if _signal_tracker is None:
        _signal_tracker = SignalTracker()
    return _signal_tracker


# ============================================================================
# HISTORICAL DATA
# ============================================================================
def load_historical_data() -> Dict[str, float]:
    """Load historical TP hit rates from backtest results."""
    if os.path.exists(BACKTEST_FILE):
        try:
            with open(BACKTEST_FILE, 'r') as f:
                bt = json.load(f)
            
            tp1_rate = bt.get('tp1_hit_rate', 0.60)
            tp2_rate = bt.get('tp2_hit_rate', 0.35)
            
            logging.info(f"Loaded historical data: TP1={tp1_rate:.2%}, TP2={tp2_rate:.2%}")
            return {'tp1_hit_rate': tp1_rate, 'tp2_hit_rate': tp2_rate}
            
        except Exception as e:
            logging.warning(f"Failed to load historical data: {e}")
    
    return {'tp1_hit_rate': 0.60, 'tp2_hit_rate': 0.35}

# Global state
HISTORICAL_DATA = load_historical_data()
