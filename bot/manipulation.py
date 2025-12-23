# manipulation.py - Grok Elite Signal Bot v27.12.10 - Institutional Manipulation Detection
# -*- coding: utf-8 -*-
"""
Detect and exploit institutional price manipulation patterns.

v27.12.10: Verified complete with all detection methods
v27.5.1: Fixed memory leak, added safety bounds
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from bot.config import ORDERBOOK_DEPTH, MANIPULATION_DETECTION_ENABLED


class ManipulationDetector:
    """
    Detects institutional manipulation patterns:
    - Spoofing (fake order book walls)
    - Stop hunts (liquidity sweeps)
    - Wash trading (artificial volume)
    - Coordinated moves (pump/dump)
    """
    
    def __init__(self):
        self.orderbook_history: Dict[str, List[Dict]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.max_history_length = 50
        self.history_ttl_minutes = 30
    
    def _cleanup_old_history(self, symbol: str):
        """Remove old history entries beyond TTL."""
        if symbol not in self.orderbook_history:
            return
        
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.history_ttl_minutes)
        self.orderbook_history[symbol] = [
            h for h in self.orderbook_history[symbol]
            if h.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) > cutoff
        ]
    
    def detect_spoofing(
        self,
        symbol: str,
        current_orderbook: Dict,
        price: float
    ) -> Optional[Dict]:
        """
        Detect spoofing - fake order book walls that disappear before being filled.
        """
        if symbol not in self.orderbook_history:
            self.orderbook_history[symbol] = []
        
        # Store current snapshot
        snapshot = {
            'timestamp': datetime.now(timezone.utc),
            'bids': current_orderbook.get('bids', [])[:ORDERBOOK_DEPTH],
            'asks': current_orderbook.get('asks', [])[:ORDERBOOK_DEPTH],
            'price': price
        }
        self.orderbook_history[symbol].append(snapshot)
        
        # Cleanup old entries
        if len(self.orderbook_history[symbol]) > self.max_history_length:
            self.orderbook_history[symbol] = self.orderbook_history[symbol][-self.max_history_length:]
        
        # Need minimum history
        if len(self.orderbook_history[symbol]) < 10:
            return None
        
        history = self.orderbook_history[symbol]
        
        # Check recent snapshots for disappeared walls
        for i in range(-5, -1):
            if abs(i) >= len(history):
                continue
            
            prev_snapshot = history[i]
            curr_snapshot = history[-1]
            
            # Check for disappeared bid walls (potential bearish spoofing)
            if prev_snapshot.get('bids'):
                prev_bids = prev_snapshot['bids']
                avg_bid_size = np.mean([amt for _, amt in prev_bids]) if prev_bids else 0
                
                # Find large bids (3x average)
                large_bids = [(p, amt) for p, amt in prev_bids if amt > avg_bid_size * 3]
                
                if large_bids:
                    curr_bid_prices = [p for p, _ in curr_snapshot.get('bids', [])]
                    
                    # Check if large bids disappeared without being filled
                    disappeared_bids = [
                        (p, amt) for p, amt in large_bids
                        if p not in curr_bid_prices and price > p * 1.001
                    ]
                    
                    if disappeared_bids:
                        return {
                            'type': 'spoofing',
                            'direction': 'Short',
                            'confidence': 70,
                            'reason': f"Fake bid wall disappeared at {disappeared_bids[0][0]:.4f}",
                            'wall_size': disappeared_bids[0][1],
                            'manipulation_score': 0.8
                        }
            
            # Check for disappeared ask walls (potential bullish spoofing)
            if prev_snapshot.get('asks'):
                prev_asks = prev_snapshot['asks']
                avg_ask_size = np.mean([amt for _, amt in prev_asks]) if prev_asks else 0
                
                # Find large asks (3x average)
                large_asks = [(p, amt) for p, amt in prev_asks if amt > avg_ask_size * 3]
                
                if large_asks:
                    curr_ask_prices = [p for p, _ in curr_snapshot.get('asks', [])]
                    
                    # Check if large asks disappeared without being filled
                    disappeared_asks = [
                        (p, amt) for p, amt in large_asks
                        if p not in curr_ask_prices and price < p * 0.999
                    ]
                    
                    if disappeared_asks:
                        return {
                            'type': 'spoofing',
                            'direction': 'Long',
                            'confidence': 70,
                            'reason': f"Fake ask wall disappeared at {disappeared_asks[0][0]:.4f}",
                            'wall_size': disappeared_asks[0][1],
                            'manipulation_score': 0.8
                        }
        
        return None
    
    def detect_stop_hunt(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_price: float
    ) -> Optional[Dict]:
        """
        Detect stop hunt - quick sweep of liquidity followed by reversal.
        """
        if len(df) < 20:
            return None
        
        recent = df.tail(20).copy()
        
        # Calculate swing levels
        swing_high = recent['high'].rolling(5, center=True).max()
        swing_low = recent['low'].rolling(5, center=True).min()
        
        # Check recent candles for sweep pattern
        for i in range(-3, 0):
            if abs(i) > len(recent):
                continue
                
            candle = recent.iloc[i]
            
            # Bullish stop hunt (sweep lows, reverse up)
            try:
                swing_ref = swing_low.iloc[i-3] if (i-3) >= -len(swing_low) else swing_low.iloc[0]
                if candle['low'] < swing_ref * 0.998:
                    vol_surge = candle['volume'] > recent['volume'].mean() * 1.5
                    
                    if vol_surge and candle['close'] > candle['open']:
                        if current_price > candle['high']:
                            return {
                                'type': 'stop_hunt',
                                'direction': 'Long',
                                'confidence': 75,
                                'reason': f"Stop hunt below ${swing_ref:.4f}, strong reversal",
                                'sweep_low': candle['low'],
                                'manipulation_score': 0.85
                            }
            except (IndexError, KeyError):
                pass
            
            # Bearish stop hunt (sweep highs, reverse down)
            try:
                swing_ref = swing_high.iloc[i-3] if (i-3) >= -len(swing_high) else swing_high.iloc[0]
                if candle['high'] > swing_ref * 1.002:
                    vol_surge = candle['volume'] > recent['volume'].mean() * 1.5
                    
                    if vol_surge and candle['close'] < candle['open']:
                        if current_price < candle['low']:
                            return {
                                'type': 'stop_hunt',
                                'direction': 'Short',
                                'confidence': 75,
                                'reason': f"Stop hunt above ${swing_ref:.4f}, strong reversal",
                                'sweep_high': candle['high'],
                                'manipulation_score': 0.85
                            }
            except (IndexError, KeyError):
                pass
        
        return None
    
    def detect_wash_trading(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Detect wash trading - artificially inflated volume without real price movement.
        """
        if len(df) < 50:
            return None
        
        recent = df.tail(50).copy()
        
        # Calculate price change vs volume ratio
        recent['price_change'] = abs(recent['close'] - recent['open']) / recent['open'] * 100
        recent['vol_to_movement'] = recent['volume'] / (recent['price_change'] + 0.01)
        
        # High volume with minimal movement = potential wash
        avg_ratio = recent['vol_to_movement'].mean()
        std_ratio = recent['vol_to_movement'].std()
        
        # Check last 5 candles
        for i in range(-5, 0):
            candle = recent.iloc[i]
            
            # Extremely high volume with minimal movement
            if candle['vol_to_movement'] > avg_ratio + 3 * std_ratio:
                if candle['price_change'] < 0.2:  # Less than 0.2% movement
                    return {
                        'type': 'wash_trading',
                        'direction': 'neutral',
                        'confidence': 60,
                        'reason': f"Suspicious volume spike ({candle['volume']:,.0f}) with minimal movement",
                        'volume': candle['volume'],
                        'manipulation_score': 0.6
                    }
        
        return None
    
    def detect_coordinated_move(
        self,
        symbol: str,
        df: pd.DataFrame,
        orderbook: Dict
    ) -> Optional[Dict]:
        """
        Detect coordinated pump/dump - sudden large moves with volume surge.
        """
        if len(df) < 10:
            return None
        
        recent = df.tail(10).copy()
        
        # Calculate 5-minute equivalent price change (rough)
        price_change_5m = (recent['close'].iloc[-1] / recent['close'].iloc[-2] - 1) * 100
        
        # Calculate volume surge
        avg_vol = recent['volume'].iloc[:-1].mean()
        last_vol = recent['volume'].iloc[-1]
        vol_surge_factor = last_vol / avg_vol if avg_vol > 0 else 1
        
        # Detect pump (>2% move with >2x volume)
        if abs(price_change_5m) > 2.0 and vol_surge_factor > 2.0:
            # Check orderbook for thin liquidity
            total_bid_vol = sum(amt for _, amt in orderbook.get('bids', [])[:10])
            total_ask_vol = sum(amt for _, amt in orderbook.get('asks', [])[:10])
            
            # Imbalanced orderbook after big move = likely manipulation
            if total_bid_vol > 0 and total_ask_vol > 0:
                imbalance = abs(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
                
                if imbalance > 0.4:  # 40% imbalance
                    return {
                        'type': 'coordinated_move',
                        'direction': 'Short' if price_change_5m > 0 else 'Long',
                        'confidence': 65,
                        'reason': f"{'Pump' if price_change_5m > 0 else 'Dump'} detected ({price_change_5m:+.1f}% move)",
                        'move_size': price_change_5m,
                        'volume_factor': vol_surge_factor,
                        'manipulation_score': 0.75
                    }
        
        return None
    
    def detect_all_manipulations(
        self,
        symbol: str,
        df: pd.DataFrame,
        orderbook: Dict,
        current_price: float
    ) -> List[Dict]:
        """
        Run all manipulation detection methods.
        """
        if not MANIPULATION_DETECTION_ENABLED:
            return []
        
        manipulations = []
        
        # Run each detector
        spoofing = self.detect_spoofing(symbol, orderbook, current_price)
        if spoofing:
            manipulations.append(spoofing)
        
        stop_hunt = self.detect_stop_hunt(symbol, df, current_price)
        if stop_hunt:
            manipulations.append(stop_hunt)
        
        wash = self.detect_wash_trading(symbol, df)
        if wash:
            manipulations.append(wash)
        
        coord = self.detect_coordinated_move(symbol, df, orderbook)
        if coord:
            manipulations.append(coord)
        
        return manipulations
    
    def get_manipulation_confluence(
        self,
        symbol: str,
        trade_direction: str,
        df: pd.DataFrame,
        orderbook: Dict,
        current_price: float
    ) -> Tuple[float, str]:
        """
        Get manipulation confluence for a trade.
        
        Args:
            symbol: Trading pair
            trade_direction: 'Long' or 'Short'
            df: OHLCV DataFrame
            orderbook: Current orderbook
            current_price: Current price
        
        Returns:
            (confidence_boost, reason_string)
        """
        if not MANIPULATION_DETECTION_ENABLED:
            return 0.0, ""
        
        manipulations = self.detect_all_manipulations(symbol, df, orderbook, current_price)
        
        if not manipulations:
            return 0.0, ""
        
        # Find manipulations aligned with trade direction
        aligned_manipulations = [
            m for m in manipulations
            if m.get('direction') == trade_direction
        ]
        
        if not aligned_manipulations:
            return 0.0, ""
        
        # Get best manipulation signal
        best = max(aligned_manipulations, key=lambda m: m.get('manipulation_score', 0))
        
        confidence_boost = best['manipulation_score'] * 10
        reason = f"+{best['type'].replace('_', ' ').title()}"
        
        return confidence_boost, reason


# Create singleton instance
manipulation_detector = ManipulationDetector()
