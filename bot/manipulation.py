# manipulation.py - Grok Elite Signal Bot v27.5.1 - Institutional Manipulation Detection
"""
Detect and exploit institutional price manipulation patterns.
v27.5.1: Fixed memory leak, added safety bounds
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from bot.config import ORDERBOOK_DEPTH, MANIPULATION_DETECTION_ENABLED

class ManipulationDetector:
    
    def __init__(self):
        self.orderbook_history: Dict[str, List[Dict]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.max_history_length = 50
    
    def detect_spoofing(
        self,
        symbol: str,
        current_orderbook: Dict,
        price: float
    ) -> Optional[Dict]:
        if symbol not in self.orderbook_history:
            self.orderbook_history[symbol] = []
        
        snapshot = {
            'timestamp': datetime.now(timezone.utc),
            'bids': current_orderbook.get('bids', [])[:ORDERBOOK_DEPTH],
            'asks': current_orderbook.get('asks', [])[:ORDERBOOK_DEPTH],
            'price': price
        }
        self.orderbook_history[symbol].append(snapshot)
        
        if len(self.orderbook_history[symbol]) > self.max_history_length:
            self.orderbook_history[symbol] = self.orderbook_history[symbol][-self.max_history_length:]
        
        if len(self.orderbook_history[symbol]) < 10:
            return None
        
        history = self.orderbook_history[symbol]
        
        for i in range(-5, -1):
            if abs(i) >= len(history):
                continue
            
            prev_snapshot = history[i]
            curr_snapshot = history[-1]
            
            if prev_snapshot['bids']:
                prev_bids = prev_snapshot['bids']
                avg_bid_size = np.mean([amt for _, amt in prev_bids]) if prev_bids else 0
                
                large_bids = [(p, amt) for p, amt in prev_bids if amt > avg_bid_size * 3]
                
                if large_bids:
                    curr_bid_prices = [p for p, _ in curr_snapshot['bids']]
                    
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
            
            if prev_snapshot['asks']:
                prev_asks = prev_snapshot['asks']
                avg_ask_size = np.mean([amt for _, amt in prev_asks]) if prev_asks else 0
                
                large_asks = [(p, amt) for p, amt in prev_asks if amt > avg_ask_size * 3]
                
                if large_asks:
                    curr_ask_prices = [p for p, _ in curr_snapshot['asks']]
                    
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
        if len(df) < 20:
            return None
        
        recent = df.tail(20).copy()
        
        swing_high = recent['high'].rolling(5, center=True).max()
        swing_low = recent['low'].rolling(5, center=True).min()
        
        for i in range(-3, 0):
            candle = recent.iloc[i]
            
            if i < -1:
                if candle['low'] < swing_low.iloc[i-3] * 0.998:
                    vol_surge = candle['volume'] > recent['volume'].mean() * 1.5
                    
                    if vol_surge:
                        if candle['close'] > candle['open']:
                            if current_price > candle['high']:
                                return {
                                    'type': 'stop_hunt',
                                    'direction': 'Long',
                                    'confidence': 75,
                                    'reason': f"Stop hunt below ${swing_low.iloc[i-3]:.4f}, strong reversal",
                                    'sweep_low': candle['low'],
                                    'manipulation_score': 0.85
                                }
            
            if i < -1:
                if candle['high'] > swing_high.iloc[i-3] * 1.002:
                    vol_surge = candle['volume'] > recent['volume'].mean() * 1.5
                    
                    if vol_surge:
                        if candle['close'] < candle['open']:
                            if current_price < candle['low']:
                                return {
                                    'type': 'stop_hunt',
                                    'direction': 'Short',
                                    'confidence': 75,
                                    'reason': f"Stop hunt above ${swing_high.iloc[i-3]:.4f}, strong reversal",
                                    'sweep_high': candle['high'],
                                    'manipulation_score': 0.85
                                }
        
        return None
    
    def detect_wash_trading(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[Dict]:
        if len(df) < 50:
            return None
        
        recent = df.tail(50).copy()
        
        recent['price_change'] = abs(recent['close'] - recent['open']) / recent['open'] * 100
        recent['vol_to_movement'] = recent['volume'] / (recent['price_change'] + 0.01)
        
        normal_ratio = recent['vol_to_movement'].median()
        
        for i in range(-5, 0):
            candle = recent.iloc[i]
            ratio = candle['vol_to_movement']
            
            if ratio > normal_ratio * 3:
                if candle['price_change'] < 0.5:
                    return {
                        'type': 'wash_trading',
                        'direction': None,
                        'confidence': 60,
                        'reason': f"Suspicious volume spike with minimal price movement ({candle['price_change']:.2f}%)",
                        'volume_ratio': ratio / normal_ratio,
                        'manipulation_score': 0.6,
                        'action': 'avoid'
                    }
        
        return None
    
    def detect_coordinated_move(
        self,
        symbol: str,
        df: pd.DataFrame,
        orderbook: Dict
    ) -> Optional[Dict]:
        if len(df) < 10:
            return None
        
        recent = df.tail(10).copy()
        
        price_change_5m = (recent['close'].iloc[-1] - recent['close'].iloc[-5]) / recent['close'].iloc[-5] * 100
        
        if abs(price_change_5m) > 1.5:
            recent_vol = recent['volume'].iloc[-5:].mean()
            normal_vol = recent['volume'].iloc[-10:-5].mean()
            
            vol_surge_factor = recent_vol / normal_vol if normal_vol > 0 else 1
            
            if vol_surge_factor > 2:
                if orderbook.get('bids') and orderbook.get('asks'):
                    bid_depth = sum(amt for _, amt in orderbook['bids'][:10])
                    ask_depth = sum(amt for _, amt in orderbook['asks'][:10])
                    
                    avg_depth = (bid_depth + ask_depth) / 2
                    
                    if avg_depth < 50:
                        direction = 'Short' if price_change_5m > 0 else 'Long'
                        
                        return {
                            'type': 'coordinated_move',
                            'direction': direction,
                            'confidence': 70,
                            'reason': f"Coordinated {'pump' if price_change_5m > 0 else 'dump'} detected ({price_change_5m:+.1f}% move)",
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
        if not MANIPULATION_DETECTION_ENABLED:
            return []
        
        manipulations = []
        
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
        if not MANIPULATION_DETECTION_ENABLED:
            return 0.0, ""
        
        manipulations = self.detect_all_manipulations(symbol, df, orderbook, current_price)
        
        if not manipulations:
            return 0.0, ""
        
        aligned_manipulations = [
            m for m in manipulations
            if m.get('direction') == trade_direction
        ]
        
        if not aligned_manipulations:
            return 0.0, ""
        
        best = max(aligned_manipulations, key=lambda m: m.get('manipulation_score', 0))
        
        confidence_boost = best['manipulation_score'] * 10
        reason = f"+{best['type'].replace('_', ' ').title()}"
        
        return confidence_boost, reason

manipulation_detector = ManipulationDetector()
