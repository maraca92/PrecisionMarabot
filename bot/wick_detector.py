# wick_detector.py - Grok Elite Signal Bot v27.11.0 - Euphoria/Capitulation Detection
"""
Detects high-probability reversal signals based on rejection candle wicks:
- Euphoria tops: Long upper wicks after uptrends (bearish reversal)
- Capitulation bottoms: Long lower wicks after downtrends (bullish reversal)

Historical data shows these patterns are highly reliable reversal signals when
combined with proper confluence filters (volume, structure, psychology).

Author: Enhanced for SMC/ICT methodology
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class WickType(Enum):
    """Classification of wick rejection patterns."""
    EUPHORIA_TOP = "euphoria_top"
    CAPITULATION_BOTTOM = "capitulation_bottom"
    DISTRIBUTION_WICK = "distribution_wick"  # Multiple upper wicks
    ACCUMULATION_WICK = "accumulation_wick"  # Multiple lower wicks
    NONE = "none"


class WickStrength(Enum):
    """Strength classification for detected wicks."""
    EXTREME = "extreme"      # Wick ratio > 4x, volume > 2.5x
    STRONG = "strong"        # Wick ratio > 3x, volume > 2x
    MODERATE = "moderate"    # Wick ratio > 2.5x, volume > 1.5x
    WEAK = "weak"            # Meets minimum criteria only


@dataclass
class WickDetectionResult:
    """
    Result of wick pattern detection analysis.
    
    Attributes:
        detected: Whether a significant wick pattern was found
        wick_type: Classification of the wick pattern
        direction: Suggested trade direction ('Long' or 'Short')
        strength: Strength classification of the signal
        confidence: Confidence score (0-100)
        wick_ratio: Ratio of wick to body size
        volume_ratio: Ratio of candle volume to average
        rejection_price: Price level where rejection occurred
        close_position: Where close is relative to candle range (0-1)
        trend_context: Description of preceding trend
        confluence_factors: List of supporting factors
        manipulation_score: Score similar to other manipulation detections (0-10)
        reason: Human-readable explanation
        candle_data: Raw data of the rejection candle(s)
        timestamp: Detection timestamp
    """
    detected: bool = False
    wick_type: WickType = WickType.NONE
    direction: str = ""
    strength: WickStrength = WickStrength.WEAK
    confidence: float = 0.0
    wick_ratio: float = 0.0
    volume_ratio: float = 0.0
    rejection_price: float = 0.0
    close_position: float = 0.5
    trend_context: str = ""
    confluence_factors: List[str] = field(default_factory=list)
    manipulation_score: float = 0.0
    reason: str = ""
    candle_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "detected": self.detected,
            "wick_type": self.wick_type.value,
            "direction": self.direction,
            "strength": self.strength.value,
            "confidence": round(self.confidence, 1),
            "wick_ratio": round(self.wick_ratio, 2),
            "volume_ratio": round(self.volume_ratio, 2),
            "rejection_price": self.rejection_price,
            "close_position": round(self.close_position, 3),
            "trend_context": self.trend_context,
            "confluence_factors": self.confluence_factors,
            "manipulation_score": round(self.manipulation_score, 1),
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class WickDetectorConfig:
    """
    Configuration parameters for wick detection.
    Tuned for crypto markets with high volatility.
    """
    # Wick ratio thresholds
    min_wick_body_ratio: float = 2.5        # Minimum wick-to-body ratio
    strong_wick_ratio: float = 3.0          # Strong signal threshold
    extreme_wick_ratio: float = 4.0         # Extreme signal threshold
    
    # Close position thresholds (0 = low, 1 = high)
    euphoria_close_max: float = 0.4         # Close must be in lower 40% for euphoria
    capitulation_close_min: float = 0.6     # Close must be in upper 40% for capitulation
    
    # Volume thresholds
    min_volume_ratio: float = 1.5           # Minimum volume surge
    strong_volume_ratio: float = 2.0        # Strong volume confirmation
    extreme_volume_ratio: float = 2.5       # Extreme volume (capitulation)
    
    # Trend analysis
    trend_lookback: int = 20                # Candles to analyze for trend
    min_trend_strength: float = 0.6         # Minimum trend strength (0-1)
    
    # Sequential wick detection
    sequential_lookback: int = 3            # Candles to check for sequential wicks
    sequential_wick_ratio: float = 2.0      # Lower threshold for sequential
    
    # Confidence adjustments
    base_confidence: float = 65.0           # Starting confidence
    max_confidence: float = 95.0            # Cap confidence
    
    # Body size filter (avoid dojis being misclassified)
    min_body_percent: float = 0.05          # Body must be >5% of candle range


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class WickDetector:
    """
    Detects euphoria tops and capitulation bottoms using wick analysis.
    
    This detector identifies high-probability reversal points by analyzing:
    1. Wick-to-body ratios (rejection strength)
    2. Close position within candle (commitment)
    3. Volume confirmation (participation)
    4. Trend context (exhaustion after strong move)
    5. Sequential patterns (distribution/accumulation)
    
    Designed for integration with SMC/ICT trading methodology.
    """
    
    def __init__(self, config: Optional[WickDetectorConfig] = None):
        """
        Initialize the wick detector.
        
        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or WickDetectorConfig()
        logger.info(f"WickDetector initialized with config: min_ratio={self.config.min_wick_body_ratio}")
    
    def detect_wick_reversal(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        current_price: Optional[float] = None,
        htf_bias: Optional[str] = None,
        order_blocks: Optional[List[Dict]] = None,
        stop_hunt_result: Optional[Dict] = None,
        fear_greed: Optional[Dict] = None,
        structure_break: Optional[Dict] = None
    ) -> WickDetectionResult:
        """
        Main detection function for euphoria/capitulation wicks.
        
        Args:
            df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            symbol: Trading pair symbol for logging
            current_price: Current market price for context
            htf_bias: Higher timeframe bias ('bullish', 'bearish', 'neutral')
            order_blocks: List of nearby order blocks for confluence
            stop_hunt_result: Result from stop hunt detection for confluence
            fear_greed: Fear & Greed index data for psychology alignment
            structure_break: Structure break data for alignment
            
        Returns:
            WickDetectionResult with full analysis
        """
        result = WickDetectionResult()
        
        # Validate input
        if df is None or len(df) < self.config.trend_lookback + 5:
            logger.warning(f"[{symbol}] Insufficient data for wick detection")
            return result
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"[{symbol}] Missing required columns in DataFrame")
            return result
        
        try:
            # Get recent candles (last 3 closed + current if available)
            recent_df = df.tail(self.config.sequential_lookback + 1).copy()
            
            # Analyze each recent candle for wick patterns
            wick_analyses = []
            for i in range(len(recent_df)):
                candle = recent_df.iloc[i]
                analysis = self._analyze_single_candle(candle, df, i)
                if analysis['has_significant_wick']:
                    wick_analyses.append(analysis)
            
            if not wick_analyses:
                logger.debug(f"[{symbol}] No significant wicks detected in recent candles")
                return result
            
            # Get the strongest wick signal
            best_wick = max(wick_analyses, key=lambda x: x['wick_score'])
            
            # Determine wick type based on direction
            if best_wick['wick_direction'] == 'upper':
                wick_type = WickType.EUPHORIA_TOP
                direction = "Short"
            else:
                wick_type = WickType.CAPITULATION_BOTTOM
                direction = "Long"
            
            # Calculate trend context
            trend_info = self._analyze_trend_context(df, wick_type)
            
            # Check for sequential wicks (distribution/accumulation)
            sequential_info = self._detect_sequential_wicks(recent_df, best_wick['wick_direction'])
            
            # Calculate volume confirmation
            volume_ratio = self._calculate_volume_ratio(best_wick['volume'], df)
            
            # Build confluence factors
            confluence_factors = []
            confidence = self.config.base_confidence
            manipulation_score = 0.0
            
            # Factor 1: Wick ratio strength
            if best_wick['wick_ratio'] >= self.config.extreme_wick_ratio:
                confluence_factors.append(f"Extreme wick ratio ({best_wick['wick_ratio']:.1f}x)")
                confidence += 12
                manipulation_score += 3
            elif best_wick['wick_ratio'] >= self.config.strong_wick_ratio:
                confluence_factors.append(f"Strong wick ratio ({best_wick['wick_ratio']:.1f}x)")
                confidence += 8
                manipulation_score += 2
            else:
                confluence_factors.append(f"Moderate wick ratio ({best_wick['wick_ratio']:.1f}x)")
                confidence += 4
                manipulation_score += 1
            
            # Factor 2: Close position (commitment)
            close_pos = best_wick['close_position']
            if wick_type == WickType.EUPHORIA_TOP and close_pos < 0.3:
                confluence_factors.append(f"Strong bearish close (lower {close_pos*100:.0f}%)")
                confidence += 8
                manipulation_score += 1.5
            elif wick_type == WickType.CAPITULATION_BOTTOM and close_pos > 0.7:
                confluence_factors.append(f"Strong bullish close (upper {(1-close_pos)*100:.0f}%)")
                confidence += 8
                manipulation_score += 1.5
            elif self._is_valid_close_position(close_pos, wick_type):
                confluence_factors.append("Valid close position")
                confidence += 4
                manipulation_score += 0.5
            
            # Factor 3: Volume confirmation
            if volume_ratio >= self.config.extreme_volume_ratio:
                confluence_factors.append(f"Extreme volume surge ({volume_ratio:.1f}x avg)")
                confidence += 10
                manipulation_score += 2
            elif volume_ratio >= self.config.strong_volume_ratio:
                confluence_factors.append(f"Strong volume ({volume_ratio:.1f}x avg)")
                confidence += 6
                manipulation_score += 1.5
            elif volume_ratio >= self.config.min_volume_ratio:
                confluence_factors.append(f"Volume confirmation ({volume_ratio:.1f}x avg)")
                confidence += 3
                manipulation_score += 1
            
            # Factor 4: Trend context (exhaustion)
            if trend_info['is_exhaustion_context']:
                confluence_factors.append(f"Trend exhaustion ({trend_info['description']})")
                confidence += 8
                manipulation_score += 1.5
            elif trend_info['trend_strength'] > 0.5:
                confluence_factors.append(f"Trend: {trend_info['description']}")
                confidence += 4
                manipulation_score += 0.5
            
            # Factor 5: Sequential wicks (distribution/accumulation)
            if sequential_info['detected']:
                if sequential_info['count'] >= 3:
                    confluence_factors.append(f"Strong {sequential_info['pattern']} ({sequential_info['count']} wicks)")
                    confidence += 10
                    manipulation_score += 2
                    # Upgrade wick type
                    if wick_type == WickType.EUPHORIA_TOP:
                        wick_type = WickType.DISTRIBUTION_WICK
                    else:
                        wick_type = WickType.ACCUMULATION_WICK
                else:
                    confluence_factors.append(f"{sequential_info['pattern']} forming ({sequential_info['count']} wicks)")
                    confidence += 5
                    manipulation_score += 1
            
            # Factor 6: HTF bias alignment
            if htf_bias:
                if (wick_type in [WickType.EUPHORIA_TOP, WickType.DISTRIBUTION_WICK] and htf_bias.lower() == 'bearish') or \
                   (wick_type in [WickType.CAPITULATION_BOTTOM, WickType.ACCUMULATION_WICK] and htf_bias.lower() == 'bullish'):
                    confluence_factors.append(f"Aligned with HTF {htf_bias} bias")
                    confidence += 6
                    manipulation_score += 1
                elif (wick_type in [WickType.EUPHORIA_TOP, WickType.DISTRIBUTION_WICK] and htf_bias.lower() == 'bullish') or \
                     (wick_type in [WickType.CAPITULATION_BOTTOM, WickType.ACCUMULATION_WICK] and htf_bias.lower() == 'bearish'):
                    confluence_factors.append(f"Counter-trend to HTF {htf_bias} bias (higher risk)")
                    confidence -= 8
            
            # Factor 7: Order block confluence
            if order_blocks:
                ob_confluence = self._check_order_block_confluence(
                    best_wick['rejection_price'], 
                    order_blocks, 
                    wick_type
                )
                if ob_confluence['found']:
                    confluence_factors.append(f"Wick into {ob_confluence['ob_type']} OB at ${ob_confluence['ob_price']:.2f}")
                    confidence += 8
                    manipulation_score += 2
            
            # Factor 8: Stop hunt confluence
            if stop_hunt_result and stop_hunt_result.get('detected'):
                if stop_hunt_result.get('type') == 'stop_hunt':
                    confluence_factors.append("Confirmed liquidity sweep (stop hunt)")
                    confidence += 10
                    manipulation_score += 2.5
            
            # Factor 9: Fear & Greed alignment (psychology)
            if fear_greed:
                fg_value = fear_greed.get('value', 50)
                if wick_type in [WickType.CAPITULATION_BOTTOM, WickType.ACCUMULATION_WICK] and fg_value <= 25:
                    confluence_factors.append(f"Extreme fear ({fg_value}) + capitulation wick")
                    confidence += 8
                    manipulation_score += 1.5
                elif wick_type in [WickType.EUPHORIA_TOP, WickType.DISTRIBUTION_WICK] and fg_value >= 75:
                    confluence_factors.append(f"Extreme greed ({fg_value}) + euphoria wick")
                    confidence += 8
                    manipulation_score += 1.5
            
            # Factor 10: Structure break alignment
            if structure_break:
                struct_dir = structure_break.get('direction', '').lower()
                struct_type = structure_break.get('type', '')
                if struct_type == 'CHoCH':
                    if (direction == 'Long' and struct_dir == 'bullish') or \
                       (direction == 'Short' and struct_dir == 'bearish'):
                        confluence_factors.append(f"CHoCH alignment ({struct_dir})")
                        confidence += 10
                        manipulation_score += 2
            
            # Determine strength classification
            strength = self._classify_strength(
                best_wick['wick_ratio'],
                volume_ratio,
                len(confluence_factors)
            )
            
            # Cap confidence
            confidence = min(confidence, self.config.max_confidence)
            
            # Cap manipulation score
            manipulation_score = min(manipulation_score, 10.0)
            
            # Build reason string
            reason = self._build_reason_string(
                wick_type, 
                direction, 
                best_wick, 
                confluence_factors,
                symbol
            )
            
            # Populate result
            result.detected = True
            result.wick_type = wick_type
            result.direction = direction
            result.strength = strength
            result.confidence = confidence
            result.wick_ratio = best_wick['wick_ratio']
            result.volume_ratio = volume_ratio
            result.rejection_price = best_wick['rejection_price']
            result.close_position = best_wick['close_position']
            result.trend_context = trend_info['description']
            result.confluence_factors = confluence_factors
            result.manipulation_score = manipulation_score
            result.reason = reason
            result.candle_data = {
                'open': best_wick['open'],
                'high': best_wick['high'],
                'low': best_wick['low'],
                'close': best_wick['close'],
                'volume': best_wick['volume']
            }
            
            # Log detection
            logger.info(
                f"[{symbol}] 🕯️ WICK DETECTED: {wick_type.value} | "
                f"Direction: {direction} | Confidence: {confidence:.1f}% | "
                f"Wick Ratio: {best_wick['wick_ratio']:.2f}x | "
                f"Volume: {volume_ratio:.1f}x | "
                f"Manipulation Score: {manipulation_score:.1f}/10"
            )
            logger.info(f"[{symbol}] Confluence factors: {', '.join(confluence_factors)}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{symbol}] Error in wick detection: {e}", exc_info=True)
            return result
    
    def _analyze_single_candle(
        self, 
        candle: pd.Series, 
        df: pd.DataFrame,
        candle_index: int
    ) -> Dict[str, Any]:
        """
        Analyze a single candle for wick characteristics.
        
        Args:
            candle: Single candle data (OHLCV)
            df: Full DataFrame for context
            candle_index: Index of this candle in recent data
            
        Returns:
            Dictionary with wick analysis results
        """
        o, h, l, c, v = candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
        
        candle_range = h - l
        if candle_range == 0:
            return {'has_significant_wick': False}
        
        body = abs(c - o)
        body_percent = body / candle_range
        
        # Filter out dojis (very small body)
        if body_percent < self.config.min_body_percent:
            body = candle_range * self.config.min_body_percent  # Use minimum body for ratio calc
        
        # Calculate wicks
        if c >= o:  # Bullish candle
            upper_wick = h - c
            lower_wick = o - l
        else:  # Bearish candle
            upper_wick = h - o
            lower_wick = c - l
        
        # Avoid division by zero
        body = max(body, candle_range * 0.01)
        
        upper_ratio = upper_wick / body
        lower_ratio = lower_wick / body
        
        # Determine dominant wick
        if upper_ratio >= self.config.min_wick_body_ratio and upper_ratio > lower_ratio:
            wick_direction = 'upper'
            wick_ratio = upper_ratio
            rejection_price = h
            close_position = (c - l) / candle_range if candle_range > 0 else 0.5
            has_significant = self._is_valid_close_position(close_position, WickType.EUPHORIA_TOP)
        elif lower_ratio >= self.config.min_wick_body_ratio and lower_ratio > upper_ratio:
            wick_direction = 'lower'
            wick_ratio = lower_ratio
            rejection_price = l
            close_position = (c - l) / candle_range if candle_range > 0 else 0.5
            has_significant = self._is_valid_close_position(close_position, WickType.CAPITULATION_BOTTOM)
        else:
            return {'has_significant_wick': False}
        
        # Calculate wick score (for ranking multiple wicks)
        avg_vol = df['volume'].mean() if df['volume'].mean() > 0 else 1
        wick_score = wick_ratio * (1 + (v / avg_vol))
        
        return {
            'has_significant_wick': has_significant,
            'wick_direction': wick_direction,
            'wick_ratio': wick_ratio,
            'rejection_price': rejection_price,
            'close_position': close_position,
            'wick_score': wick_score,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'body': body,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v,
            'candle_index': candle_index
        }
    
    def _is_valid_close_position(self, close_position: float, wick_type: WickType) -> bool:
        """Check if close position is valid for the wick type."""
        if wick_type in [WickType.EUPHORIA_TOP, WickType.DISTRIBUTION_WICK]:
            return close_position <= self.config.euphoria_close_max
        elif wick_type in [WickType.CAPITULATION_BOTTOM, WickType.ACCUMULATION_WICK]:
            return close_position >= self.config.capitulation_close_min
        return False
    
    def _analyze_trend_context(
        self, 
        df: pd.DataFrame, 
        wick_type: WickType
    ) -> Dict[str, Any]:
        """
        Analyze the trend leading up to the wick.
        
        For euphoria tops, we want a preceding uptrend.
        For capitulation bottoms, we want a preceding downtrend.
        """
        lookback = self.config.trend_lookback
        if len(df) < lookback + 5:
            return {'is_exhaustion_context': False, 'trend_strength': 0, 'description': 'Insufficient data'}
        
        # Get trend segment (excluding most recent candles)
        trend_df = df.iloc[-(lookback + 3):-3]
        
        # Calculate price change
        start_price = trend_df['close'].iloc[0]
        end_price = trend_df['close'].iloc[-1]
        price_change_pct = (end_price - start_price) / start_price * 100
        
        # Calculate trend strength using linear regression slope
        closes = trend_df['close'].values
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        
        # Normalize slope
        avg_price = np.mean(closes)
        normalized_slope = (slope / avg_price) * 100  # Percentage change per candle
        
        # Determine trend direction and strength
        trend_strength = min(abs(normalized_slope) / 0.5, 1.0)  # Cap at 1.0
        
        if normalized_slope > 0.1:
            trend_direction = 'uptrend'
            is_exhaustion = wick_type in [WickType.EUPHORIA_TOP, WickType.DISTRIBUTION_WICK]
        elif normalized_slope < -0.1:
            trend_direction = 'downtrend'
            is_exhaustion = wick_type in [WickType.CAPITULATION_BOTTOM, WickType.ACCUMULATION_WICK]
        else:
            trend_direction = 'sideways'
            is_exhaustion = False
        
        # Check for extended move (overextension)
        is_extended = abs(price_change_pct) > 10  # >10% move
        
        description = f"{trend_direction.capitalize()} ({price_change_pct:+.1f}%)"
        if is_extended:
            description += " - Extended"
        
        return {
            'is_exhaustion_context': is_exhaustion and trend_strength > self.config.min_trend_strength,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'price_change_pct': price_change_pct,
            'is_extended': is_extended,
            'description': description
        }
    
    def _detect_sequential_wicks(
        self, 
        recent_df: pd.DataFrame, 
        primary_direction: str
    ) -> Dict[str, Any]:
        """
        Detect sequential wicks indicating distribution/accumulation.
        
        Multiple rejection wicks in the same direction = smart money activity.
        """
        count = 0
        
        for i in range(len(recent_df)):
            candle = recent_df.iloc[i]
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            
            candle_range = h - l
            if candle_range == 0:
                continue
            
            body = abs(c - o)
            body = max(body, candle_range * 0.01)
            
            if c >= o:
                upper_wick = h - c
                lower_wick = o - l
            else:
                upper_wick = h - o
                lower_wick = c - l
            
            if primary_direction == 'upper':
                if upper_wick / body >= self.config.sequential_wick_ratio:
                    count += 1
            else:
                if lower_wick / body >= self.config.sequential_wick_ratio:
                    count += 1
        
        pattern = "Distribution" if primary_direction == 'upper' else "Accumulation"
        
        return {
            'detected': count >= 2,
            'count': count,
            'pattern': pattern
        }
    
    def _calculate_volume_ratio(self, candle_volume: float, df: pd.DataFrame) -> float:
        """Calculate volume ratio compared to recent average."""
        lookback = 20
        if len(df) < lookback:
            lookback = len(df)
        
        avg_volume = df['volume'].tail(lookback).mean()
        if avg_volume == 0:
            return 1.0
        
        return candle_volume / avg_volume
    
    def _check_order_block_confluence(
        self, 
        rejection_price: float, 
        order_blocks: List[Dict],
        wick_type: WickType
    ) -> Dict[str, Any]:
        """Check if wick rejection occurred at an order block level."""
        if not order_blocks:
            return {'found': False}
        
        for ob in order_blocks:
            ob_high = ob.get('high', ob.get('zone_high', ob.get('top', 0)))
            ob_low = ob.get('low', ob.get('zone_low', ob.get('bottom', 0)))
            ob_type = ob.get('type', ob.get('direction', 'unknown'))
            
            # Check if rejection price is within or near the OB
            tolerance = (ob_high - ob_low) * 0.2  # 20% tolerance
            
            is_at_ob = ob_low - tolerance <= rejection_price <= ob_high + tolerance
            
            if is_at_ob:
                # Verify OB type matches wick direction
                ob_type_lower = str(ob_type).lower()
                if (wick_type in [WickType.EUPHORIA_TOP, WickType.DISTRIBUTION_WICK] and 
                    ob_type_lower in ['bearish', 'supply', 'resistance', 'short']):
                    return {
                        'found': True,
                        'ob_type': 'bearish',
                        'ob_price': (ob_high + ob_low) / 2
                    }
                elif (wick_type in [WickType.CAPITULATION_BOTTOM, WickType.ACCUMULATION_WICK] and 
                      ob_type_lower in ['bullish', 'demand', 'support', 'long']):
                    return {
                        'found': True,
                        'ob_type': 'bullish',
                        'ob_price': (ob_high + ob_low) / 2
                    }
        
        return {'found': False}
    
    def _classify_strength(
        self, 
        wick_ratio: float, 
        volume_ratio: float,
        confluence_count: int
    ) -> WickStrength:
        """Classify the overall strength of the wick signal."""
        score = 0
        
        if wick_ratio >= self.config.extreme_wick_ratio:
            score += 3
        elif wick_ratio >= self.config.strong_wick_ratio:
            score += 2
        else:
            score += 1
        
        if volume_ratio >= self.config.extreme_volume_ratio:
            score += 3
        elif volume_ratio >= self.config.strong_volume_ratio:
            score += 2
        elif volume_ratio >= self.config.min_volume_ratio:
            score += 1
        
        if confluence_count >= 6:
            score += 2
        elif confluence_count >= 4:
            score += 1
        
        if score >= 7:
            return WickStrength.EXTREME
        elif score >= 5:
            return WickStrength.STRONG
        elif score >= 3:
            return WickStrength.MODERATE
        return WickStrength.WEAK
    
    def _build_reason_string(
        self,
        wick_type: WickType,
        direction: str,
        wick_data: Dict,
        confluence_factors: List[str],
        symbol: str
    ) -> str:
        """Build a human-readable reason string."""
        type_names = {
            WickType.EUPHORIA_TOP: "Euphoria Top",
            WickType.CAPITULATION_BOTTOM: "Capitulation Bottom",
            WickType.DISTRIBUTION_WICK: "Distribution Pattern",
            WickType.ACCUMULATION_WICK: "Accumulation Pattern"
        }
        
        type_name = type_names.get(wick_type, "Wick Pattern")
        
        reason = f"{type_name} detected on {symbol}: "
        reason += f"Strong rejection wick ({wick_data['wick_ratio']:.1f}x body) "
        reason += f"at ${wick_data['rejection_price']:.2f}. "
        reason += f"Confluence: {', '.join(confluence_factors[:3])}"
        
        if len(confluence_factors) > 3:
            reason += f" (+{len(confluence_factors)-3} more factors)"
        
        return reason


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def detect_euphoria_capitulation(
    df: pd.DataFrame,
    symbol: str = "",
    config: Optional[WickDetectorConfig] = None,
    **kwargs
) -> WickDetectionResult:
    """
    Convenience function to detect euphoria/capitulation wicks.
    
    Args:
        df: OHLCV DataFrame
        symbol: Trading pair symbol
        config: Optional configuration
        **kwargs: Additional arguments passed to detect_wick_reversal
        
    Returns:
        WickDetectionResult
    """
    detector = WickDetector(config)
    return detector.detect_wick_reversal(df, symbol, **kwargs)


def get_wick_confluence(result: WickDetectionResult) -> Dict[str, Any]:
    """
    Extract confluence information for integration with other modules.
    
    Args:
        result: WickDetectionResult from detection
        
    Returns:
        Dictionary with confluence data for signal generation
    """
    if not result.detected:
        return {
            'has_wick_signal': False,
            'confluence_boost': 0,
            'context_string': ""
        }
    
    # Calculate confluence boost (0-3 scale to match other modules)
    boost = 0
    if result.confidence >= 85:
        boost = 3
    elif result.confidence >= 75:
        boost = 2
    elif result.confidence >= 65:
        boost = 1
    
    # Build context string for Claude prompt
    context_parts = [
        f"- **{result.wick_type.value.replace('_', ' ').title()}** detected: ",
        f"Strong price rejection at ${result.rejection_price:.2f} ",
        f"(wick {result.wick_ratio:.1f}x body, {result.volume_ratio:.1f}x volume). ",
        f"Direction: {result.direction} with {result.confidence:.0f}% confidence. ",
        f"Manipulation score: {result.manipulation_score:.1f}/10."
    ]
    
    if result.trend_context:
        context_parts.append(f" Trend context: {result.trend_context}.")
    
    context_string = "".join(context_parts)
    
    return {
        'has_wick_signal': True,
        'confluence_boost': boost,
        'direction': result.direction,
        'confidence': result.confidence,
        'manipulation_score': result.manipulation_score,
        'context_string': context_string,
        'wick_type': result.wick_type.value,
        'rejection_price': result.rejection_price,
        'factors': result.confluence_factors
    }


def format_wick_for_telegram(result: WickDetectionResult) -> str:
    """
    Format wick detection result for Telegram message.
    
    Args:
        result: WickDetectionResult from detection
        
    Returns:
        Formatted string for Telegram
    """
    if not result.detected:
        return ""
    
    wick_emoji = "📈" if result.direction == "Long" else "📉"
    strength_emoji = {
        WickStrength.EXTREME: "🔥🔥🔥",
        WickStrength.STRONG: "🔥🔥",
        WickStrength.MODERATE: "🔥",
        WickStrength.WEAK: "⚡"
    }.get(result.strength, "⚡")
    
    type_display = result.wick_type.value.replace('_', ' ').title()
    
    msg = f"\n{wick_emoji} **{type_display}** {strength_emoji}\n"
    msg += f"   Rejection at: ${result.rejection_price:.2f}\n"
    msg += f"   Wick Strength: {result.strength.value.title()}\n"
    msg += f"   Volume: {result.volume_ratio:.1f}x average\n"
    msg += f"   Confidence: {result.confidence:.0f}%\n"
    
    if result.manipulation_score >= 7:
        msg += "   ⚡ High manipulation score - Institutional activity likely\n"
    
    if result.trend_context:
        msg += f"   Trend: {result.trend_context}\n"
    
    return msg


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Create test data with a clear euphoria top pattern
    np.random.seed(42)
    
    # Simulate uptrend then euphoria top (long upper wick)
    prices = [100]
    volumes = [1000]
    
    for i in range(49):
        # Uptrend
        change = np.random.uniform(0.5, 2.0)
        prices.append(prices[-1] + change)
        volumes.append(np.random.uniform(800, 1200))
    
    # Add euphoria top candle (long upper wick)
    last_close = prices[-1]
    test_df = pd.DataFrame({
        'open': prices[:-1] + [last_close],
        'high': [p + np.random.uniform(0.5, 1.5) for p in prices[:-1]] + [last_close + 8],  # Long upper wick
        'low': [p - np.random.uniform(0.2, 0.5) for p in prices[:-1]] + [last_close - 0.5],
        'close': prices[:-1] + [last_close + 0.3],  # Close near open (in lower part)
        'volume': volumes[:-1] + [volumes[-1] * 2.5]  # Volume surge
    })
    
    # Test detection
    detector = WickDetector()
    result = detector.detect_wick_reversal(test_df, symbol="TEST/USDT")
    
    if result.detected:
        print(f"✅ Wick detected: {result.wick_type.value}")
        print(f"   Direction: {result.direction}")
        print(f"   Confidence: {result.confidence:.1f}%")
        print(f"   Wick Ratio: {result.wick_ratio:.2f}x")
        print(f"   Volume Ratio: {result.volume_ratio:.2f}x")
        print(f"   Manipulation Score: {result.manipulation_score:.1f}/10")
        print(f"   Factors: {', '.join(result.confluence_factors)}")
    else:
        print("❌ No wick pattern detected")
