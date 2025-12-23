# signal_evaluator.py - Grok Elite Signal Bot v27.10.0 - Unified Signal Grading
"""
UNIFIED SIGNAL EVALUATION SYSTEM

Replaces the scattered scoring logic with a single, weighted system:
- 40% Confluence (count × quality weight)
- 30% OB Score (strength, freshness, distance)
- 20% Psychology/Volume alignment
- 10% Structure (BOS/CHoCH, HTF bias)

v27.10.0: New unified SignalEvaluator class
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ============================================================================
# CONFIGURABLE WEIGHTS (can be tuned via config.py or backtesting)
# ============================================================================

DEFAULT_WEIGHTS = {
    'confluence': 0.40,      # 40% - Core factor
    'ob_score': 0.30,        # 30% - Order block quality
    'psychology': 0.20,      # 20% - Market psychology + volume
    'structure': 0.10,       # 10% - BOS/CHoCH, HTF alignment
}

# Grade thresholds (configurable)
GRADE_THRESHOLDS = {
    'A': 85,
    'B': 70,
    'C': 55,
    'D': 40,
    'F': 0
}

# Position size multipliers by grade
SIZE_MULTIPLIERS = {
    'A': 1.0,
    'B': 0.85,
    'C': 0.65,
    'D': 0.0,
    'F': 0.0
}

# Executable grades
EXECUTABLE_GRADES = ['A', 'B', 'C']


# ============================================================================
# DATA CLASSES FOR CLEAN INPUT
# ============================================================================

@dataclass
class ConfluenceData:
    """Confluence factors for a trade."""
    factors: List[str] = field(default_factory=list)
    count: int = 0
    has_ob: bool = False
    has_fvg: bool = False
    has_liq_sweep: bool = False
    has_pd_zone: bool = False
    has_htf: bool = False
    has_volume: bool = False
    has_momentum: bool = False
    has_divergence: bool = False
    has_funding: bool = False
    has_oi: bool = False
    has_mtf: bool = False
    has_ote: bool = False
    has_structure: bool = False
    has_psychology: bool = False


@dataclass
class OBData:
    """Order block data."""
    strength: float = 0.0
    freshness: float = 0.0      # 0-1, higher = more recent
    distance_pct: float = 100.0  # Distance from current price
    mitigation: float = 0.0     # 0-1, lower = cleaner
    volume_ratio: float = 1.0   # Volume at OB formation vs average
    htf_aligned: bool = False


@dataclass
class PsychologyData:
    """Market psychology data."""
    fear_greed: Optional[int] = None
    fear_greed_label: str = ""
    long_short_ratio: Optional[float] = None
    funding_rate: Optional[float] = None
    volume_comparison: Optional[Dict] = None
    sentiment_signal: str = "neutral"  # 'long', 'short', 'neutral'


@dataclass
class StructureData:
    """Market structure data."""
    bos_detected: bool = False
    choch_detected: bool = False
    structure_type: str = ""     # 'BOS', 'CHoCH', ''
    structure_direction: str = ""  # 'bullish', 'bearish', ''
    htf_trend: str = ""          # 'Uptrend', 'Downtrend', 'Sideways'
    trend_aligned: bool = False
    is_counter_trend: bool = False


@dataclass
class TradeParams:
    """Trade parameters for R:R calculation."""
    direction: str = "Long"
    entry_low: float = 0.0
    entry_high: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    current_price: float = 0.0
    claude_confidence: int = 65


# ============================================================================
# UNIFIED SIGNAL EVALUATOR
# ============================================================================

class SignalEvaluator:
    """
    Unified signal evaluation system.
    
    Usage:
        evaluator = SignalEvaluator(weights=custom_weights)
        result = evaluator.evaluate(
            confluence=confluence_data,
            ob=ob_data,
            psychology=psych_data,
            structure=structure_data,
            trade=trade_params
        )
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize with optional custom weights."""
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def evaluate(
        self,
        confluence: ConfluenceData,
        ob: OBData,
        psychology: PsychologyData,
        structure: StructureData,
        trade: TradeParams
    ) -> Dict[str, Any]:
        """
        Evaluate a trade signal and return comprehensive grading.
        
        Returns:
            {
                'grade': 'A'/'B'/'C'/'D'/'F',
                'score': 0-100,
                'executable': bool,
                'size_mult': 0.0-1.0,
                'component_scores': {confluence, ob, psychology, structure},
                'weighted_scores': {confluence, ob, psychology, structure},
                'reasons': [str],
                'warnings': [str],
                'rr_ratio': float,
                'entry_distance_pct': float
            }
        """
        reasons = []
        warnings = []
        
        # Calculate component scores (each 0-100)
        confluence_score, conf_reasons = self._score_confluence(confluence)
        ob_score, ob_reasons = self._score_ob(ob, trade)
        psych_score, psych_reasons = self._score_psychology(psychology, trade)
        structure_score, struct_reasons = self._score_structure(structure, trade)
        
        reasons.extend(conf_reasons)
        reasons.extend(ob_reasons)
        reasons.extend(psych_reasons)
        reasons.extend(struct_reasons)
        
        # Apply weights
        weighted_scores = {
            'confluence': confluence_score * self.weights['confluence'],
            'ob_score': ob_score * self.weights['ob_score'],
            'psychology': psych_score * self.weights['psychology'],
            'structure': structure_score * self.weights['structure'],
        }
        
        # Total score
        total_score = sum(weighted_scores.values())
        total_score = min(100, max(0, total_score))
        
        # Calculate R:R
        entry_mid = (trade.entry_low + trade.entry_high) / 2
        sl_dist = abs(trade.sl - entry_mid)
        rr_ratio = abs(trade.tp1 - entry_mid) / sl_dist if sl_dist > 0 else 0
        
        # Entry distance
        entry_dist_pct = abs(entry_mid - trade.current_price) / trade.current_price * 100 if trade.current_price > 0 else 100
        
        # Apply bonuses/penalties
        total_score = self._apply_bonuses(
            total_score, trade, rr_ratio, entry_dist_pct,
            confluence, structure, reasons, warnings
        )
        
        # Determine grade
        grade = self._score_to_grade(total_score)
        executable = grade in EXECUTABLE_GRADES
        size_mult = SIZE_MULTIPLIERS.get(grade, 0.0)
        
        # Final validation checks
        if not executable:
            warnings.append(f"Grade {grade} not executable")
        
        if rr_ratio < 1.5:
            warnings.append(f"Low R:R ({rr_ratio:.1f})")
        
        if entry_dist_pct > 5.0:
            warnings.append(f"Entry far ({entry_dist_pct:.1f}%)")
        
        logging.info(
            f"SignalEvaluator: Grade {grade} ({total_score:.0f}/100) | "
            f"Conf={confluence_score:.0f} OB={ob_score:.0f} Psych={psych_score:.0f} Struct={structure_score:.0f}"
        )
        
        return {
            'grade': grade,
            'score': int(total_score),
            'executable': executable,
            'size_mult': size_mult,
            'component_scores': {
                'confluence': confluence_score,
                'ob_score': ob_score,
                'psychology': psych_score,
                'structure': structure_score
            },
            'weighted_scores': weighted_scores,
            'reasons': reasons[:5],  # Top 5 reasons
            'warnings': warnings,
            'rr_ratio': rr_ratio,
            'entry_distance_pct': entry_dist_pct
        }
    
    def _score_confluence(self, conf: ConfluenceData) -> Tuple[float, List[str]]:
        """Score confluence factors (0-100)."""
        reasons = []
        score = 0
        
        # Base score from factor count
        # 3 factors = 50, 4 = 65, 5 = 80, 6+ = 90+
        if conf.count >= 6:
            score = 95
            reasons.append(f"Excellent confluence ({conf.count})")
        elif conf.count >= 5:
            score = 85
            reasons.append(f"Strong confluence ({conf.count})")
        elif conf.count >= 4:
            score = 70
            reasons.append(f"Good confluence ({conf.count})")
        elif conf.count >= 3:
            score = 55
            reasons.append(f"Minimum confluence ({conf.count})")
        else:
            score = 30
            reasons.append(f"Weak confluence ({conf.count})")
        
        # Quality bonuses for key factors
        if conf.has_ob:
            score += 3
        if conf.has_fvg:
            score += 2
        if conf.has_liq_sweep:
            score += 4
            reasons.append("+LiqSweep")
        if conf.has_ote:
            score += 3
            reasons.append("+OTE")
        if conf.has_divergence:
            score += 2
        
        return min(100, score), reasons
    
    def _score_ob(self, ob: OBData, trade: TradeParams) -> Tuple[float, List[str]]:
        """Score order block quality (0-100)."""
        reasons = []
        score = 0
        
        # Strength (0-35 pts): 2.0-3.5 mapped
        if ob.strength >= 3.5:
            score += 35
            reasons.append(f"Elite OB ({ob.strength:.1f})")
        elif ob.strength >= 3.0:
            score += 30
            reasons.append(f"Strong OB ({ob.strength:.1f})")
        elif ob.strength >= 2.5:
            score += 25
        elif ob.strength >= 2.0:
            score += 20
        elif ob.strength >= 1.5:
            score += 12
        else:
            score += 5
        
        # Freshness (0-20 pts)
        score += int(ob.freshness * 20)
        
        # Distance (0-25 pts)
        if ob.distance_pct < 1.0:
            score += 25
            reasons.append("Close entry")
        elif ob.distance_pct < 2.0:
            score += 20
        elif ob.distance_pct < 3.0:
            score += 15
        elif ob.distance_pct < 5.0:
            score += 10
        else:
            score += 3
        
        # Mitigation penalty (0-10 pts)
        mitigation_penalty = ob.mitigation * 10
        score -= mitigation_penalty
        
        # Volume bonus (0-10 pts)
        if ob.volume_ratio >= 2.0:
            score += 10
        elif ob.volume_ratio >= 1.5:
            score += 6
        elif ob.volume_ratio >= 1.2:
            score += 3
        
        # HTF alignment bonus
        if ob.htf_aligned:
            score += 5
        
        return min(100, max(0, score)), reasons
    
    def _score_psychology(self, psych: PsychologyData, trade: TradeParams) -> Tuple[float, List[str]]:
        """Score market psychology alignment (0-100)."""
        reasons = []
        score = 50  # Neutral baseline
        
        direction_lower = trade.direction.lower()
        
        # Fear & Greed (0-30 pts swing)
        if psych.fear_greed is not None:
            fg = psych.fear_greed
            
            if fg <= 20:  # Extreme fear
                if direction_lower == 'long':
                    score += 25
                    reasons.append(f"+ExtremeFear({fg})")
                else:
                    score -= 15
            elif fg <= 35:  # Fear
                if direction_lower == 'long':
                    score += 12
            elif fg >= 80:  # Extreme greed
                if direction_lower == 'short':
                    score += 25
                    reasons.append(f"+ExtremeGreed({fg})")
                else:
                    score -= 15
            elif fg >= 65:  # Greed
                if direction_lower == 'short':
                    score += 12
        
        # Long/Short ratio (0-20 pts swing)
        if psych.long_short_ratio is not None:
            ls = psych.long_short_ratio
            
            if ls >= 0.70:  # Crowded longs
                if direction_lower == 'short':
                    score += 15
                    reasons.append(f"+CrowdedLong({ls:.0%})")
                else:
                    score -= 10
            elif ls <= 0.30:  # Crowded shorts
                if direction_lower == 'long':
                    score += 15
                    reasons.append(f"+CrowdedShort({1-ls:.0%})")
                else:
                    score -= 10
        
        # Funding rate (0-10 pts swing)
        if psych.funding_rate is not None:
            fr = psych.funding_rate
            
            if fr > 0.05:  # High positive = longs overleveraged
                if direction_lower == 'short':
                    score += 8
                else:
                    score -= 5
            elif fr < -0.05:  # High negative = shorts overleveraged
                if direction_lower == 'long':
                    score += 8
                else:
                    score -= 5
        
        # Volume comparison
        if psych.volume_comparison:
            dominant = psych.volume_comparison.get('dominant', 'balanced')
            if dominant == 'bybit' and direction_lower == 'short':
                score += 5  # Futures-driven move may reverse
            elif dominant == 'binance' and direction_lower == 'long':
                score += 5  # Spot accumulation
        
        return min(100, max(0, score)), reasons
    
    def _score_structure(self, struct: StructureData, trade: TradeParams) -> Tuple[float, List[str]]:
        """Score market structure alignment (0-100)."""
        reasons = []
        score = 50  # Neutral baseline
        
        direction_lower = trade.direction.lower()
        
        # HTF trend alignment (0-30 pts)
        if struct.trend_aligned:
            score += 30
            reasons.append("+TrendAlign")
        elif struct.is_counter_trend:
            score += 10  # Counter-trend gets partial credit (bounces are valid!)
            reasons.append("Counter-trend")
        
        # BOS/CHoCH detection (0-20 pts)
        if struct.choch_detected:
            if (struct.structure_direction == 'bullish' and direction_lower == 'long') or \
               (struct.structure_direction == 'bearish' and direction_lower == 'short'):
                score += 20
                reasons.append("+CHoCH")
            else:
                score -= 10
        elif struct.bos_detected:
            if (struct.structure_direction == 'bullish' and direction_lower == 'long') or \
               (struct.structure_direction == 'bearish' and direction_lower == 'short'):
                score += 12
                reasons.append("+BOS")
            else:
                score -= 5
        
        return min(100, max(0, score)), reasons
    
    def _apply_bonuses(
        self,
        score: float,
        trade: TradeParams,
        rr_ratio: float,
        entry_dist_pct: float,
        confluence: ConfluenceData,
        structure: StructureData,
        reasons: List[str],
        warnings: List[str]
    ) -> float:
        """Apply final bonuses and penalties."""
        
        # R:R bonus (0-8 pts)
        if rr_ratio >= 3.0:
            score += 8
            reasons.append(f"R:R {rr_ratio:.1f}")
        elif rr_ratio >= 2.5:
            score += 6
        elif rr_ratio >= 2.0:
            score += 4
        elif rr_ratio < 1.5:
            score -= 5
            warnings.append("R:R below 1.5")
        
        # Entry distance penalty
        if entry_dist_pct > 5.0:
            score -= 8
        elif entry_dist_pct > 3.0:
            score -= 4
        
        # Claude confidence boost
        if trade.claude_confidence >= 80:
            score += 5
        elif trade.claude_confidence >= 70:
            score += 3
        elif trade.claude_confidence < 60:
            score -= 3
        
        # Counter-trend penalty (already handled in structure but add warning)
        if structure.is_counter_trend:
            warnings.append("Counter-trend: TP1 only recommended")
        
        return score
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= GRADE_THRESHOLDS['A']:
            return 'A'
        elif score >= GRADE_THRESHOLDS['B']:
            return 'B'
        elif score >= GRADE_THRESHOLDS['C']:
            return 'C'
        elif score >= GRADE_THRESHOLDS['D']:
            return 'D'
        else:
            return 'F'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_confluence_data(
    trade: Dict,
    momentum_data: Optional[Dict],
    divergence_data: Optional[Dict],
    funding_data: Optional[Dict],
    oi_data: Optional[Dict],
    volume_comparison: Optional[Dict],
    timeframe_agreement: int,
    has_ote: bool,
    structure_aligned: bool = False,
    psych_aligned: bool = False
) -> ConfluenceData:
    """Build ConfluenceData from raw inputs."""
    
    factors = []
    
    reason = trade.get('reason', '').upper()
    
    has_ob = 'OB' in reason or 'ORDER BLOCK' in reason
    has_fvg = 'FVG' in reason or 'FAIR VALUE' in reason
    has_liq_sweep = 'LIQ' in reason or 'SWEEP' in reason
    has_pd_zone = 'DISCOUNT' in reason or 'PREMIUM' in reason
    has_htf = 'HTF' in reason
    has_volume = 'VOLUME' in reason or 'VOL' in reason
    
    if has_ob:
        factors.append('+OB')
    if has_fvg:
        factors.append('+FVG')
    if has_liq_sweep:
        factors.append('+LiqSweep')
    if has_pd_zone:
        factors.append('+PD_Zone')
    if has_htf:
        factors.append('+HTF')
    if has_volume:
        factors.append('+Volume')
    
    # Momentum
    has_momentum = False
    if momentum_data:
        mom_signal = momentum_data.get('signal', '').lower()
        direction = trade.get('direction', '').lower()
        if (direction == 'long' and mom_signal == 'bullish') or \
           (direction == 'short' and mom_signal == 'bearish'):
            has_momentum = True
            factors.append('+Momentum')
    
    # Cross-exchange
    has_divergence = False
    if divergence_data and divergence_data.get('significant'):
        has_divergence = True
        factors.append('+Divergence')
    
    has_funding = False
    if funding_data:
        funding = funding_data.get('combined_rate', funding_data.get('avg_rate_pct', 0))
        direction = trade.get('direction', '').lower()
        if (direction == 'long' and funding < -0.01) or \
           (direction == 'short' and funding > 0.01):
            has_funding = True
            factors.append('+Funding')
    
    # OI
    has_oi = False
    if oi_data and oi_data.get('oi_change_pct', 0) != 0:
        has_oi = True
        factors.append('+OI')
    
    # Volume comparison
    if volume_comparison and volume_comparison.get('bybit_dominant'):
        factors.append('+BybitVol')
    
    # MTF
    has_mtf = False
    if timeframe_agreement >= 2:
        has_mtf = True
        factors.append('+MTF_Full')
    elif timeframe_agreement >= 1:
        has_mtf = True
        factors.append('+MTF')
    
    # OTE
    if has_ote:
        factors.append('+OTE')
    
    # Structure
    has_structure = structure_aligned
    if structure_aligned:
        factors.append('+Structure')
    
    # Psychology
    has_psychology = psych_aligned
    if psych_aligned:
        factors.append('+Psychology')
    
    return ConfluenceData(
        factors=factors,
        count=len(factors),
        has_ob=has_ob,
        has_fvg=has_fvg,
        has_liq_sweep=has_liq_sweep,
        has_pd_zone=has_pd_zone,
        has_htf=has_htf,
        has_volume=has_volume,
        has_momentum=has_momentum,
        has_divergence=has_divergence,
        has_funding=has_funding,
        has_oi=has_oi,
        has_mtf=has_mtf,
        has_ote=has_ote,
        has_structure=has_structure,
        has_psychology=has_psychology
    )


def build_ob_data(
    zone: Dict,
    current_price: float,
    df: Optional[Any] = None
) -> OBData:
    """Build OBData from zone dict."""
    import pandas as pd
    
    ob_mid = (zone.get('zone_low', 0) + zone.get('zone_high', 0)) / 2
    distance_pct = abs(current_price - ob_mid) / current_price * 100 if current_price > 0 else 100
    
    # Calculate freshness (assume index is provided or default to 0.5)
    freshness = 0.5
    if df is not None and 'index' in zone:
        total_bars = len(df)
        if total_bars > 0:
            freshness = zone.get('index', 0) / total_bars
    
    # Volume ratio
    volume_ratio = zone.get('volume_ratio', 1.0)
    if df is not None and 'volume' in df.columns:
        vol_mean = df['volume'].mean()
        idx = zone.get('index', -1)
        if idx >= 0 and idx < len(df) and vol_mean > 0:
            volume_ratio = df['volume'].iloc[idx] / vol_mean
    
    return OBData(
        strength=zone.get('strength', zone.get('ob_strength', 1.5)),
        freshness=freshness,
        distance_pct=distance_pct,
        mitigation=zone.get('mitigation', 0),
        volume_ratio=volume_ratio,
        htf_aligned=zone.get('htf_aligned', False)
    )


def build_psychology_data(
    fear_greed: Optional[Dict],
    long_short: Optional[Dict],
    funding_data: Optional[Dict],
    volume_comparison: Optional[Dict]
) -> PsychologyData:
    """Build PsychologyData from raw inputs."""
    
    fg_value = fear_greed.get('value') if fear_greed else None
    fg_label = fear_greed.get('label', '') if fear_greed else ''
    ls_ratio = long_short.get('long_pct') if long_short else None
    
    funding_rate = None
    if funding_data:
        funding_rate = funding_data.get('avg_rate_pct', funding_data.get('combined_rate'))
    
    # Determine sentiment signal
    signal = 'neutral'
    long_score = 0
    short_score = 0
    
    if fg_value:
        if fg_value <= 20:
            long_score += 3
        elif fg_value <= 35:
            long_score += 1
        elif fg_value >= 80:
            short_score += 3
        elif fg_value >= 65:
            short_score += 1
    
    if ls_ratio:
        if ls_ratio >= 0.70:
            short_score += 2
        elif ls_ratio <= 0.30:
            long_score += 2
    
    if funding_rate:
        if funding_rate > 0.05:
            short_score += 1
        elif funding_rate < -0.05:
            long_score += 1
    
    net = long_score - short_score
    if net >= 2:
        signal = 'long'
    elif net <= -2:
        signal = 'short'
    
    return PsychologyData(
        fear_greed=fg_value,
        fear_greed_label=fg_label,
        long_short_ratio=ls_ratio,
        funding_rate=funding_rate,
        volume_comparison=volume_comparison,
        sentiment_signal=signal
    )


def build_structure_data(
    structure_break: Optional[Dict],
    htf_trend: str,
    trade_direction: str
) -> StructureData:
    """Build StructureData from raw inputs."""
    
    trend_aligned = False
    is_counter_trend = False
    
    dir_lower = trade_direction.lower()
    
    if htf_trend == 'Uptrend':
        trend_aligned = dir_lower == 'long'
        is_counter_trend = dir_lower == 'short'
    elif htf_trend == 'Downtrend':
        trend_aligned = dir_lower == 'short'
        is_counter_trend = dir_lower == 'long'
    
    bos_detected = False
    choch_detected = False
    structure_type = ''
    structure_direction = ''
    
    if structure_break:
        structure_type = structure_break.get('type', '')
        structure_direction = structure_break.get('direction', '')
        bos_detected = structure_type == 'BOS'
        choch_detected = structure_type == 'CHoCH'
    
    return StructureData(
        bos_detected=bos_detected,
        choch_detected=choch_detected,
        structure_type=structure_type,
        structure_direction=structure_direction,
        htf_trend=htf_trend,
        trend_aligned=trend_aligned,
        is_counter_trend=is_counter_trend
    )


def build_trade_params(
    trade: Dict,
    current_price: float
) -> TradeParams:
    """Build TradeParams from trade dict."""
    return TradeParams(
        direction=trade.get('direction', 'Long'),
        entry_low=trade.get('entry_low', 0),
        entry_high=trade.get('entry_high', 0),
        sl=trade.get('sl', 0),
        tp1=trade.get('tp1', 0),
        tp2=trade.get('tp2', 0),
        current_price=current_price,
        claude_confidence=trade.get('confidence', 65)
    )


# ============================================================================
# CONVENIENCE FUNCTION FOR BACKWARDS COMPATIBILITY
# ============================================================================

def evaluate_signal(
    trade: Dict,
    current_price: float,
    zone: Dict,
    momentum_data: Optional[Dict] = None,
    divergence_data: Optional[Dict] = None,
    funding_data: Optional[Dict] = None,
    oi_data: Optional[Dict] = None,
    volume_comparison: Optional[Dict] = None,
    timeframe_agreement: int = 0,
    has_ote: bool = False,
    structure_break: Optional[Dict] = None,
    htf_trend: str = "Sideways",
    fear_greed: Optional[Dict] = None,
    long_short: Optional[Dict] = None,
    df: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a signal with all data.
    
    This is the main entry point for replacing scattered scoring logic.
    """
    
    # Build data classes
    confluence = build_confluence_data(
        trade, momentum_data, divergence_data, funding_data,
        oi_data, volume_comparison, timeframe_agreement, has_ote,
        structure_aligned=structure_break is not None and structure_break.get('signal', '').upper() == trade.get('direction', '').upper(),
        psych_aligned=fear_greed is not None and fear_greed.get('value', 50) <= 25
    )
    
    ob = build_ob_data(zone, current_price, df)
    psychology = build_psychology_data(fear_greed, long_short, funding_data, volume_comparison)
    structure = build_structure_data(structure_break, htf_trend, trade.get('direction', 'Long'))
    trade_params = build_trade_params(trade, current_price)
    
    # Create evaluator and run
    evaluator = SignalEvaluator(weights=weights)
    return evaluator.evaluate(confluence, ob, psychology, structure, trade_params)


# ============================================================================
# GRADE DISPLAY HELPERS
# ============================================================================

def format_grade_display(result: Dict[str, Any]) -> str:
    """Format grade result for Telegram message."""
    emoji = {'A': '🌟', 'B': '✅', 'C': '⚡', 'D': '⚠️', 'F': '❌'}.get(result['grade'], '❓')
    return f"{emoji} Grade {result['grade']} ({result['score']}/100)"


def get_grade_emoji(grade: str) -> str:
    """Get emoji for grade."""
    return {'A': '🌟', 'B': '✅', 'C': '⚡', 'D': '⚠️', 'F': '❌'}.get(grade, '❓')
