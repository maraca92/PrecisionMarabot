# ai_consensus.py - Grok Elite Signal Bot v27.4.0 - Dual AI Decision Logic
"""
Combines Claude (primary) and Grok (secondary) for optimal trade decisions.

Decision Matrix:
- Both agree (high conf) → Execute with high confidence
- Claude confident, Grok warns → Execute with reduced confidence
- Disagreement → No trade (safety first)
- Either validation fails → Skip

v27.4.0: Dual AI consensus system
"""
import logging
from typing import Dict, Any, Optional, Tuple

from bot.config import (
    AI_CONSENSUS_MIN_CONFIDENCE, AI_CONSENSUS_BOOST_AGREEMENT,
    AI_CONSENSUS_PENALTY_DISAGREEMENT, AI_MIN_CONFIDENCE_SOLO
)

async def get_consensus_decision(
    claude_result: Dict[str, Any],
    grok_result: Dict[str, Any],
    symbol: str,
    current_price: float
) -> Dict[str, Any]:
    
    claude_trade = claude_result.get('trade')
    grok_trade = grok_result.get('trade')
    
    claude_no_trade = claude_result.get('no_trade', False)
    grok_no_trade = grok_result.get('no_trade', False)
    
    if claude_no_trade and grok_no_trade:
        return {
            'no_trade': True,
            'consensus': 'both_skip',
            'reason': f"Claude: {claude_result.get('reason', 'N/A')} | Grok: {grok_result.get('reason', 'N/A')}"
        }
    
    if claude_trade and grok_no_trade:
        claude_conf = claude_trade.get('confidence', 0)
        
        if claude_conf >= AI_MIN_CONFIDENCE_SOLO:
            adjusted_trade = claude_trade.copy()
            adjusted_trade['confidence'] = int(claude_conf * 0.9)
            adjusted_trade['consensus'] = 'claude_solo'
            adjusted_trade['consensus_note'] = f"Grok skeptical: {grok_result.get('reason', 'N/A')}"
            
            logging.info(f"{symbol}: Claude solo trade (conf {adjusted_trade['confidence']}%)")
            
            return {'trade': adjusted_trade}
        else:
            return {
                'no_trade': True,
                'consensus': 'claude_low_confidence',
                'reason': f"Claude {claude_conf}% (need {AI_MIN_CONFIDENCE_SOLO}%+), Grok declined"
            }
    
    if grok_trade and claude_no_trade:
        return {
            'no_trade': True,
            'consensus': 'grok_solo_rejected',
            'reason': f"Claude declined (primary analysis must approve). Claude: {claude_result.get('reason', 'N/A')}"
        }
    
    if claude_trade and grok_trade:
        return analyze_dual_trade_agreement(
            claude_trade, grok_trade, symbol, current_price
        )
    
    return {
        'no_trade': True,
        'consensus': 'invalid_response',
        'reason': 'One or both AIs returned invalid format'
    }

def analyze_dual_trade_agreement(
    claude_trade: Dict,
    grok_trade: Dict,
    symbol: str,
    current_price: float
) -> Dict[str, Any]:
    
    claude_dir = claude_trade.get('direction', '').lower()
    grok_dir = grok_trade.get('direction', '').lower()
    
    if claude_dir != grok_dir:
        return {
            'no_trade': True,
            'consensus': 'directional_conflict',
            'reason': f"Claude suggests {claude_dir}, Grok suggests {grok_dir}"
        }
    
    agreement_score = calculate_agreement_score(claude_trade, grok_trade, current_price)
    
    base_confidence = claude_trade.get('confidence', 70)
    
    if agreement_score > 0.8:
        final_confidence = min(95, int(base_confidence + AI_CONSENSUS_BOOST_AGREEMENT))
        consensus_level = 'strong_agreement'
        
    elif agreement_score > 0.6:
        final_confidence = base_confidence
        consensus_level = 'moderate_agreement'
        
    else:
        final_confidence = max(60, int(base_confidence - AI_CONSENSUS_PENALTY_DISAGREEMENT))
        consensus_level = 'weak_agreement'
    
    if final_confidence < AI_CONSENSUS_MIN_CONFIDENCE:
        return {
            'no_trade': True,
            'consensus': 'low_final_confidence',
            'reason': f"Post-consensus confidence {final_confidence}% < {AI_CONSENSUS_MIN_CONFIDENCE}%"
        }
    
    consensus_trade = claude_trade.copy()
    consensus_trade['confidence'] = final_confidence
    consensus_trade['consensus'] = consensus_level
    consensus_trade['agreement_score'] = agreement_score
    consensus_trade['ai_source'] = 'claude+grok'
    
    consensus_trade['claude_reasoning'] = claude_trade.get('reason', 'N/A')
    consensus_trade['grok_reasoning'] = grok_trade.get('reason', 'N/A')
    
    consensus_trade['reason'] = f"{claude_trade.get('reason', 'Claude OB')} | Grok: {grok_trade.get('reason', 'Confirmed')}"
    
    logging.info(
        f"{symbol}: Consensus {consensus_level} "
        f"(agreement {agreement_score:.2f}, final conf {final_confidence}%)"
    )
    
    return {'trade': consensus_trade}

def calculate_agreement_score(
    claude_trade: Dict,
    grok_trade: Dict,
    current_price: float
) -> float:
    
    score = 0.0
    weights = 0.0
    
    try:
        claude_entry_mid = (claude_trade['entry_low'] + claude_trade['entry_high']) / 2
        grok_entry_mid = (grok_trade['entry_low'] + grok_trade['entry_high']) / 2
        
        entry_diff_pct = abs(claude_entry_mid - grok_entry_mid) / current_price * 100
        
        if entry_diff_pct < 0.5:
            score += 0.3
        elif entry_diff_pct < 1.0:
            score += 0.2
        elif entry_diff_pct < 2.0:
            score += 0.1
        
        weights += 0.3
        
    except (KeyError, ZeroDivisionError):
        pass
    
    try:
        claude_sl = claude_trade['sl']
        grok_sl = grok_trade['sl']
        
        sl_diff_pct = abs(claude_sl - grok_sl) / current_price * 100
        
        if sl_diff_pct < 0.5:
            score += 0.25
        elif sl_diff_pct < 1.0:
            score += 0.15
        elif sl_diff_pct < 2.0:
            score += 0.05
        
        weights += 0.25
        
    except (KeyError, ZeroDivisionError):
        pass
    
    try:
        claude_tp1 = claude_trade['tp1']
        grok_tp1 = grok_trade['tp1']
        
        tp1_diff_pct = abs(claude_tp1 - grok_tp1) / current_price * 100
        
        if tp1_diff_pct < 0.5:
            score += 0.20
        elif tp1_diff_pct < 1.0:
            score += 0.12
        elif tp1_diff_pct < 2.0:
            score += 0.06
        
        weights += 0.20
        
    except (KeyError, ZeroDivisionError):
        pass
    
    try:
        claude_tp2 = claude_trade['tp2']
        grok_tp2 = grok_trade['tp2']
        
        tp2_diff_pct = abs(claude_tp2 - grok_tp2) / current_price * 100
        
        if tp2_diff_pct < 1.0:
            score += 0.15
        elif tp2_diff_pct < 2.0:
            score += 0.09
        elif tp2_diff_pct < 3.0:
            score += 0.03
        
        weights += 0.15
        
    except (KeyError, ZeroDivisionError):
        pass
    
    try:
        claude_lev = claude_trade.get('leverage', 3)
        grok_lev = grok_trade.get('leverage', 3)
        
        lev_diff = abs(claude_lev - grok_lev)
        
        if lev_diff == 0:
            score += 0.10
        elif lev_diff == 1:
            score += 0.05
        
        weights += 0.10
        
    except (KeyError, TypeError):
        pass
    
    if weights > 0:
        return score / weights
    else:
        return 0.5

def get_consensus_summary(result: Dict[str, Any]) -> str:
    if result.get('no_trade'):
        consensus = result.get('consensus', 'unknown')
        reason = result.get('reason', 'N/A')
        
        summaries = {
            'both_skip': '🚫 Both AIs declined',
            'claude_low_confidence': '⚠️ Claude not confident enough',
            'grok_solo_rejected': '⛔ Claude (primary) declined',
            'directional_conflict': '⚡ Direction conflict',
            'low_final_confidence': '📉 Confidence too low post-consensus',
            'invalid_response': '❌ Invalid AI response'
        }
        
        return summaries.get(consensus, f"No trade: {reason}")
    
    if 'trade' in result:
        trade = result['trade']
        consensus = trade.get('consensus', 'unknown')
        agreement = trade.get('agreement_score', 0.5)
        conf = trade.get('confidence', 0)
        
        summaries = {
            'strong_agreement': f'✅ Strong agreement (score {agreement:.2f}, conf {conf}%)',
            'moderate_agreement': f'☑️ Moderate agreement (score {agreement:.2f}, conf {conf}%)',
            'weak_agreement': f'⚠️ Weak agreement (score {agreement:.2f}, conf {conf}%)',
            'claude_solo': f'🎯 Claude solo (Grok skeptical, conf {conf}%)'
        }
        
        return summaries.get(consensus, f"Consensus: {conf}%")
    
    return "Unknown consensus state"
