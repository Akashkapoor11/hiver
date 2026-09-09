from __future__ import annotations
from dataclasses import dataclass
from .intents import INTENTS, risk_flags

@dataclass
class EscalationDecision:
    action: str
    reason: str
    rules_triggered: list[str]

class EscalationPolicy:
    def __init__(self, auto_threshold: float = 0.62):
        self.auto_threshold = auto_threshold

    def decide(self, text: str, intent: str, intent_confidence: float, retrieval_similarity: float = 0.0) -> EscalationDecision:
        flags = risk_flags(text)
        reasons = []
        if flags:
            flag_str = ', '.join(flags)
            reasons.append(f'sensitive or high-risk signal detected ({flag_str})')
        if INTENTS.get(intent, {}).get('escalate_default'):
            reasons.append(f'intent {intent} often needs account-specific or transactional handling')
        if intent_confidence < self.auto_threshold:
            reasons.append(f'intent confidence {intent_confidence:.2f} below auto-handle threshold {self.auto_threshold:.2f}')
        if retrieval_similarity < 0.28:
            reasons.append('weak historical match for grounding (similarity < 0.28)')
        # Conservative policy: any hard risk signal or low confidence/grounding escalates.
        should_escalate = bool(
            flags
            or intent_confidence < self.auto_threshold
            or retrieval_similarity < 0.28
            or INTENTS.get(intent, {}).get('escalate_default')
        )
        if should_escalate:
            return EscalationDecision(
                'escalate',
                '; '.join(reasons) or 'case is outside safe autonomous scope',
                flags
            )
        return EscalationDecision(
            'auto_handle',
            'common issue with adequate intent confidence and a sufficiently similar historical resolution; no hard risk signal detected',
            flags
        )
