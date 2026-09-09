from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from .model import IntentModel
from .retrieval import HistoricalRetriever
from .escalation import EscalationPolicy
from .generation import draft_reply
from .intents import propose_intent
from .config import settings

class SupportAgent:
    def __init__(self, intent_model=None, retriever=None, policy=None):
        self.intent_model = intent_model
        self.retriever = retriever
        self.policy = policy or EscalationPolicy(settings.auto_threshold)

    def predict(self, text: str):
        if self.intent_model is not None:
            labels, conf, _ = self.intent_model.predict_with_confidence([text])
            intent, confidence = labels[0], float(conf[0])
        else:
            intent, confidence = propose_intent(text)
        evidence = self.retriever.search(text, settings.retrieval_k) if self.retriever is not None else []
        sim = evidence[0]['similarity'] if evidence else 0.0
        decision = self.policy.decide(text, intent, confidence, sim)
        generation = draft_reply(text, intent, evidence, decision.action=='escalate')
        return {
            'brand': settings.brand,
            'customer_message': text,
            'intent': intent,
            'intent_confidence': round(confidence, 4),
            'decision': decision.action,
            'decision_reason': decision.reason,
            'rules_triggered': decision.rules_triggered,
            'reply': generation['reply'],
            'generation_mode': generation['mode'],
            'generation_rationale': generation.get('rationale',''),
            'evidence': evidence,
            'evidence_ids': generation.get('evidence_ids', [e.get('support_tweet_id') for e in evidence]),
            'unsupported_claims': generation.get('unsupported_claims', []),
        }
