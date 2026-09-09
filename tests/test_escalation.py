"""Escalation policy tests — edge cases and safety invariants."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.escalation import EscalationPolicy, EscalationDecision
from hiver_agent.intents import INTENTS


POLICY = EscalationPolicy(auto_threshold=0.62)


class TestEscalationSafetyInvariants:
    """Safety invariants that must ALWAYS hold regardless of policy tuning."""

    def test_hacked_account_always_escalates(self):
        decision = POLICY.decide('my account was hacked', 'apple_id_icloud_account', 0.95, 0.90)
        assert decision.action == 'escalate', 'Hacked account must always escalate'

    def test_overheating_safety_always_escalates(self):
        decision = POLICY.decide('my iPhone is overheating and smoking', 'device_hardware_charging', 0.95, 0.90)
        assert decision.action == 'escalate', 'Safety hazard must always escalate'

    def test_password_issue_always_escalates(self):
        decision = POLICY.decide('my Apple ID password is not working', 'apple_id_icloud_account', 0.95, 0.90)
        assert decision.action == 'escalate', 'Account privacy signal must always escalate'

    def test_scam_report_escalates(self):
        decision = POLICY.decide('I got a scam call about iCloud', 'apple_id_icloud_account', 0.95, 0.90)
        assert decision.action == 'escalate', 'Scam report must always escalate'

    def test_high_risk_intent_escalates(self):
        """Intents with escalate_default=True must escalate even with high confidence."""
        for intent, meta in INTENTS.items():
            if meta['escalate_default']:
                decision = POLICY.decide('some message', intent, 0.99, 0.99)
                assert decision.action == 'escalate', \
                    f'{intent} has escalate_default=True but was auto-handled'


class TestEscalationThresholds:
    def test_low_confidence_escalates(self):
        decision = POLICY.decide('some message', 'battery_power', 0.40, 0.80)
        assert decision.action == 'escalate'
        assert 'confidence' in decision.reason.lower() or 'threshold' in decision.reason.lower()

    def test_weak_retrieval_escalates(self):
        decision = POLICY.decide('some message', 'battery_power', 0.80, 0.10)
        assert decision.action == 'escalate'
        assert 'historical' in decision.reason.lower() or 'grounding' in decision.reason.lower()

    def test_high_confidence_good_retrieval_auto_handles(self):
        """battery_power has escalate_default=False — should auto-handle with good signals."""
        decision = POLICY.decide(
            'my battery is draining after the update',
            'battery_power',
            0.85,
            0.55
        )
        assert decision.action == 'auto_handle', \
            f'Expected auto_handle, got escalate. Reason: {decision.reason}'

    def test_boundary_confidence_just_above_threshold(self):
        decision = POLICY.decide('battery drain', 'battery_power', 0.63, 0.55)
        # battery_power is not escalate_default, no risk flags, good retrieval → auto_handle
        assert decision.action == 'auto_handle'

    def test_boundary_confidence_just_below_threshold(self):
        decision = POLICY.decide('battery drain', 'battery_power', 0.61, 0.55)
        assert decision.action == 'escalate'


class TestEscalationDecisionStructure:
    def test_decision_has_required_fields(self):
        decision = POLICY.decide('my iPhone battery is draining', 'battery_power', 0.80, 0.60)
        assert hasattr(decision, 'action')
        assert hasattr(decision, 'reason')
        assert hasattr(decision, 'rules_triggered')
        assert decision.action in ('auto_handle', 'escalate')
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0
        assert isinstance(decision.rules_triggered, list)

    def test_escalate_has_non_empty_reason(self):
        decision = POLICY.decide('hacked account', 'apple_id_icloud_account', 0.20, 0.10)
        assert decision.action == 'escalate'
        assert len(decision.reason) > 10


class TestCustomThreshold:
    def test_custom_threshold_respected(self):
        strict_policy = EscalationPolicy(auto_threshold=0.90)
        # High confidence (0.80) that would pass default threshold (0.62) fails the strict one
        decision = strict_policy.decide('battery drain', 'battery_power', 0.80, 0.60)
        assert decision.action == 'escalate'

    def test_lenient_threshold(self):
        lenient = EscalationPolicy(auto_threshold=0.30)
        decision = lenient.decide('battery drain', 'battery_power', 0.40, 0.60)
        assert decision.action == 'auto_handle'
