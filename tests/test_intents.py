"""Comprehensive intent classification tests."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.intents import propose_intent, risk_flags, keyword_scores, INTENTS, normalize


# ─── Basic intent coverage ───────────────────────────────────────────────────

class TestIntentCoverage:
    def test_battery_drain(self):
        intent, _ = propose_intent('my battery is draining really fast after the update')
        assert intent == 'battery_power'

    def test_battery_charge_duration(self):
        intent, _ = propose_intent('battery life is terrible, only lasts 2 hours')
        assert intent == 'battery_power'

    def test_device_hardware_wont_charge(self):
        intent, _ = propose_intent("my iPhone won't charge at all, the port is broken")
        assert intent == 'device_hardware_charging'

    def test_device_hardware_screen(self):
        intent, _ = propose_intent('the screen on my iPhone is cracked and unresponsive')
        assert intent == 'device_hardware_charging'

    def test_ios_update(self):
        intent, _ = propose_intent('how do I update to iOS 17 on my phone')
        assert intent == 'ios_update_software'

    def test_ios_bug_after_update(self):
        intent, _ = propose_intent('ever since the iOS update my phone is very slow')
        assert intent == 'ios_update_software'

    def test_apps_media_appstore(self):
        intent, _ = propose_intent('I cannot download apps from the App Store')
        assert intent == 'apps_media'

    def test_apps_media_music(self):
        intent, _ = propose_intent('Apple Music is not playing my songs')
        assert intent == 'apps_media'

    def test_apple_id_signin(self):
        intent, _ = propose_intent('I cannot sign in to my Apple ID account')
        assert intent == 'apple_id_icloud_account'

    def test_connectivity_wifi(self):
        intent, _ = propose_intent('my WiFi keeps disconnecting on my MacBook')
        assert intent == 'connectivity_network'

    def test_connectivity_bluetooth(self):
        intent, _ = propose_intent('Bluetooth will not connect to my AirPods')
        assert intent == 'connectivity_network'

    def test_purchases_billing(self):
        intent, _ = propose_intent('I was charged twice for my subscription')
        assert intent == 'purchases_billing_subscriptions'

    def test_sync_setup_backup(self):
        intent, _ = propose_intent('my iCloud backup is not working, how do I restore')
        assert intent == 'sync_setup_data'

    def test_features_accessibility(self):
        intent, _ = propose_intent('how do I enable accessibility features')
        assert intent == 'features_accessibility_other'

    def test_unknown_falls_back(self):
        intent, conf = propose_intent('hello')
        assert intent == 'features_accessibility_other'
        assert conf < 0.20


# ─── Confidence properties ───────────────────────────────────────────────────

class TestConfidence:
    def test_confidence_in_range(self):
        for text in [
            'battery drain after iOS update',
            'Apple ID password issue',
            'WiFi disconnecting',
            'hello world',
        ]:
            _, conf = propose_intent(text)
            assert 0.0 <= conf <= 1.0, f'Confidence out of range for: {text}'

    def test_high_confidence_for_strong_signal(self):
        _, conf = propose_intent('my battery is draining and battery life is terrible after iOS update')
        assert conf >= 0.55, f'Expected high confidence for multi-keyword message, got {conf}'

    def test_low_confidence_for_ambiguous(self):
        _, conf = propose_intent('please help')
        assert conf < 0.50

    def test_max_confidence_capped(self):
        _, conf = propose_intent('battery battery battery drain drain drain battery life power')
        assert conf <= 1.0


# ─── Risk flags ──────────────────────────────────────────────────────────────

class TestRiskFlags:
    def test_account_privacy_flag(self):
        assert 'privacy' in risk_flags('my Apple ID password will not work')

    def test_unsafe_hacked_flag(self):
        assert 'unsafe' in risk_flags('my account was hacked')

    def test_unsafe_scam_flag(self):
        assert 'unsafe' in risk_flags('I got a scam call about iCloud')

    def test_high_risk_overheating(self):
        assert 'high_risk' in risk_flags('my iPhone is overheating and smoking')

    def test_frustration_flag(self):
        assert 'frustration' in risk_flags("I'm done with Apple, switching to Android")

    def test_no_flags_for_normal_message(self):
        flags = risk_flags('how do I update my iPhone to iOS 17?')
        assert len(flags) == 0

    def test_multiple_flags_possible(self):
        flags = risk_flags('my Apple ID was hacked and I need my password reset urgently')
        assert 'unsafe' in flags
        assert 'privacy' in flags


# ─── Normalization ───────────────────────────────────────────────────────────

class TestNormalize:
    def test_strips_url(self):
        assert 'http' not in normalize('check this out https://apple.com/support')

    def test_strips_mention(self):
        assert '@AppleSupport' not in normalize('@AppleSupport my battery is draining')
        assert 'battery' in normalize('@AppleSupport my battery is draining')

    def test_lowercases(self):
        assert normalize('Battery DRAINING iOS') == 'battery draining ios'

    def test_deduplicates_whitespace(self):
        result = normalize('battery    draining   fast')
        assert '  ' not in result


# ─── Keyword scores ──────────────────────────────────────────────────────────

class TestKeywordScores:
    def test_scores_are_non_negative(self):
        scores = keyword_scores('my battery is dying after iOS update')
        for v in scores.values():
            assert v >= 0

    def test_all_intents_have_score(self):
        scores = keyword_scores('hello')
        assert set(scores.keys()) == set(INTENTS.keys())

    def test_multi_keyword_higher_score(self):
        single = keyword_scores('battery')['battery_power']
        multi = keyword_scores('battery drain battery life')['battery_power']
        assert multi > single
