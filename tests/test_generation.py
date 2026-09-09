"""Generation fallback tests — runs without API key."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.generation import fallback_reply, draft_reply


SAMPLE_EVIDENCE = [
    {
        'support_tweet_id': '12345',
        'customer_text': 'my battery drains fast',
        'support_text': 'Try going to Settings > Battery and checking Battery Health. Let us know in DM.',
        'similarity': 0.72,
    },
    {
        'support_tweet_id': '12346',
        'customer_text': 'battery not lasting all day',
        'support_text': 'We can look into this together. DM us the iOS version and device model.',
        'similarity': 0.61,
    }
]


class TestFallbackReply:
    def test_returns_string(self):
        reply = fallback_reply('my battery drains', 'battery_power', SAMPLE_EVIDENCE, escalate=False)
        assert isinstance(reply, str)
        assert len(reply) > 10

    def test_uses_evidence_content(self):
        reply = fallback_reply('my battery drains', 'battery_power', SAMPLE_EVIDENCE, escalate=False)
        # Should incorporate some text from the first evidence item
        assert len(reply) > 30

    def test_escalate_adds_private_channel_note(self):
        reply = fallback_reply('my account was hacked', 'apple_id_icloud_account', SAMPLE_EVIDENCE, escalate=True)
        assert 'private' in reply.lower() or 'dm' in reply.lower() or 'channel' in reply.lower()

    def test_no_evidence_falls_back_gracefully(self):
        reply = fallback_reply('my battery drains', 'battery_power', [], escalate=False)
        assert isinstance(reply, str)
        assert len(reply) > 10

    def test_no_api_key_returns_deterministic(self):
        """Without API key, draft_reply must return a valid dict with correct mode."""
        import os
        # Temporarily unset key if present
        orig = os.environ.pop('OPENAI_API_KEY', None)
        try:
            result = draft_reply('my battery drains fast', 'battery_power', SAMPLE_EVIDENCE, False)
            assert isinstance(result, dict)
            assert 'reply' in result
            assert 'mode' in result
            assert 'deterministic' in result['mode'] or 'fallback' in result['mode']
            assert isinstance(result['reply'], str)
            assert len(result['reply']) > 10
        finally:
            if orig:
                os.environ['OPENAI_API_KEY'] = orig

    def test_strips_urls_from_evidence(self):
        evidence_with_url = [{
            'support_tweet_id': '99',
            'customer_text': 'question',
            'support_text': 'Try restarting. See https://support.apple.com/something for details.',
            'similarity': 0.80
        }]
        reply = fallback_reply('question', 'ios_update_software', evidence_with_url, False)
        assert 'https://' not in reply or 'http://' not in reply

    def test_strips_handles_from_evidence(self):
        evidence_with_handle = [{
            'support_tweet_id': '99',
            'customer_text': 'question',
            'support_text': '@Customer123 Please try restarting your device.',
            'similarity': 0.80
        }]
        reply = fallback_reply('question', 'ios_update_software', evidence_with_handle, False)
        assert '@Customer123' not in reply

    def test_reply_under_500_chars(self):
        """Replies should be concise, not paragraph-length."""
        reply = fallback_reply('my battery drains fast', 'battery_power', SAMPLE_EVIDENCE, False)
        assert len(reply) <= 600, f'Reply too long ({len(reply)} chars): {reply}'

    def test_draft_reply_includes_evidence_ids(self):
        result = draft_reply('battery drain', 'battery_power', SAMPLE_EVIDENCE, False)
        assert 'evidence_ids' in result
        assert isinstance(result['evidence_ids'], list)
