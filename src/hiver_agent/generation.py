from __future__ import annotations
import json, os, re
from typing import List, Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SYSTEM_PROMPT = """You are an evidence-first customer support drafter for AppleSupport.
Use only the provided customer message and historical support examples.
Do not invent account facts, policies, refunds, eligibility, prices, or diagnostics.
Prefer the brand's observed support behavior and ask for the smallest missing detail when needed.
Never ask a customer to post private credentials, full card numbers, passwords, or security codes publicly.
Return JSON with keys: reply, rationale, evidence_ids, unsupported_claims.
"""

def _clean_response(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fallback_reply(customer_text: str, intent: str, evidence: List[Dict], escalate: bool) -> str:
    snippets = [_clean_response(e.get('support_text','')) for e in evidence if e.get('support_text')]
    first = snippets[0] if snippets else ''
    # Remove handle and URL noise from historical text before adapting it.
    first = re.sub(r'https?://\S+', '', first)
    first = re.sub(r'@[A-Za-z0-9_]+', '', first)
    first = _clean_response(first)
    if first and len(first) > 220:
        first = first[:220].rsplit(' ',1)[0] + '…'
    if escalate:
        if first:
            return f"Thanks for reaching out. {first} Because this may need account-specific help, please continue with a private support channel rather than sharing sensitive details publicly."
        return "Thanks for reaching out. This may need account-specific assistance, so please continue with Apple Support through a private support channel and avoid posting sensitive information publicly."
    if first:
        return f"Thanks for reaching out. {first}"
    return "Thanks for reaching out. Could you share the device/model and software version where you're seeing this issue?"

def draft_reply(customer_text: str, intent: str, evidence: List[Dict], escalate: bool, model: str | None = None) -> Dict:
    key = os.getenv('OPENAI_API_KEY')
    if not key or OpenAI is None:
        return {'reply': fallback_reply(customer_text, intent, evidence, escalate), 'mode':'deterministic_fallback', 'rationale':'Generated from the highest-similarity historical support response with a conservative escalation policy.', 'evidence_ids':[e.get('support_tweet_id') for e in evidence], 'unsupported_claims':[]}
    client = OpenAI(api_key=key)
    evidence_block = '\n'.join([f"ID={e.get('support_tweet_id')} | customer={e.get('customer_text')} | support={e.get('support_text')} | similarity={e.get('similarity')}" for e in evidence])
    prompt = f"""Customer message:\n{customer_text}\n\nPredicted intent: {intent}\nEscalation: {escalate}\n\nHistorical evidence:\n{evidence_block}\n\nDraft a concise, brand-consistent reply. If escalation is true, make the need for private human help explicit. Do not claim an issue is fixed unless the evidence supports it."""
    try:
        response = client.responses.create(model=model or os.getenv('OPENAI_MODEL','gpt-4o'), instructions=SYSTEM_PROMPT, input=prompt)
        raw = response.output_text.strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.I|re.S).strip()
        data = json.loads(raw)
        data['mode'] = 'openai_responses_api'
        return data
    except Exception as exc:
        return {'reply': fallback_reply(customer_text, intent, evidence, escalate), 'mode':'fallback_after_api_error', 'rationale':f'LLM generation failed safely: {type(exc).__name__}.', 'evidence_ids':[e.get('support_tweet_id') for e in evidence], 'unsupported_claims':['LLM call failed; deterministic fallback used.']}
