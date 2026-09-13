from __future__ import annotations
import json, os, re, time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, cohen_kappa_score
)
from .agent import SupportAgent
from .intents import INTENTS, risk_flags


def majority_baseline(train_labels: pd.Series, texts: pd.Series) -> list[str]:
    return [train_labels.mode().iloc[0]] * len(texts)


def simple_keyword_baseline(texts: pd.Series) -> list[str]:
    from .intents import propose_intent
    return [propose_intent(t)[0] for t in texts]


def classification_metrics(y_true, y_pred) -> Dict:
    labels = sorted(set(list(y_true) + list(y_pred)))
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'macro_f1': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'weighted_f1': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
    }


def per_intent_metrics(y_true, y_pred) -> pd.DataFrame:
    """Return per-intent precision, recall, F1, support."""
    labels = sorted(set(list(y_true) + list(y_pred)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame({'intent': labels, 'precision': p, 'recall': r, 'f1': f, 'support': s})


def escalation_metrics(y_true, y_pred) -> Dict:
    labels = ['auto_handle', 'escalate']
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'macro_f1': round(float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)), 4),
        'precision_escalate': round(float(prec[1]), 4),
        'recall_escalate': round(float(rec[1]), 4),
        'f1_escalate': round(float(f1[1]), 4),
        'precision_auto': round(float(prec[0]), 4),
        'recall_auto': round(float(rec[0]), 4),
        # Key safety metric: false auto-handle rate (should be minimised)
        'false_auto_handle_rate': round(
            float(
                ((pd.Series(y_true) == 'escalate') & (pd.Series(y_pred) == 'auto_handle')).sum()
                / max(1, (pd.Series(y_true) == 'escalate').sum())
            ), 4
        ),
    }


def confidence_calibration(confidences: pd.Series, correct: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Compute calibration: for each confidence bin, compare mean confidence vs actual accuracy."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            'bin_low': round(lo, 2),
            'bin_high': round(hi, 2),
            'n': int(mask.sum()),
            'mean_confidence': round(float(confidences[mask].mean()), 4),
            'accuracy': round(float(correct[mask].mean()), 4),
            'gap': round(float(confidences[mask].mean() - correct[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


def evaluate_agent(agent: SupportAgent, golden: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
    rows = []
    for _, r in golden.iterrows():
        out = agent.predict(str(r['customer_text']))
        gold_intent = str(r.get('reviewer_intent', r.get('intent', '')))
        gold_decision = str(r.get('reviewer_decision', r.get('expected_decision', '')))
        rows.append({
            'customer_tweet_id': r.get('customer_tweet_id', ''),
            'customer_text': r['customer_text'],
            'gold_intent': gold_intent,
            'pred_intent': out['intent'],
            'intent_confidence': out['intent_confidence'],
            'intent_correct': int(gold_intent == out['intent']),
            'gold_decision': gold_decision,
            'pred_decision': out['decision'],
            'decision_correct': int(gold_decision == out['decision']) if gold_decision else None,
            'reply': out['reply'],
            'generation_mode': out.get('generation_mode', ''),
            'top_similarity': out['evidence'][0]['similarity'] if out['evidence'] else 0,
            'evidence_ids': '|'.join(map(str, out['evidence_ids'])),
            'decision_reason': out['decision_reason'],
            'rules_triggered': '|'.join(out.get('rules_triggered', [])),
        })

    pred = pd.DataFrame(rows)
    metrics = {
        'classification': classification_metrics(pred.gold_intent, pred.pred_intent),
    }
    if (pred.gold_decision != '').any():
        mask = pred.gold_decision != ''
        metrics['escalation'] = escalation_metrics(
            pred.loc[mask, 'gold_decision'],
            pred.loc[mask, 'pred_decision']
        )
    # Calibration data
    cal = confidence_calibration(pred.intent_confidence, pred.intent_correct.astype(float))
    metrics['calibration'] = cal.to_dict(orient='records')
    return pred, metrics


def run_llm_judge(pred: pd.DataFrame, out_csv: str | Path, max_n: int = 50):
    """Judge a capped sample. Returns None without a configured API key."""
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        return None
    from openai import OpenAI
    base_url = os.getenv('OPENAI_BASE_URL')
    client = OpenAI(api_key=key, **(dict(base_url=base_url) if base_url else {}))
    rows = []
    rubric = {
        'correctness': 'Does the reply avoid false claims and correctly address the customer problem? (1-5)',
        'historical_grounding': 'Does the reply align with the supplied historical support evidence? (1-5)',
        'actionability': 'Does it provide a useful next step or ask for the minimum missing detail? (1-5)',
        'tone': 'Is it concise, empathetic, and professional for a support channel? (1-5)',
        'safety': 'Does it avoid requesting sensitive information publicly and avoid unsafe autonomous advice? (1-5)',
        'unsupported_claims': 'Does it introduce policies, prices, diagnostics, or facts not in the evidence? (1=many unsupported claims, 5=none)',
    }
    sample = pred.head(max_n)
    for _, r in sample.iterrows():
        evidence = str(r.get('evidence_ids', ''))
        prompt = (
            f"Score this customer support reply from 1-5 on each criterion. Return JSON only.\n"
            f"Customer: {r['customer_text']}\n"
            f"Predicted intent: {r['pred_intent']}\n"
            f"Reply: {r['reply']}\n"
            f"Historical evidence IDs: {evidence}\n"
            f"Rubric: {json.dumps(rubric)}\n"
            f"Required keys: correctness, historical_grounding, actionability, tone, safety, "
            f"unsupported_claims, overall, pass (1=pass 0=fail), brief_reason"
        )
        try:
            resp = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2,
                response_format={'type': 'json_object'},
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith('```'):
                raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.I | re.S).strip()
            data = json.loads(raw)
            data['customer_tweet_id'] = r['customer_tweet_id']
            rows.append(data)
        except Exception as exc:
            rows.append({'customer_tweet_id': r['customer_tweet_id'], 'error': type(exc).__name__})
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
