#!/usr/bin/env python
import argparse, json, sys, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
except ImportError:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.model import IntentModel
from hiver_agent.retrieval import HistoricalRetriever
from hiver_agent.agent import SupportAgent
from hiver_agent.escalation import EscalationPolicy
from hiver_agent.evaluation import (
    majority_baseline, simple_keyword_baseline,
    classification_metrics, escalation_metrics, per_intent_metrics,
    evaluate_agent, run_llm_judge
)

p = argparse.ArgumentParser(description='Evaluate the Hiver support agent')
p.add_argument('--golden', required=True, help='Golden set CSV')
p.add_argument('--cases', required=True, help='AppleSupport cases CSV')
p.add_argument('--allow-draft', action='store_true',
               help='Allow draft (unapproved) golden rows — for dev smoke tests only')
p.add_argument('--results', default='results', help='Output directory')
p.add_argument('--brand', default='AppleSupport')
p.add_argument('--no-judge', action='store_true', help='Skip LLM judge even if API key is set')
args = p.parse_args()

Path(args.results).mkdir(parents=True, exist_ok=True)

# ---- Load and validate golden set ----------------------------------------
golden = pd.read_csv(args.golden).fillna('')
if 'status' in golden.columns and not args.allow_draft:
    non_approved = ~(golden.status == 'approved')
    if non_approved.any():
        raise SystemExit(
            f'Submission-grade evaluation requires every golden row to be human-approved.\n'
            f'Found {non_approved.sum()} non-approved rows.\n'
            f'Run: python scripts/approve_golden.py  OR  use --allow-draft for development.'
        )
    golden = golden[golden.status == 'approved'].copy()

# Normalise label columns: prefer reviewer_* over bare intent/expected_decision
golden['intent'] = golden.get('reviewer_intent', golden.get('intent', golden.get('reviewer_intent', '')))
if 'reviewer_intent' in golden.columns:
    golden['intent'] = golden['reviewer_intent'].where(golden['reviewer_intent'] != '', golden['intent'])
golden['expected_decision'] = golden.get('reviewer_decision', golden.get('expected_decision', ''))
if 'reviewer_decision' in golden.columns:
    golden['expected_decision'] = golden['reviewer_decision'].where(
        golden['reviewer_decision'] != '', golden['expected_decision']
    )

print(f'Golden set: {len(golden)} rows  (draft_allowed={args.allow_draft})')

# ---- Load training cases (excluding golden IDs) --------------------------
cases = pd.read_csv(args.cases).fillna('')
golden_ids = set(golden.customer_tweet_id.astype(str))
train = cases[~cases.customer_tweet_id.astype(str).isin(golden_ids)].copy()
train = train[train.weak_confidence >= 0.58]
print(f'Training cases: {len(train)} (after golden exclusion + confidence filter)')

# ---- Load or fit model / retriever ---------------------------------------
model_path = Path('data/processed/intent_model.joblib')
index_path = Path('data/processed/retriever.joblib')

if model_path.exists():
    model = IntentModel.load(model_path)
    print(f'Loaded intent model from {model_path}')
else:
    print('Fitting intent model...')
    model = IntentModel().fit(train.customer_text.astype(str), train.intent_proposed.astype(str))
    model.save(model_path)

if index_path.exists():
    retriever = HistoricalRetriever.load(index_path)
    print(f'Loaded retriever from {index_path}')
else:
    print('Fitting retriever (this may take ~30s)...')
    retriever = HistoricalRetriever(max_docs=15000).fit(train)
    retriever.save(index_path)

retriever.set_excluded_ids(golden_ids)
agent = SupportAgent(model, retriever, EscalationPolicy(0.62))

# ---- Baselines -----------------------------------------------------------
major = majority_baseline(train.intent_proposed, golden.customer_text)
kw = simple_keyword_baseline(golden.customer_text)

eval_status = 'DEVELOPMENT_ONLY_DRAFT_LABELS' if args.allow_draft else 'SUBMISSION_GRADE_HUMAN_APPROVED'

base = {
    'evaluation_status': eval_status,
    'golden_n': len(golden),
    'majority': {'classification': classification_metrics(golden.intent, major)},
    'keyword': {'classification': classification_metrics(golden.intent, kw)},
}

# ---- Agent evaluation ----------------------------------------------------
print('Running agent evaluation...')
pred, agent_metrics = evaluate_agent(agent, golden)
pred.to_csv(Path(args.results) / 'agent_predictions.csv', index=False)
base['agent'] = agent_metrics

print(f"\nIntent accuracy:        {agent_metrics['classification']['accuracy']:.3f}")
print(f"Intent macro-F1:        {agent_metrics['classification']['macro_f1']:.3f}")
if 'escalation' in agent_metrics:
    e = agent_metrics['escalation']
    print(f"Escalate precision:     {e['precision_escalate']:.3f}")
    print(f"Escalate recall:        {e['recall_escalate']:.3f}")
    print(f"False auto-handle rate: {e['false_auto_handle_rate']:.3f}  ← key safety metric")

# ---- Per-intent breakdown ------------------------------------------------
per_intent = per_intent_metrics(pred.gold_intent, pred.pred_intent)
per_intent.to_csv(Path(args.results) / 'per_intent_metrics.csv', index=False)
print(f"\nPer-intent F1 range: {per_intent.f1.min():.3f} – {per_intent.f1.max():.3f}")

# ---- Confusion matrix + classification report ----------------------------
labels = sorted(set(golden.intent.astype(str)))
import sklearn.metrics as skm
cm = pd.DataFrame(
    skm.confusion_matrix(golden.intent, pred.pred_intent, labels=labels),
    index=labels, columns=labels
)
cm.to_csv(Path(args.results) / 'confusion_matrix.csv')

rep = skm.classification_report(golden.intent, pred.pred_intent, output_dict=True, zero_division=0)
pd.DataFrame(rep).T.to_csv(Path(args.results) / 'classification_report.csv')

# ---- LLM judge (optional) ------------------------------------------------
if not args.no_judge:
    judge = run_llm_judge(pred.head(50), Path(args.results) / 'reply_judge_results.csv')
else:
    judge = None

if judge is None:
    base['reply_judge'] = {'status': 'not_run', 'reason': 'OPENAI_API_KEY not configured or --no-judge'}
else:
    numeric_cols = [c for c in ['correctness', 'historical_grounding', 'actionability',
                                 'tone', 'safety', 'overall'] if c in judge.columns]
    pass_rate = float(judge['pass'].dropna().astype(int).mean()) if 'pass' in judge.columns else None
    base['reply_judge'] = {
        'status': 'run',
        'n': len(judge),
        'pass_rate': round(pass_rate, 3) if pass_rate is not None else None,
        'mean_scores': {c: round(float(judge[c].dropna().mean()), 3) for c in numeric_cols},
    }

# ---- Manifest + final output ---------------------------------------------
manifest = {
    'brand': args.brand,
    'golden_rows': len(golden),
    'train_rows': len(train),
    'llm_judge_run': not args.no_judge and os.getenv('OPENAI_API_KEY') is not None,
    'draft_allowed': args.allow_draft,
    'evaluation_status': eval_status,
}
Path(args.results, 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
Path(args.results, 'metrics.json').write_text(json.dumps(base, indent=2))
print(f'\nFull metrics written to {args.results}/metrics.json')
print(json.dumps({k: v for k, v in base.items() if k != 'calibration'}, indent=2))
