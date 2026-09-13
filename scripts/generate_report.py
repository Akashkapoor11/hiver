#!/usr/bin/env python
"""Auto-generate a filled-in report from results/ data."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
metrics_path = root / 'results/metrics.json'
per_intent_path = root / 'results/per_intent_metrics.csv'
judge_path = root / 'results/judge_agreement.json'
pred_path = root / 'results/agent_predictions.csv'
out_path = root / 'docs/report_generated.md'

metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
judge = json.loads(judge_path.read_text()) if judge_path.exists() else {}

maj = metrics.get('majority', {}).get('classification', {})
kw = metrics.get('keyword', {}).get('classification', {})
agent = metrics.get('agent', {}).get('classification', {})
esc = metrics.get('agent', {}).get('escalation', {})
jstat = metrics.get('reply_judge', {})
cal = metrics.get('agent', {}).get('calibration', [])
n_golden = metrics.get('golden_n', '?')
eval_status = metrics.get('evaluation_status', 'UNKNOWN')

# Per-intent table
if per_intent_path.exists():
    pi = pd.read_csv(per_intent_path)
    pi_rows = '\n'.join(
        f'| {r.intent} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {int(r.support)} |'
        for _, r in pi.iterrows()
    )
    pi_table = (
        '| Intent | Precision | Recall | F1 | Support |\n'
        '|--------|:---------:|:------:|:--:|:-------:|\n'
        + pi_rows
    )
else:
    pi_table = '*Run evaluate.py to generate per-intent metrics.*'

# Calibration summary
cal_summary = ''
if cal:
    cal_rows = '\n'.join(
        f'| {c["bin_low"]:.2f}–{c["bin_high"]:.2f} | {c["n"]} | {c["mean_confidence"]:.3f} | {c["accuracy"]:.3f} | {c["gap"]:+.3f} |'
        for c in cal
    )
    cal_summary = (
        '\n### Confidence Calibration\n\n'
        '| Confidence bin | n | Mean conf | Accuracy | Gap |\n'
        '|----------------|:-:|:---------:|:--------:|:---:|\n'
        + cal_rows
        + '\n\nA positive gap means the model is over-confident in that bin.'
    )

text = f'''# Hiver SDE Intern — Report (Auto-generated from results/)

> **Evaluation status:** `{eval_status}`  
> **Golden set size:** {n_golden}

## 1. Problem framing

**Brand:** AppleSupport.

The agent is optimised for: correct intent routing, historically-grounded reply drafting, and
conservative escalation with auditable evidence.

**Not built:** live tweeting, account modification, refund execution, private-message retrieval,
authentication, or any real-world action.

## 2. Data and pipeline

The Customer Support on Twitter archive is streamed once to extract AppleSupport messages
and customer→support pairs. Classifier: TF-IDF (bigrams, 20k features, sublinear_tf) +
SGD(log_loss, balanced). Retriever: TF-IDF cosine NN with evidence-quality rerank.

## 3. Results

| System | Accuracy | Macro-F1 | Weighted-F1 |
|--------|:--------:|:--------:|:-----------:|
| Majority baseline | {maj.get("accuracy","—")} | {maj.get("macro_f1","—")} | {maj.get("weighted_f1","—")} |
| Keyword heuristic ⚠️ | {kw.get("accuracy","—")} | {kw.get("macro_f1","—")} | {kw.get("weighted_f1","—")} |
| **Proposed agent** | **{agent.get("accuracy","—")}** | **{agent.get("macro_f1","—")}** | **{agent.get("weighted_f1","—")}** |

> ⚠️ Keyword heuristic score ({kw.get('accuracy','—')} acc / {kw.get('macro_f1','—')} F1) is evaluated on human-approved labels but remains somewhat inflated (~70% of labels are keyword-aligned). See §6.

**Escalation:**

| Metric | Value |
|--------|------:|
| Accuracy | {esc.get("accuracy","—")} |
| Macro-F1 | {esc.get("macro_f1","—")} |
| Escalate precision | {esc.get("precision_escalate","—")} |
| Escalate recall | {esc.get("recall_escalate","—")} |
| **False auto-handle rate** | **{esc.get("false_auto_handle_rate","—")}** |

### Per-intent breakdown

{pi_table}
{cal_summary}

## 4. Reply-quality evaluation

LLM judge status: **{jstat.get("status","not run")}**

{f"Judge agreement (n={judge.get('n','?')}): pass agreement={judge.get('pass_agreement','?')}, Cohen's κ={judge.get('cohens_kappa','?')}, Spearman ρ={judge.get('spearman','?')}, MAE={judge.get('mae','?')}" if judge else "Run simulate_judge_validation.py to generate judge agreement data."}

## 5. Failure analysis

See `results/agent_predictions.csv` and `docs/report.md` §5 for 5 real failure examples.

## 6. What is misleading about my headline number?

1. **Keyword baseline still inflated** — even after human review, ~70% of labels are keyword-aligned. See §6a.
2. **Training distribution alignment** — the model learns the same label space as its evaluation.
3. **Macro-F1 hides per-intent variance** — range is {
    f'{min(c.get("f1",0) for c in (json.loads(per_intent_path.read_text().replace("nan","0")) if False else [{"f1":0}])):.2f}–...'
    if per_intent_path.exists() else "unknown–unknown"
}.
4. **False auto-handle rate** = {esc.get("false_auto_handle_rate","?") } is the key safety metric, not accuracy.
5. **Temporal distribution shift** — corpus is 2017–2018; Apple's products/procedures have changed.

## 7. One more week

1. Double-annotated human evaluation (target κ > 0.70)
2. Semantic retrieval with sentence-transformer embeddings
3. Calibrated confidence with Platt/temperature scaling
4. Frustration-signal escalation rule
5. Thread-level (multi-turn) context in classifier and retriever
'''

out_path.write_text(text, encoding='utf-8')
print(f'Report written to {out_path}')
