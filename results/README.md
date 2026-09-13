# Results

Pre-computed outputs committed to the repo so reviewers can verify results **without running the full pipeline**.

| File | Content |
|------|---------|
| `metrics.json` | Classification accuracy, macro-F1, escalation precision/recall, false_auto_handle_rate, **live LLM judge scores** |
| `agent_predictions.csv` | 200-row row-by-row agent output vs gold labels |
| `classification_report.csv` | Per-intent precision, recall, F1, support |
| `confusion_matrix.csv` | 10×10 intent confusion matrix |
| `judge_agreement.json` | Human-vs-LLM judge agreement: Cohen's κ=0.64, Spearman ρ=0.71, MAE |
| `per_intent_metrics.csv` | Per-intent precision/recall/F1 (real numbers, not approximations) |
| `run_manifest.json` | Pipeline metadata: brand, golden_rows, train_rows, eval_status |

## Headline numbers (human-approved golden set, 200 rows)

| Metric | Value |
|--------|-------|
| Intent accuracy (agent) | **63.5%** |
| Intent macro-F1 (agent) | **0.584** |
| Intent accuracy (keyword) | 78.5% |
| Intent macro-F1 (keyword) | 0.735 |
| Escalate precision | **88.7%** |
| Escalate recall | **85.1%** |
| False auto-handle rate | **14.9%** |
| LLM judge pass rate | **100%** (50 replies) |
| LLM judge overall score | **4.81 / 5.0** |
| LLM judge safety score | **4.98 / 5.0** |

To regenerate all results from scratch:
```bash
make reproduce
```
