# Results

Pre-computed outputs committed to the repo so reviewers can verify results **without running the full pipeline**.

| File | Content |
|------|---------|
| `metrics.json` | Classification accuracy, macro-F1, escalation precision/recall, `false_auto_handle_rate` |
| `agent_predictions.csv` | 200-row row-by-row agent output vs gold labels |
| `classification_report.csv` | Per-intent precision, recall, F1, support |
| `confusion_matrix.csv` | 10×10 intent confusion matrix |
| `judge_agreement.json` | Human-vs-LLM judge agreement: Cohen's κ, Spearman ρ, MAE |
| `run_manifest.json` | Pipeline metadata: brand, golden_rows, train_rows, eval_status |

To regenerate all results from scratch:
```bash
make reproduce
```
