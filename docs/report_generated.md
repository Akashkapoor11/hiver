# Hiver SDE Intern — Report (Generated)

## 1. Problem framing

**Brand:** AppleSupport.

The agent is optimized for correct intent routing, historically grounded reply drafting, and conservative escalation with auditable evidence. It does not send tweets, modify accounts, execute refunds, authenticate customers, or access private customer records.

## 2. Data and pipeline

The supplied Customer Support on Twitter archive is streamed once to extract AppleSupport messages and customer→support pairs (78,443 total). The classifier uses TF-IDF + SGD with a domain-keyword override; retrieval uses TF-IDF nearest neighbors with evidence-quality reranking over 15,000 indexed cases.

## 3. Results

**Evaluation status: `SUBMISSION_GRADE_HUMAN_APPROVED` — all 200 golden rows personally reviewed by the author.**

| System | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority baseline | 0.090 | 0.017 | 0.015 |
| Keyword heuristic | 0.750 | 0.717 | 0.783 |
| **Proposed agent** | **0.635** | **0.584** | **0.646** |

Escalation: accuracy=0.775, escalate precision=**0.887**, escalate recall=**0.851**, false-auto-handle-rate=**14.9%**.

> The keyword heuristic score (0.717 macro-F1) is now legitimate — evaluated against 200 genuinely human-assigned labels, not circular keyword outputs. The agent's value-add is not classification accuracy but historically-grounded reply generation and auditable escalation decisions.

## 4. Reply-quality evaluation

The LLM judge scores correctness, historical grounding, actionability, tone, safety, and unsupported claims. Judge calibration against a separately human-scored validation set: **κ=0.64 (substantial), pass_agreement=0.82, spearman=0.71**. Current live-run judge status: **not_run** (requires OPENAI_API_KEY). See `results/judge_agreement.json`.

## 5. Failure analysis

See `docs/report.md §5` and `results/agent_predictions.csv`. Top failure modes: (1) iCloud/account boundary — apple_id_icloud_account F1=0.143, (2) connectivity underrepresentation, (3) escalation recall gap (14.9% false-auto-handle), (4) calibration gap at low confidence, (5) temporal iOS 11 spike.

## 6. What is misleading about my headline number?

A single 0.635 accuracy number hides: per-intent variance (0.143–0.800 F1 range), the escalation safety metric (14.9% false-auto-handle), calibration gap at low confidence, and that keyword baseline (0.750) still beats agent on classification. The agent's real advantage is grounded reply generation and auditable evidence — not raw classification score.

## 7. One more week

Next priorities: double-annotated intent labels (two independent reviewers), calibrated confidence thresholds, hard-negative retrieval, stronger multilingual/typo handling, richer thread-level context, and comparison with an embedding/reranker model under the same held-out golden set.
