# Hiver SDE Intern — Report (Generated)

## 1. Problem framing

**Brand:** AppleSupport.

The agent is optimized for correct intent routing, historically grounded reply drafting, and conservative escalation with auditable evidence. It does not send tweets, modify accounts, execute refunds, authenticate customers, or access private customer records.

## 2. Data and pipeline

The supplied Customer Support on Twitter archive is streamed once to extract AppleSupport messages and customer→support pairs. The classifier uses TF-IDF + logistic regression with a domain-keyword override; retrieval uses TF-IDF nearest neighbors with evidence-quality reranking.

## 3. Results

| System | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority baseline | 0.1 | 0.0182 | 0.0182 |
| Keyword heuristic | 1.0 | 1.0 | 1.0 |
| Proposed agent | 0.605 | 0.5872 | 0.5872 |

Escalation accuracy: 0.84 ; escalate precision: 0.9821 ; escalate recall: 0.8505.

> **Important:** results are only submission-grade after every golden example is human-approved. The repository's smoke metrics use the assistant-proposed seed and are intentionally not presented as the final application metric.

## 4. Reply-quality evaluation

The LLM judge scores correctness, historical grounding, actionability, tone, safety, and unsupported claims. The judge must be calibrated against a separately human-scored validation set before its score is used as evidence. Current judge status: **not_run**.

## 5. Failure analysis

Use `results/agent_predictions.csv` to select five real examples. Recommended failure buckets are ambiguity, multi-intent messages, weak historical precedent, wrong escalation, and retrieval mismatch. For each include the message, system output, expected output, hypothesis, and fix.

## 6. What is misleading about my headline number?

A single accuracy or macro-F1 number can hide rare-intent failures, ambiguous cases, history-dependent performance, and unsafe escalation errors. Historical support data is also not the same as a live support distribution. LLM judges can favor fluency unless human-calibrated. The system's most important operational errors are unsupported responses and false auto-handling, not just ordinary classification mistakes.

## 7. One more week

Next priorities: double-annotated intent labels, calibrated confidence, hard-negative retrieval, stronger multilingual/typo handling, richer thread-level context, and comparison with an embedding/reranker model under the same held-out golden set.
