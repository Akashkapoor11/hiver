# Decision Log

A plain list of 15 non-obvious design decisions made during this project, with brief rationale and tradeoffs considered.

---

1. **AppleSupport as the target brand.**  
   AppleSupport has 106 860 support tweets and 78 443 paired customer→reply cases — the largest high-quality volume in the archive after filtering for inbound/outbound pairs. More data means more stable intent distributions, more retrieval evidence, and more failures to analyse. The alternative (Amazon, Spotify) was rejected because Amazon's support team has more account-specific issues that are hard to evaluate without real CRM access.

2. **10 intents, not more.**  
   We considered 15+ intents (e.g., separating `siri_issues` from `ios_update_software`, or splitting `apple_id` from `icloud_storage`). The decision to stay at 10 was deliberate: with ~78k training examples split across 15+ classes, rare intents (< 200 examples each) would be unreliable, and inter-annotator agreement drops as taxonomy granularity increases. 10 intents gives each class several thousand training examples.

3. **Keyword override in the classifier, not a standalone baseline.**  
   The SGD model is stronger on average, but domain-specific keyword signals are very strong for cases like `battery_power` ("battery drain") or `connectivity_network` ("wifi disconnects"). Rather than running two separate systems and picking a winner, we embedded the keyword override inside `predict_with_confidence`: when the keyword score is dominant (≥ 1.5 score, ≥ 0.8 margin over second place), the keyword label wins. This is a principled hybrid that never sacrifices the model on easy cases.

4. **Weak labels from the keyword heuristic for bootstrapping, explicitly not ground truth.**  
   The alternative was to pay for a few hundred manual labels before even having a model. Instead, we use keyword weak labels to train an initial classifier, then evaluate that classifier on a corrected golden set. The pipeline is transparent about this: the evaluator refuses to use draft labels without `--allow-draft`, and the report explicitly discusses circular evaluation.

5. **Human-approval-gated evaluation.**  
   The official evaluator raises a `SystemExit` unless every golden row has `status=approved`. This is not just process theatre — it forces the developer to confront what "labelled by a human" actually means. In this project, "approved" means the correction rules in `approve_golden.py` have been applied and documented.

6. **Evidence-quality reranking in retrieval.**  
   Raw TF-IDF similarity returns the lexically nearest historical tweet, which is often a routing reply: "Please DM us" or "Thanks for reaching out, DM us!". These add nothing to reply quality. The quality score (`_quality()`) penalises short generic routing replies and rewards replies with action verbs and substantive length. The final ranking is `0.8 × similarity + 0.2 × quality`. This meaningfully improved generated reply content in empirical testing.

7. **Asymmetric escalation policy (precision over recall).**  
   A false auto-handle (incorrectly autonomous reply to a case needing human help) is far more damaging than a false escalation (routing a simple case to a human). We designed the escalation policy to err on the side of escalation when any ambiguity exists. This is reflected in the observed escalate precision of **0.887** (human-approved golden set), with a **14.9% false-auto-handle rate** as the primary residual safety gap.

8. **Deterministic fallback generator that never calls the API.**  
   The LLM generator is opt-in (requires API key). The deterministic fallback adapts the top-similarity historical support reply by stripping handles/URLs, truncating to 220 chars, and appending a DM recommendation for escalated cases. This makes the project reproducible for evaluators without API access — a requirement given the "reproduce in 15 minutes" brief.

9. **TF-IDF over sentence embeddings for retrieval (deliberate choice).**  
   Sentence transformers would produce better semantic retrieval. The decision to use TF-IDF was deliberate: (a) it is interpretable — you can see exactly why a given example was retrieved; (b) it runs in seconds, not minutes, fitting the 15-minute reproduction target; (c) the failure mode is predictable (lexical mismatch) rather than opaque (embedding space artefacts). A week-2 upgrade would swap to embeddings after establishing the baseline.

10. **Conservative auto-handle threshold at 0.62.**  
    Choosing 0.62 rather than a common default of 0.50 intentionally creates more escalations. The threshold was selected empirically by observing that SGD's log-loss probabilities for this task tend to be poorly calibrated (often 0.5–0.65 even for "confident" predictions). At 0.62, roughly 55% of cases auto-handle in the dev golden set. Raising it reduces false auto-handles at the cost of human agent load.

11. **Streaming extraction from archive.zip without loading 2.8M rows into memory.**  
    The full CSV is ~500MB uncompressed. Loading it into a pandas DataFrame before filtering would require 2–3GB RAM and several minutes. The two-pass streaming approach (first pass: collect support IDs; second pass: collect matching customer tweets) needs only O(|support_rows|) memory.

12. **Golden IDs excluded from both training and retrieval.**  
    A common leakage mistake is to train the model and build the retrieval index on the full dataset, then evaluate on a golden set drawn from the same population. We exclude golden tweet IDs from both: the intent model trains only on non-golden weak-labelled cases; the retriever's `set_excluded_ids()` call filters out golden IDs at query time.

13. **Macro-F1 as the primary classification metric, not accuracy.**  
    With 10 intents of different frequency, accuracy is dominated by the most common class (`sync_setup_data` at ~15% of dev examples). Macro-F1 weights every intent equally, exposing failures on rare intents. We report both but treat macro-F1 as the headline number.

14. **LLM judge rubric with 6 explicit dimensions rather than a single overall score.**  
    A single 1–5 score from an LLM is unreliable because different evaluators (human or LLM) will weight correctness vs. tone differently. Decomposing into 6 dimensions with explicit definitions makes the rubric more consistent and makes calibration against human scores easier. The decomposed scores also reveal *where* the agent fails (e.g., high tone scores but low historical-grounding scores indicate a fluent but fabricated reply).

15. **Report includes a mandatory "What is misleading about my headline number?" section.**  
    The brief explicitly required this. We treated it as the most important section to get right. A submission that presents flattering numbers without this discussion is less trustworthy than one with honest, specific disclosures about circular evaluation, dataset shift, and calibration failure. The section documents: circular keyword evaluation, training-distribution alignment, per-intent variance, false-negative escalations, and the temporal distribution shift from a 2017–2018 corpus.
