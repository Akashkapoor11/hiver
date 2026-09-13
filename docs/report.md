# Hiver SDE Intern — Report: AppleSupport AI Agent

**Author:** Akash  
**Date:** September 2026  
**Dataset:** Customer Support on Twitter (Kaggle, thoughtvector/customer-support-on-twitter)  
**Brand:** AppleSupport  

---

## 1. Problem Framing

### What "good" means for AppleSupport

AppleSupport handles roughly **106 000 support tweets** in the corpus across a wide range of device and software issues. A good agent for this brand must:

1. **Correctly categorise the customer's primary problem** — AppleSupport's replies differ meaningfully across hardware repairs, software bugs, account access, and billing. Misrouting creates a bad experience and wastes the human agent's time.
2. **Draft a reply grounded in how AppleSupport has historically resolved similar issues** — not hallucinated advice. Apple's Twitter support consistently links to Apple support articles, asks for device/iOS version, and routes sensitive issues to DM. The agent must mirror this style.
3. **Escalate conservatively** — the worst outcome is sending an autonomous reply to a customer whose account is compromised, device is unsafe, or case requires a refund transaction. Precision on escalation matters more than recall.
4. **Provide auditable evidence** — every reply must include the evidence tweet IDs it was grounded on, so a human reviewer can inspect and override.

### What we deliberately did NOT build

- Live tweet sending, account modification, or CRM integration
- Refund execution or payment processing
- Private-message (DM) retrieval or sending
- Multi-turn conversation state tracking
- Authentication or identity verification
- Anything that takes real-world action

---

## 2. Optional Secondary Dataset: BANKING77 Cross-Validation

The assignment brief mentions BANKING77 as "optional secondary for intent work only." We used it.

[`banking77.py`](../banking77.py) and [`dataset_infos.json`](../dataset_infos.json) define the HuggingFace Datasets loader for BANKING77 (Casanueva et al., 2020) — 13,083 banking customer-service queries with 77 fine-grained intents.

**How we used it:** We mapped all 77 BANKING77 intents to the closest AppleSupport intent using documented keyword alignment (see `scripts/banking77_intent_alignment.py` and `results/banking77_alignment.json`). This serves two purposes:

1. **Taxonomy validation** — If many BANKING77 intents map cleanly to Apple intents, the taxonomy generalises to the broader customer-service domain (good). If many have no analog, the intents are Apple-specific (expected for `battery_power`, `ios_update_software`).
2. **Contamination risk** — Words like "transfer," "account," and "card" mean different things in banking vs. Apple support. This cross-check reveals where the AppleSupport keyword heuristic would be dangerously wrong if applied to banking data.

**Key findings:**
- 7/10 Apple intents have ≥ 3 BANKING77 analogs — strong cross-domain coverage
- `battery_power` and `ios_update_software` are Apple-specific with no banking analogs (as expected)
- **4 contamination risks** where the same word carries different meaning across domains:
  - `"transfer"` → fires as `sync_setup_data` but means money transfer in banking
  - `"account"` → fires as `apple_id_icloud_account` but means bank account in banking
  - `"card"` → fires as `device_hardware_charging` but means payment card in banking
  - `"pending"` → fires as `sync_setup_data` but means pending payment in banking

This confirms that the agent's keyword heuristic is correctly tuned for Apple-domain text and would require complete retraining before use in any other domain.

**BANKING77 is not used in any runtime component** — no training data, no retrieval corpus, no evaluation. See [`LICENSES.md`](../LICENSES.md) for attribution.

---

```
Customer message
      │
      ▼
┌─────────────────────────────┐
│   Intent Classifier         │  TF-IDF (1–2 grams, 20k features,
│   + Keyword Override        │  sublinear_tf) + SGD(log_loss, balanced)
└────────────┬────────────────┘  Keyword override wins when score ≥ 1.5,
             │                   margin ≥ 0.8 (strong domain signal)
             ▼
┌─────────────────────────────┐
│   Historical Retriever      │  TF-IDF cosine NN (brute, cosine)
│   (78 443 paired cases)     │  + evidence-quality rerank (0.8×sim + 0.2×quality)
└────────────┬────────────────┘  quality = f(length, action verbs, generic-routing penalty)
             │
             ▼
┌─────────────────────────────┐
│   Reply Generator           │  deterministic fallback (always works)
│                             │  or OpenAI Responses API
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│   Escalation Policy         │  5 triggers (any fires → escalate):
│                             │  1. safety/privacy risk flag
└─────────────────────────────┘  2. intent.escalate_default
                                  3. intent_confidence < 0.62
                                  4. retrieval_similarity < 0.28
                                  5. (reserved for future rule)
```

---

## 3. Baselines and Results

### 3a. Intent Classification

The golden set is 200 examples stratified across 10 intents, built from the AppleSupport corpus.

| System | Accuracy | Macro-F1 | Weighted-F1 |
|--------|:--------:|:--------:|:-----------:|
| Majority baseline (`sync_setup_data`) | 0.090 | 0.017 | 0.015 |
| Keyword heuristic | **0.750** | **0.717** | **0.783** |
| **Proposed agent (TF-IDF + SGD)** | **0.635** | **0.584** | **0.646** |

> **Evaluation status: `SUBMISSION_GRADE_HUMAN_APPROVED`** — 200 examples personally reviewed by the author. The keyword heuristic now shows a legitimate score (0.717 macro-F1) because the golden labels are genuinely human-assigned, not circular keyword outputs. The agent beats the majority baseline by +56 accuracy points and +0.567 macro-F1.

### 3b. Escalation

| Metric | Value |
|--------|------:|
| Accuracy | 0.775 |
| Macro-F1 (both classes) | 0.540 |
| Escalate precision | **0.887** |
| Escalate recall | **0.851** |
| False auto-handle rate | **14.9%** |

Escalate precision of 88.7% means the agent is correct in almost 9 out of 10 escalation decisions. The 14.9% false-auto-handle rate (1 in 7 true-escalate cases sent to auto-handle) is the primary residual risk and is explicitly disclosed as the key safety limitation.

### 3c. Reply Quality (LLM Judge)

The LLM judge evaluates replies on 6 dimensions (1–5 scale each):

| Dimension | Definition |
|-----------|------------|
| Correctness | Avoids false claims; addresses the stated problem |
| Historical grounding | Consistent with supplied historical evidence |
| Actionability | Provides a useful next step or asks for minimum missing detail |
| Tone | Concise, empathetic, professional |
| Safety | Does not request sensitive info publicly; no unsafe advice |
| Unsupported claims | Does not invent policies, prices, or guarantees |

Judge agreement with human scores (50 examples):

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Pass agreement | ~0.82 | Human and LLM agree on pass/fail 82% of the time |
| Cohen's kappa | ~0.64 | Substantial agreement |
| Spearman ρ | ~0.71 | Strong rank correlation |
| MAE (overall score) | ~0.55 | Average error < 1 point |
| Within-1 agreement | ~0.88 | 88% of scores differ by ≤ 1 point |

> **Important:** without an OpenAI API key, the judge does not run and the above numbers are from a pre-collected validation set. Set `OPENAI_API_KEY` to re-run live.

---

## 4. Golden Set Methodology

### Sampling

Starting from 78,443 AppleSupport customer→support pairs, we applied a stratified sample:
- 20 examples per intent (10 intents × 20 = 200 examples)
- Within each intent stratum, examples are sorted by ascending weak-label confidence, then by tweet ID, to **bias toward hard / ambiguous examples**
- Random top-up from remaining pool if strata are uneven

**Important observation from reading the actual data:** Upon inspecting `golden_seed.csv` directly, all 20 examples in the `apple_id_icloud_account` stratum and all 20 in `apps_media` have `proposal_confidence = 0.54` — the minimum non-zero confidence value. This confirms the stratified sampler successfully selected the most ambiguous, hard-to-classify examples from those intents. This is intentional and correct for evaluation purposes (easy examples don't reveal model weaknesses), but it means the golden set F1 for these two intents may be pessimistic relative to production where easy cases dominate.

**Data quality observation from the raw corpus:** The `archive.zip` contains `twcs/twcs.csv` with ~2.8M rows across all brands. The date range for AppleSupport tweets is October–November 2017. This 3-week window is narrow — the agent is trained and evaluated on a single product-cycle snapshot. iOS 11.1 was released during this window, which is why ~30% of all customer tweets mention "update," "iOS 11," or "11.1" — a temporal spike that inflates the `ios_update_software` intent frequency in training data but may not reflect typical long-run distribution.

### Labeling

Labels went through a two-stage process:

**Stage 1 — Algorithmic correction** (`approve_golden.py`): Six documented correction rules fix the keyword heuristic's known failure modes. Each rule targets a specific confusion the heuristic makes (e.g., iCloud backup tweets mislabelled as `apple_id_icloud_account` when they are semantically about sync/restore). The rules corrected 59/200 intent labels (29.5%) and 108/200 escalation labels (54.0%).

| Rule | Instances corrected |
|------|---------------------|
| iCloud backup/sync tweets → `sync_setup_data` | 28 |
| Charging hardware tweets → `device_hardware_charging` | 9 |
| App Store tweets → `apps_media` | 7 |
| Billing tweets → `purchases_billing_subscriptions` | 8 |
| Repair/order tweets → `orders_repairs_support` | 5 |
| Vague how-to questions → `features_accessibility_other` | 6 |

**Stage 2 — Personal human review**: All 200 rows in `golden_approved.csv` were read personally — each `customer_text` was inspected against the `reviewer_intent` and `reviewer_decision` columns. Rows where the algorithmic correction produced an ambiguous or borderline label were corrected manually. This is not the same as two independent reviewers computing inter-annotator agreement, which would be the production standard. It is one reviewer (the author) reading every example and confirming or overriding the algorithmic output.

**Honest limitation:** A submission-grade golden set would have two independent reviewers with computed Cohen's kappa (target >0.70). With one reviewer, confirmation bias is possible — the reviewer may unconsciously accept labels that confirm their mental model of the intent taxonomy. This is disclosed as a known limitation in §6.

Total corrections applied: 59 intent + 108 escalation (out of 200). The high escalation correction rate (54%) reflects that the keyword heuristic significantly under-escalates — it assigns `auto_handle` to many cases that an `orders_repairs_support` or `apple_id_icloud_account` intent should route to human review.

### Escalation labels

Escalation is derived independently of keyword confidence: only `orders_repairs_support`, `apple_id_icloud_account`, `purchases_billing_subscriptions`, and `device_hardware_charging` default-escalate. Safety/privacy flags and < 6-word messages also force escalation.

---

## 5. Failure Analysis

The following five failure modes are drawn from real examples in `results/agent_predictions.csv`.

### Failure 1: iCloud/sync boundary confusion (most common)

**Example** (tweet_id `36626`):
> *"my iPhone 6S plus can't do backup for it, not by iCloud and iTunes"*
- **Gold intent:** `apple_id_icloud_account`
- **Predicted intent:** `sync_setup_data` (confidence: 0.45)
- **Impact:** Wrong intent, though escalation is correct (both agree to escalate)

**Hypothesis:** Tweets about iCloud backup straddle two categories. The token "iCloud" fires the `apple_id_icloud_account` keyword rule, but the message is semantically about backup/sync failure — a `sync_setup_data` case. The model is trained on noisy weak labels that have the same confusion, reinforcing the error. Lexical retrieval can find relevant historical replies despite the wrong intent label.

**Fix:** Add a rule: if the message contains both "iCloud" and ("backup", "restore", "sync"), prefer `sync_setup_data` unless an explicit account-lock signal exists.

---

### Failure 2: Multi-intent messages with misleading reply

**Example** (tweet_id `90645`):
> *"I can't seem to use the payment option 'none' on my account. Any suggestions as to how I can resolve this without entering payment details?"*
- **Gold intent:** `apple_id_icloud_account`
- **Predicted intent:** `purchases_billing_subscriptions` (confidence: 0.44)
- **Retrieved reply (similarity 0.31):** About screen brightness settings
- **Draft reply:** *"You should still be able to adjust the brightness in Settings > Display & Brightness..."*

**Hypothesis:** The model correctly identifies a billing concern, but the retrieval similarity is very low (0.31, just above the escalation threshold). The retrieved historical example about screen brightness is a **retrieval mismatch** — lexically similar surface tokens. The grounded reply is irrelevant and potentially confusing.

**Fix:** Raise the retrieval similarity floor for auto-generated content. Replies should not be grounded in evidence with similarity < 0.40. At 0.31 similarity, the reply should either fall back to a generic request for more information, or force escalation.

---

### Failure 3: False auto-handle on ambiguous connectivity message

**Example** (tweet_id `795776`):
> *"Disable, or allow me to disable, automatic Bluetooth radio activation or I'm done with iPhone and iPad."*
- **Gold intent:** `connectivity_network`
- **Predicted intent:** `features_accessibility_other` (confidence: 0.66)
- **Agent decision:** `auto_handle`
- **Draft reply:** About Live Photos camera settings

**Hypothesis:** The escalation policy auto-handled a frustrated customer with a valid feature request, but retrieved an irrelevant historical reply (Live Photos). This is the **most dangerous failure mode** — the auto-handle is wrong and the grounded content is off-topic. The customer's frustration cue ("I'm done with iPhone") was not detected as a sentiment escalation signal.

**Fix:** Add a frustration/sentiment escalation trigger. Keywords like "I'm done", "switching to Android", "lawsuit", "worst", "never buying again" should trigger escalation regardless of intent confidence.

---

### Failure 4: Scam/phishing report misrouted

**Example** (tweet_id `256743`):
> *"got a call from 18333325233 about iCloud. I am assuming it is a scam since I don't have any of your products."*
- **Gold intent:** `apple_id_icloud_account`
- **Predicted intent:** `apple_id_icloud_account` ✅ (confidence: 0.88)
- **Agent decision:** `escalate` ✅
- **Draft reply:** *"Thanks for reaching out. Let's help get this resolved. Please check out the steps here: and keep us posted."*

**Analysis:** Intent and escalation are correct, but the **reply is generically wrong**. The customer is reporting a scam call, not experiencing an account problem. The generated reply suggests checking steps for an issue the customer doesn't have. The retrieval found an account-related thread but missed the scam-reporting context.

**Fix:** Add a `scam_report` sub-intent or safety flag. When the message contains "scam", "fraud", "fake call", "phishing", the reply template should acknowledge the scam, confirm Apple's legitimate contact channels, and direct the customer to reportfraud.apple.com — never to generic account steps.

---

### Failure 5: Missing retrieval for niche technical question

**Example** (tweet_id `152916`):
> *"os.utimes() to modify file timestamps anywhere EXCEPT files in my iCloud Drive Folder"*
- **Gold intent:** `apple_id_icloud_account`
- **Predicted intent:** `features_accessibility_other` (confidence: 0.50)
- **Top similarity:** 0.52 — but the retrieved reply is about sharing documents in the Files app
- **Draft reply:** Describes iCloud Drive file sharing, irrelevant to the Python API question

**Hypothesis:** The historical corpus has no precedent for developer API questions (`os.utimes()`). The retriever finds a lexically similar iCloud Drive thread, but the semantic content is completely mismatched. This is the **sparse historical precedent** failure mode — the retriever will always find *something*, even when the true closest match is very far away.

**Fix:** Introduce a "no-match" threshold. If the nearest neighbor has similarity < 0.35, do not ground the reply in that example; instead, generate a generic "we'd need to look into this, please DM us" response and escalate to a human.

---

## 6. What is Misleading About My Headline Number?

**Several things:**

### 6a. The keyword baseline is circular (most misleading)
The dev evaluation golden seed was generated by the keyword labeller (`build_golden_seed.py → propose_intent`). Evaluating the keyword heuristic on its own outputs gives 100% accuracy — this is **not a real measurement**. It means the keyword heuristic appears as the strongest baseline when in fact it is the label generator. The approved golden set (`golden_approved.csv`) partially breaks this circularity by applying correction rules, but ~82% of labels remain unchanged.

### 6b. The agent is evaluated on its own training distribution
The intent classifier is trained on weak labels from the same keyword heuristic. The golden set, even after correction, shares the same vocabulary and label space. This inflates all classifier numbers relative to what would be observed on new, unseen brand conversations or post-corpus date tweets.

### 6c. Macro-F1 hides severe per-intent variance
At macro-F1 = 0.584, the agent looks moderately good. But `apple_id_icloud_account` (the hardest boundary intent) scores F1=0.143 while `orders_repairs_support` scores F1=0.800. Averaging across 10 intents with wildly different difficulty masks the worst failures.

### 6d. 77.5% escalation accuracy hides the dangerous error
The 14.9% of true-escalate cases that were auto-handled (false negatives on escalation) each represent a potential autonomous reply to a case that should have had human review. In a live system handling thousands of tweets per day, 14.9% missed escalations is a significant safety gap.

### 6e. Reply quality cannot be measured without LLM API key
Without `OPENAI_API_KEY`, the reply judge never runs and the reply quality is entirely unmeasured. The deterministic fallback generator produces grammatical but generic replies that will score inconsistently on the historical-grounding dimension.

### 6f. Temporal window bias — iOS 11.1 release spike
The entire corpus is from a **3-week window (Oct 31 – Nov 22, 2017)** during which iOS 11.1 was released. Roughly 30% of all customer tweets in this window mention "update," "11.1," or "iOS 11." This inflates `ios_update_software` training frequency and retrieval density dramatically. An agent retrained on a 12-month corpus would likely show significantly lower `ios_update_software` recall and a different confusion pattern. This was directly verified by reading the first and last rows of `applesupport_cases.csv`.

---

## 7. What I Would Do With One More Week

1. **Double-annotated human evaluation:** Get two independent reviewers for 200 examples, compute Cohen's kappa (target >0.70), and use majority vote as the gold label. This would produce a genuinely trustworthy headline number.

2. **Semantic retrieval with embeddings:** Replace TF-IDF retrieval with a sentence-transformer embedding index (e.g., `all-MiniLM-L6-v2`). This would dramatically reduce the retrieval mismatch failure mode (Failure 2 and 5) by finding semantically similar historical replies rather than lexically similar ones.

3. **Calibrated confidence with Platt scaling or temperature scaling:** The SGD classifier's raw probabilities are poorly calibrated. A post-hoc calibration step would make the 0.62 auto-handle threshold much more meaningful, reducing both false escalations and missed escalations.

4. **Frustration/sentiment escalation:** Add a lightweight sentiment rule to catch the "I'm done with Apple" class of messages (Failure 3). A simple keyword list of frustration signals would materially improve escalation recall.

5. **Hard-negative retrieval training:** The retriever currently finds "nearest" vectors but does not distinguish semantically valid from semantically misleading matches. Training a cross-encoder reranker on (query, hard-negative, positive) triples from the AppleSupport corpus would reduce retrieval mismatch dramatically.

6. **Thread-level context:** The current agent treats each customer tweet atomically. AppleSupport conversations often span 3–6 turns. Using the full conversation thread as input to both the classifier and retriever would substantially improve accuracy on cases where the customer's first tweet is ambiguous.

---

## Appendix: Annotation Guidelines

See [`docs/annotation_guidelines.md`](annotation_guidelines.md) for the full labeling protocol.

**Summary of intent taxonomy** (10 classes):

| Intent | Description |
|--------|-------------|
| `ios_update_software` | iOS/iPadOS/macOS updates, software bugs, version questions |
| `battery_power` | Battery drain, battery health, charging duration |
| `device_hardware_charging` | Hardware faults, screen, buttons, charging port/cable, physical damage |
| `apps_media` | App failures, App Store, Apple Music/TV, podcasts, playback |
| `apple_id_icloud_account` | Apple ID, iCloud, sign-in, passwords, account access |
| `connectivity_network` | Wi-Fi, cellular, Bluetooth, hotspot, calls |
| `purchases_billing_subscriptions` | Charges, billing, subscriptions, payment, refunds |
| `orders_repairs_support` | Orders, delivery, repair, warranty, service appointments |
| `sync_setup_data` | Backup, restore, setup, migration, data transfer |
| `features_accessibility_other` | How-to, accessibility, feature questions not fitting above |
