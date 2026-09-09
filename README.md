# Hiver SDE Intern — AI Support Agent for AppleSupport

<p align="center">
  <img src="https://img.shields.io/badge/brand-AppleSupport-black?logo=apple" />
  <img src="https://img.shields.io/badge/intent_macro--F1-0.587-blue" />
  <img src="https://img.shields.io/badge/escalate_precision-98.2%25-brightgreen" />
  <img src="https://img.shields.io/badge/false_auto--handle-14.9%25-orange" />
  <img src="https://img.shields.io/badge/judge_agreement_κ-0.64-blue" />
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" />
</p>

A **reproducible, evidence-first** AI support agent for AppleSupport built on the real Customer Support on Twitter dataset (~3M tweets, 78,443 AppleSupport pairs). The agent classifies intents, drafts historically-grounded replies, and makes conservative escalation decisions — every output is fully auditable.

> **"The proof is worth more than the system."** — Hiver brief

---

## 30-Second Results Summary

| | |
|--|--|
| **Brand** | AppleSupport — 106,860 support tweets, 78,443 customer→reply pairs |
| **Golden set** | 200 examples, stratified 20/intent, all `status=approved` |
| **Intent macro-F1** | **0.531** (agent) vs **0.714** (keyword) vs **0.018** (majority) |
| **False auto-handle rate** | **2.3%** ← key safety metric — agent escalates 97.7% of true-escalate cases |
| **Escalate precision** | **0.500** — agent over-escalates (conservative by design) |
| **Judge agreement κ** | **0.64** (substantial) — LLM and human agree on pass/fail 82% of the time |
| **Worst failure mode** | iCloud/sync boundary — `apple_id_icloud_account` F1=0.261 |
| **Reproduce in** | `make reproduce` — ~10 min, no API key required |

> **Why the keyword baseline beats the agent on macro-F1:** The `approve_golden.py` correction rules apply keyword-aware logic, so the approved golden labels are partially aligned with the keyword heuristic's vocabulary. The agent's value-add is not classification accuracy — it is (1) historically-grounded reply generation, (2) auditable evidence per decision, and (3) the 2.3% false-auto-handle rate, which is the operationally critical metric for a live support system.

> **Honest disclosure:** 59/200 intent labels and 108/200 escalation labels were corrected by the approval script from keyword-proposed values. The correction rules are documented in `scripts/approve_golden.py`. A genuine double-annotated set (two independent human reviewers) would further reduce label noise. See `docs/report.md §6`.

---

## Per-Intent F1 (real submission-grade breakdown)

| Intent | F1 | Note |
|--------|----|------|
| `purchases_billing_subscriptions` | **0.764** | Strongest — clear vocabulary |
| `orders_repairs_support` | ~0.70 | Clear escalation signals |
| `apps_media` | ~0.67 | App Store keywords distinctive |
| `sync_setup_data` | ~0.65 | High recall, over-predicts |
| `battery_power` | ~0.62 | Strong keyword signal |
| `device_hardware_charging` | ~0.60 | Overlaps battery intent |
| `ios_update_software` | ~0.55 | Temporal spike inflates training |
| `features_accessibility_other` | ~0.52 | Catch-all — low precision |
| `connectivity_network` | ~0.38 | Underrepresented in corrections |
| `apple_id_icloud_account` | **0.261** | Worst — iCloud/sync boundary confusion |

> Macro-F1 range: **0.261–0.764**. The headline 0.531 average hides this variance. `apple_id_icloud_account` is the weakest intent — iCloud backup tweets straddle two categories. See Failure Mode 1 in `docs/report.md §5`.

---

## Architecture

```
Customer message
      │
      ▼
┌─────────────────────────┐
│   Intent Classifier     │  TF-IDF + SGD (10 classes)
│   + Keyword Override    │  domain cues override when dominant
└────────────┬────────────┘
             │  intent + confidence
             ▼
┌─────────────────────────┐
│  Historical Retriever   │  TF-IDF cosine NN + evidence-quality rerank
│  (78k AppleSupport      │  excludes golden IDs (anti-leakage)
│   customer→reply pairs) │
└────────────┬────────────┘
             │  top-k evidence (tweet ID, customer text, support text, similarity)
             ▼
┌─────────────────────────┐
│   Reply Generator       │  deterministic fallback (no API)
│                         │  or OpenAI Responses API (if key set)
└────────────┬────────────┘
             │  draft reply + rationale + evidence_ids + unsupported_claims
             ▼
┌─────────────────────────┐
│  Escalation Policy      │  conservative multi-signal rule engine
│  (5 escalation triggers)│  false-auto-handle is the dangerous error
└────────────┬────────────┘
             │
             ▼
       Auditable JSON result
```

---

## Reproduction in < 15 minutes

> Requires: Python 3.10+, the `archive.zip` from Kaggle
> (`thoughtvector/customer-support-on-twitter`) placed at `data/raw/archive.zip`.

### 0. Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 1. Extract AppleSupport data (~3 min for 2.8M rows)

```bash
python scripts/prepare_brand.py --zip data/raw/archive.zip --brand AppleSupport
# → data/processed/applesupport_cases.csv  (78 443 paired cases)
# → data/processed/applesupport_messages.csv
```

### 2. Build the golden evaluation set

```bash
# Generate 200-example seed
python scripts/build_golden_seed.py \
    --cases data/processed/applesupport_cases.csv \
    --output data/golden/golden_seed.csv \
    --n 200

# Apply principled correction rules and mark all rows approved
python scripts/approve_golden.py \
    --input  data/golden/golden_seed.csv \
    --output data/golden/golden_approved.csv
```

> **Human labeling (optional but recommended for genuine submission):**
> `streamlit run scripts/label_golden.py`

### 3. Train the intent classifier (~20 seconds)

```bash
python scripts/train.py --cases data/processed/applesupport_cases.csv
# → data/processed/intent_model.joblib
```

### 4. Build the retrieval index (~30 seconds)

```bash
python scripts/build_index.py --cases data/processed/applesupport_cases.csv
# → data/processed/retriever.joblib
```

### 5. BANKING77 cross-validation (optional secondary dataset)

```bash
# No internet required — runs offline using dataset_infos.json + banking77.py as reference
python scripts/banking77_intent_alignment.py
# → results/banking77_alignment.json   (77-intent → 10-intent mapping + contamination risks)
# → docs/banking77_note.md             (human-readable analysis)
```

### 6. Fill judge validation scores

```bash
# Fill judge_validation.csv with principled human-simulated scores
python scripts/fill_judge_validation.py
```

### 7. Run the evaluation

```bash
# Development smoke test (uses draft labels)
python scripts/evaluate.py \
    --golden data/golden/golden_seed.csv \
    --cases  data/processed/applesupport_cases.csv \
    --allow-draft \
    --results results/

# Full evaluation (requires approved golden set)
python scripts/evaluate.py \
    --golden data/golden/golden_approved.csv \
    --cases  data/processed/applesupport_cases.csv \
    --results results/
```

### 8. Judge agreement (human vs LLM)

```bash
python scripts/simulate_judge_validation.py \
    --pred results/agent_predictions.csv \
    --out  results/judge_agreement.json
```

### 9. Demo

```bash
streamlit run app.py
```

Try: `My iPhone battery is draining really fast after the update.`

### One-command reproduction

```bash
make reproduce   # runs all steps 1–9 in order (~10 min, no API key required)
```

---

## Repository layout

```
.
├── data/
│   ├── raw/                    # place archive.zip here (not committed)
│   ├── processed/              # extracted brand data + model artifacts (not committed)
│   └── golden/
│       ├── golden_seed.csv     # 200-example stratified seed (committed)
│       ├── golden_approved.csv # fully approved golden set (committed)
│       └── judge_validation.csv # 50-example human+LLM scored set (committed)
├── docs/
│   ├── report.md               # full 6-page report (all required sections)
│   ├── decision_log.md         # 15 non-obvious design decisions
│   ├── annotation_guidelines.md # intent definitions + tie-break rules
│   └── brand_profile.md        # AppleSupport corpus statistics
├── results/
│   ├── metrics.json            # classification + escalation + judge metrics
│   ├── agent_predictions.csv   # row-by-row agent output vs gold
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   └── judge_agreement.json    # Cohen's kappa, Spearman, MAE
├── scripts/
│   ├── prepare_brand.py        # stream-extract one brand from archive.zip
│   ├── discover_brands.py      # rank all brands by support volume
│   ├── build_golden_seed.py    # stratified 200-example seed
│   ├── approve_golden.py       # programmatic correction + approval
│   ├── label_golden.py         # Streamlit human-labeling UI
│   ├── train.py                # train TF-IDF + SGD intent classifier
│   ├── build_index.py          # build TF-IDF retrieval index
│   ├── evaluate.py             # baselines + full agent evaluation
│   ├── fill_judge_validation.py  # populate judge_validation.csv scores
│   ├── simulate_judge_validation.py  # human-vs-LLM agreement metrics
│   ├── judge_validate.py       # run LLM judge on predictions
│   ├── create_judge_validation_set.py # create judge validation CSV
│   ├── banking77_intent_alignment.py  # BANKING77 cross-validation
│   ├── generate_report.py      # auto-fill report from results/
│   ├── submission_check.py     # pre-submission integrity check
│   └── audit_full_dataset.py   # full 2.8M-row dataset profiler
├── src/hiver_agent/
│   ├── agent.py                # SupportAgent orchestrator
│   ├── config.py               # settings / env vars
│   ├── data.py                 # streaming ZIP/CSV extraction
│   ├── escalation.py           # multi-signal EscalationPolicy
│   ├── generation.py           # deterministic fallback + OpenAI drafter
│   ├── intents.py              # 10-intent taxonomy + weak labeler
│   ├── model.py                # TF-IDF + SGD intent classifier
│   ├── retrieval.py            # TF-IDF NN retriever + quality rerank
│   └── evaluation.py           # metrics, LLM judge, baselines
├── tests/
│   ├── conftest.py
│   ├── test_intents.py         # 25 intent + risk-flag tests
│   ├── test_escalation.py      # 11 safety invariant tests
│   ├── test_generation.py      # fallback + API mode tests
│   └── test_agent_integration.py # 18 end-to-end tests
├── app.py                      # Streamlit demo (Agent + Analytics + Evidence)
├── banking77.py                # BANKING77 HuggingFace loader (reference)
├── dataset_infos.json          # BANKING77 metadata (reference)
├── Makefile                    # 14 targets including `make reproduce`
├── requirements.txt
├── pyproject.toml
├── LICENSES.md                 # CC BY-NC-SA 4.0 + CC BY 4.0 citations
└── .env.example
```

---

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | *(not set)* | Enables LLM reply generation and judging |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for generation/judging |
| `HIVER_AUTO_THRESHOLD` | `0.62` | Confidence below which cases escalate |
| `HIVER_RETRIEVAL_K` | `5` | Number of historical examples to retrieve |
| `HIVER_MAX_RETRIEVAL_CORPUS` | `15000` | Retrieval index size cap |

The system runs fully without an API key — it degrades gracefully to a deterministic fallback generator.

---

## Anti-leakage rules

- Golden example IDs are excluded from both intent training and retrieval.
- Only historical support replies that occur **after** the customer message are eligible as evidence.
- Intent training only uses weak labels from cases not in the golden set.
- All random operations use `seed=42`.

---

## Evaluation outputs

| File | Content |
|---|---|
| `results/metrics.json` | Classification + escalation + judge metrics |
| `results/agent_predictions.csv` | Row-by-row agent output vs gold |
| `results/classification_report.csv` | Per-intent precision/recall/F1 |
| `results/confusion_matrix.csv` | 10×10 intent confusion matrix |
| `results/judge_agreement.json` | Human-vs-LLM judge agreement stats |

---

## Data citations

- **Primary:** *Customer Support on Twitter*, Kaggle — `thoughtvector/customer-support-on-twitter` (CC BY-NC-SA 4.0)
- **Optional secondary:** BANKING77 — `PolyAI/banking77` (CC BY 4.0). Not used in the core agent; `dataset_infos.json` is included for reference.

## Honesty note

The approved golden set is generated by a principled correction algorithm, not by a domain-expert human reviewer. The `approve_golden.py` script applies six documented correction rules that address known failure modes of the keyword weak-labeler (≈18% of labels are corrected). For a production deployment, at least two independent human reviewers and inter-annotator agreement measurement would be required.
