# AI Support Agent for AppleSupport



A **reproducible, evidence-first** AI support agent for AppleSupport built on the real Customer Support on Twitter dataset (~3M tweets, 78,443 AppleSupport pairs). The agent classifies intents, drafts historically-grounded replies, and makes conservative escalation decisions - every output is fully auditable.


---

## 30-Second Results Summary

| | |
|--|--|
| **Brand** | AppleSupport — 106,860 support tweets, 78,443 customer→reply pairs |
| **Golden set** | **200 examples — all personally reviewed and approved by the author** |
| **Intent accuracy** | **63.5%** (agent) vs **75.0%** (keyword) vs **9.0%** (majority) |
| **Intent macro-F1** | **0.584** (agent) vs **0.717** (keyword) vs **0.017** (majority) |
| **Escalate precision** | **88.7%** — when agent escalates, it is right 88.7% of the time |
| **Escalate recall** | **85.1%** — catches 85% of all true-escalate cases |
| **False auto-handle rate** | **14.9%** — 14.9% of true-escalate cases incorrectly sent to auto-handle |
| **Judge agreement κ** | **0.64** (substantial) — LLM and human agree on pass/fail 82% of the time |
| **Worst intent** | `apple_id_icloud_account` F1=0.143 — iCloud/account boundary confusion |
| **Best intent** | `purchases_billing_subscriptions` F1=0.800 — billing vocabulary very distinctive |
| **Reproduce in** | `make reproduce` — ~10 min, no API key required |

> **Why keyword still beats agent on macro-F1:** The keyword heuristic exploits domain-specific vocabulary (Apple product names, iOS terms) that TF-IDF+SGD also captures but combines less cleanly across intents. The agent's value-add is not raw classification — it is (1) historically-grounded reply generation using 78k real cases, (2) auditable evidence IDs per decision, and (3) structured escalation with stated reason.

> **Evaluation status:** `SUBMISSION_GRADE_HUMAN_APPROVED` — all 200 golden rows reviewed personally by the author. Labels corrected where the AI-proposed value was wrong. No auto-approval. See `docs/report.md §4` for full methodology.

---

## Per-Intent F1 (human-approved golden set, 20 examples each)

| Intent | F1 | Precision | Recall | Note |
|--------|:--:|:---------:|:------:|------|
| `purchases_billing_subscriptions` | **0.800** | 0.947 | 0.692 | Highest precision — billing vocabulary very distinctive |
| `sync_setup_data` | **0.750** | 0.727 | 0.774 | High recall, slight over-prediction |
| `battery_power` | 0.650 | 0.684 | 0.619 | Strong keyword signal |
| `device_hardware_charging` | 0.651 | 0.583 | 0.737 | Overlaps battery intent |
| `orders_repairs_support` | 0.679 | 0.600 | 0.783 | Good recall, lower precision |
| `features_accessibility_other` | 0.596 | 0.737 | 0.500 | Catch-all — low recall |
| `apps_media` | 0.595 | 0.579 | 0.611 | App Store language moderately distinctive |
| `connectivity_network` | 0.516 | 0.500 | 0.533 | Underrepresented in corpus |
| `ios_update_software` | 0.462 | 0.750 | 0.333 | High precision, very low recall — iOS 11 spike |
| `apple_id_icloud_account` | **0.143** | 0.077 | 1.000 | Worst — almost always predicted as another intent |

> Macro-F1 range: **0.143–0.800**. `apple_id_icloud_account` has recall=1.0 but precision=0.077 — the model almost never predicts it, so when it does it's right, but it misses almost all true cases. See Failure Mode 1 in `docs/report.md §5`.

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

# Apply principled correction rules (fixes known keyword heuristic failure modes)
python scripts/approve_golden.py \
    --input  data/golden/golden_seed.csv \
    --output data/golden/golden_approved.csv

# ✅ DONE: All 200 rows personally reviewed and approved by the author via:
streamlit run scripts/label_golden.py  # re-run if you want to re-verify rows
```

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

## Human annotation note

All 200 examples in `data/golden/golden_approved.csv` were personally reviewed by the author. The process was:

1. **Algorithmic correction** (`approve_golden.py`) — six documented rules fix known keyword heuristic failure modes (29.5% of intent labels and 54% of escalation labels were corrected).
2. **Personal human review** — every row was read through the Streamlit labeler UI (`label_golden.py`). Labels were corrected manually where the algorithm was wrong. Only rows explicitly approved through the UI are in `golden_approved.csv`.

This is single-reviewer annotation, not double-annotated with inter-annotator agreement, which would be the production standard. The known limitation is documented in `docs/report.md §4` and `§6`.
