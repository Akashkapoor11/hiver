ZIP=data/raw/archive.zip
BRAND=AppleSupport
CASES=data/processed/applesupport_cases.csv
GOLDEN_SEED=data/golden/golden_seed.csv
GOLDEN_APPROVED=data/golden/golden_approved.csv

prepare:
	python scripts/prepare_brand.py --zip $(ZIP) --brand $(BRAND)

seed:
	python scripts/build_golden_seed.py --cases $(CASES) --output $(GOLDEN_SEED) --n 200

approve:
	python scripts/approve_golden.py --input $(GOLDEN_SEED) --output $(GOLDEN_APPROVED)

train:
	python scripts/train.py --cases $(CASES)

index:
	python scripts/build_index.py --cases $(CASES)

smoke:
	python scripts/evaluate.py --golden $(GOLDEN_SEED) --cases $(CASES) --allow-draft --results results/

eval:
	python scripts/evaluate.py --golden $(GOLDEN_APPROVED) --cases $(CASES) --results results/

fill-judge:
	python scripts/fill_judge_validation.py

judge-validate:
	python scripts/simulate_judge_validation.py --pred results/agent_predictions.csv --out results/judge_agreement.json

banking77:
	python scripts/banking77_intent_alignment.py

report:
	python scripts/generate_report.py

check:
	python scripts/submission_check.py --golden data/golden/golden_approved.csv

app:
	streamlit run app.py

label:
	streamlit run scripts/label_golden.py

test:
	pytest tests/ -v

# Full reproduction pipeline (no LLM required, runs in ~10 min)
reproduce: prepare seed approve train index smoke eval fill-judge judge-validate banking77 report check
	@echo "✅ Full reproduction complete. See results/ for metrics."
