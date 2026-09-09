# This directory contains the golden evaluation set artefacts.
# golden_seed.csv        — 200-row stratified seed (committed)
# golden_approved.csv    — fully corrected + approved set (committed after running `make approve`)
# judge_validation.csv   — 50-example human+LLM scored set (committed)
#
# To regenerate golden_approved.csv from scratch:
#   make seed approve
