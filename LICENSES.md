# Data Licenses and Attribution

## Primary dataset

**Customer Support on Twitter**  
Source: Kaggle — `thoughtvector/customer-support-on-twitter`  
License: CC BY-NC-SA 4.0  
Used for: all training data, retrieval corpus, and golden evaluation set.

## Secondary dataset (taxonomy cross-validation only)

**BANKING77**  
Citation: Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", ACL 2020  
Source: `PolyAI/banking77` — https://github.com/PolyAI-LDN/task-specific-datasets  
License: Creative Commons Attribution 4.0 International (CC BY 4.0)  
Used for: offline cross-validation of the AppleSupport 10-intent taxonomy against the 77-intent banking domain taxonomy. **Not used for training, evaluation, or any runtime component.**  
Script: `scripts/banking77_intent_alignment.py`  
Output: `results/banking77_alignment.json`, `docs/banking77_note.md`  

The `banking77.py` loader and `dataset_infos.json` files in this repository are the HuggingFace Datasets loader and metadata for BANKING77, provided with the assignment for reference.
