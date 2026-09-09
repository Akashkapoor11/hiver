from pathlib import Path
import os
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
GOLDEN_DIR = DATA_DIR / 'golden'
RESULTS_DIR = ROOT / 'results'

@dataclass(frozen=True)
class Settings:
    brand: str = os.getenv('HIVER_BRAND', 'AppleSupport')
    seed: int = int(os.getenv('HIVER_SEED', '42'))
    model_name: str = os.getenv('OPENAI_MODEL', 'gpt-5.6-luna')
    auto_threshold: float = float(os.getenv('HIVER_AUTO_THRESHOLD', '0.62'))
    retrieval_k: int = int(os.getenv('HIVER_RETRIEVAL_K', '5'))
    max_retrieval_corpus: int = int(os.getenv('HIVER_MAX_RETRIEVAL_CORPUS', '15000'))

settings = Settings()
