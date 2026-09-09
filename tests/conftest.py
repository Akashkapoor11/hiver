"""pytest configuration — ensure src/ is on path for all test files."""
import sys
from pathlib import Path

# Add project root src/ to path so hiver_agent can be imported without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
