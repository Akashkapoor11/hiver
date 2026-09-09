#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.data import discover_brands

p=argparse.ArgumentParser()
p.add_argument('--zip', required=True)
p.add_argument('--top', type=int, default=20)
a=p.parse_args()
print(discover_brands(a.zip, a.top).to_string(index=False))
