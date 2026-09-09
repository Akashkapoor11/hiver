#!/usr/bin/env python
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.data import extract_brand

p=argparse.ArgumentParser()
p.add_argument('--zip', required=True)
p.add_argument('--brand', default='AppleSupport')
p.add_argument('--out', default='data/processed')
a=p.parse_args()
paths=extract_brand(a.zip,a.brand,a.out)
print(paths)
