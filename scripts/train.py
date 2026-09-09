#!/usr/bin/env python
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.model import IntentModel

p=argparse.ArgumentParser()
p.add_argument('--cases', required=True)
p.add_argument('--out', default='data/processed/intent_model.joblib')
args=p.parse_args()
df=pd.read_csv(args.cases)
# Only train on weak labels from non-golden historical cases. Low-confidence cases are dropped.
df=df[df.weak_confidence>=0.58].dropna(subset=['customer_text','intent_proposed'])
model=IntentModel(seed=42).fit(df.customer_text.astype(str), df.intent_proposed.astype(str))
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
model.save(args.out)
print(f'trained on {len(df)} weakly-labelled historical cases -> {args.out}')
