#!/usr/bin/env python
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.retrieval import HistoricalRetriever
p=argparse.ArgumentParser(); p.add_argument('--cases',required=True); p.add_argument('--out',default='data/processed/retriever.joblib'); p.add_argument('--max-docs',type=int,default=15000); a=p.parse_args()
df=pd.read_csv(a.cases).fillna(''); r=HistoricalRetriever(max_docs=a.max_docs).fit(df); Path(a.out).parent.mkdir(parents=True,exist_ok=True); r.save(a.out); print(f'indexed {len(r.cases)} cases -> {a.out}')
