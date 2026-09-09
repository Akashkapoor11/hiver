#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--predictions', default='results/agent_predictions.csv'); p.add_argument('--output', default='data/golden/judge_validation.csv'); p.add_argument('--n',type=int,default=50); a=p.parse_args()
df=pd.read_csv(a.predictions).head(a.n).copy()
out=pd.DataFrame({
 'item_id': df.customer_tweet_id.astype(str),
 'customer_text': df.customer_text,
 'reply': df.reply,
 'evidence': df.evidence_ids,
 'human_pass': '',
 'human_overall': '',
 'human_notes': ''
})
Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
print(f'Created {len(out)} judge-validation rows at {a.output}. Human must fill human_pass (0/1) and human_overall (1-5).')
