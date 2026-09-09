#!/usr/bin/env python
import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.intents import propose_intent, INTENTS, risk_flags

p=argparse.ArgumentParser()
p.add_argument('--cases', required=True)
p.add_argument('--output', default='data/golden/golden_seed.csv')
p.add_argument('--n', type=int, default=200)
a=p.parse_args()
df=pd.read_csv(a.cases)
df=df[df.customer_text.fillna('').str.len()>=10].copy()
df['intent'], df['proposal_confidence'] = zip(*df.customer_text.map(propose_intent))
df['expected_decision'] = df.apply(lambda r: 'escalate' if risk_flags(r.customer_text) or r.proposal_confidence < 0.62 or INTENTS[r.intent]['escalate_default'] else 'auto_handle', axis=1)
# Stratified sample by proposed intent; prioritize hard/low-confidence examples inside each stratum.
parts=[]
per=max(1,a.n//len(INTENTS))
for intent,g in df.groupby('intent'):
    g=g.sort_values(['proposal_confidence','customer_tweet_id'])
    parts.append(g.head(per))
out=pd.concat(parts, ignore_index=True)
if len(out)<a.n:
    rem=df[~df.customer_tweet_id.isin(out.customer_tweet_id)].sample(min(a.n-len(out),len(df)-len(out)),random_state=42)
    out=pd.concat([out,rem],ignore_index=True)
out=out.head(a.n)
out['status']='draft'
out['reviewer_intent']=out['intent']
out['reviewer_decision']=out['expected_decision']
out['review_notes']=''
cols=['customer_tweet_id','support_tweet_id','customer_text','support_text','intent','proposal_confidence','expected_decision','status','reviewer_intent','reviewer_decision','review_notes']
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
out[cols].to_csv(a.output,index=False)
print(f'wrote {len(out)} draft golden examples to {a.output}')
