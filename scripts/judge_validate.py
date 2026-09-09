#!/usr/bin/env python
"""Validate LLM judge agreement against human labels.

Input CSV should contain columns:
  item_id, human_pass, human_overall, reply, customer_text, evidence
Human labels can be collected in any spreadsheet; this script performs
agreement/correlation checks and reports whether the judge is trustworthy.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

p=argparse.ArgumentParser(); p.add_argument('--csv', required=True); p.add_argument('--out', default='results/judge_agreement.json'); a=p.parse_args()
df=pd.read_csv(a.csv).fillna('')
req={'item_id','human_pass','human_overall','reply','customer_text','evidence'}
missing=req-set(df.columns)
if missing: raise SystemExit(f'missing columns: {sorted(missing)}')
if not os.getenv('OPENAI_API_KEY'): raise SystemExit('OPENAI_API_KEY required for judge validation')
from openai import OpenAI
client=OpenAI()
rows=[]
for _,r in df.iterrows():
    prompt=f"""Judge this support reply against the rubric. Return JSON only with overall (1-5), pass (0/1), brief_reason.
Customer: {r.customer_text}\nReply: {r.reply}\nEvidence: {r.evidence}
A passing reply must be correct, historically grounded, actionable, safe, and free of unsupported claims."""
    resp=client.responses.create(model=os.getenv('OPENAI_MODEL','gpt-5.6-luna'),input=prompt)
    import json as _json
    j=_json.loads(resp.output_text); j['item_id']=r.item_id; rows.append(j)
jud=pd.DataFrame(rows)
merged=df[['item_id','human_pass','human_overall']].merge(jud,on='item_id')
pass_agree=float((merged.human_pass.astype(int)==merged['pass'].astype(int)).mean())
cor=float(spearmanr(merged.human_overall.astype(float), merged.overall.astype(float)).statistic)
mae=float(np.mean(np.abs(merged.human_overall.astype(float)-merged.overall.astype(float))))
out={'n':len(merged),'pass_agreement':round(pass_agree,4),'spearman':round(cor,4),'mae':round(mae,4)}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
