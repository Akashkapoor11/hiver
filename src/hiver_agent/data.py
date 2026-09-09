from __future__ import annotations
import csv, io, os, zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Dict, List
import pandas as pd
from .intents import propose_intent

FIELDS = ['tweet_id','author_id','inbound','created_at','text','response_tweet_id','in_response_to_tweet_id']

def iter_csv_from_zip(zip_path: str | Path, member: str='twcs/twcs.csv') -> Iterable[dict]:
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
        wrapper = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
        try:
            for row in csv.DictReader(wrapper):
                yield row
        finally:
            wrapper.detach()

def discover_brands(zip_path: str | Path, top: int=20) -> pd.DataFrame:
    counts = defaultdict(int)
    for row in iter_csv_from_zip(zip_path):
        if row.get('inbound') == 'False' and row.get('author_id'):
            counts[row['author_id']] += 1
    df = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top], columns=['brand','support_tweets'])
    return df

def extract_brand(zip_path: str | Path, brand: str, out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # Two streaming passes avoid loading the 2.8M-row source into memory while
    # still capturing customer replies that do not explicitly mention the brand.
    support_ids = set()
    support_rows = []
    for row in iter_csv_from_zip(zip_path):
        if row.get('author_id') == brand:
            support_ids.add(str(row.get('tweet_id','')))
            support_rows.append(row)

    raw_rows = list(support_rows)
    kept_customer_ids = set()
    for row in iter_csv_from_zip(zip_path):
        if row.get('inbound') != 'True':
            continue
        response_ids = {x.strip() for x in str(row.get('response_tweet_id','')).split(',') if x.strip()}
        parent_id = str(row.get('in_response_to_tweet_id','')).strip()
        mentions_brand = brand.lower() in (row.get('text') or '').lower()
        direct_to_brand = bool(response_ids & support_ids) or parent_id in support_ids
        if mentions_brand or direct_to_brand:
            tid = str(row.get('tweet_id',''))
            if tid not in kept_customer_ids:
                raw_rows.append(row); kept_customer_ids.add(tid)

    msg_path = out / f'{brand.lower()}_messages.csv'
    pd.DataFrame(raw_rows, columns=FIELDS).to_csv(msg_path, index=False)

    by_id = {str(r['tweet_id']): r for r in raw_rows}
    cases = []
    seen = set()
    for r in support_rows:
        parent_id = str(r.get('in_response_to_tweet_id','')).split(',')[0].strip()
        parent = by_id.get(parent_id)
        if not parent or parent.get('inbound') != 'True':
            continue
        key = (parent_id, str(r['tweet_id']))
        if key in seen:
            continue
        seen.add(key)
        intent, weak_conf = propose_intent(parent.get('text',''))
        cases.append({
            'customer_tweet_id': parent_id,
            'support_tweet_id': str(r['tweet_id']),
            'customer_created_at': parent.get('created_at',''),
            'support_created_at': r.get('created_at',''),
            'customer_text': parent.get('text',''),
            'support_text': r.get('text',''),
            'intent_proposed': intent,
            'weak_confidence': weak_conf,
        })
    cases_df = pd.DataFrame(cases)
    cases_path = out / f'{brand.lower()}_cases.csv'
    cases_df.to_csv(cases_path, index=False)
    stats = {
        'brand': brand,
        'raw_rows': len(raw_rows),
        'customer_rows': sum(r.get('inbound')=='True' for r in raw_rows),
        'support_rows': len(support_rows),
        'paired_cases': len(cases_df),
    }
    import json
    stats_path = out / f'{brand.lower()}_stats.json'
    stats_path.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return {'messages': str(msg_path), 'cases': str(cases_path), 'stats': str(stats_path)}
