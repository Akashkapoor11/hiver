"""
Full dataset audit script — processes twcs.csv in chunks.
No full load into memory. ~516 MB / ~2.8M rows.
"""
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

CSV = 'data/raw/twcs/twcs.csv'
CHUNK = 100_000

brand_support_count   = defaultdict(int)   # outbound tweets per brand
brand_inbound_count   = defaultdict(int)   # inbound tweets per brand (customers mentioning brand)
total_rows            = 0
inbound_count         = 0
outbound_count        = 0
null_text             = 0
null_author           = 0
date_min              = None
date_max              = None
sample_rows           = []                 # first 5 rows
sample_brand_rows     = {}                 # first 3 rows per top brand

print("Scanning full dataset in chunks...")

for i, chunk in enumerate(pd.read_csv(CSV, chunksize=CHUNK, dtype=str, low_memory=False)):
    total_rows += len(chunk)

    # Track nulls
    null_text   += chunk['text'].isna().sum()
    null_author += chunk['author_id'].isna().sum()

    # Split inbound vs outbound
    inbound_mask  = chunk['inbound'].str.strip().str.lower() == 'true'
    outbound_mask = ~inbound_mask

    inbound_count  += inbound_mask.sum()
    outbound_count += outbound_mask.sum()

    # Count support tweets per brand (outbound = brand responding)
    for brand, cnt in chunk.loc[outbound_mask, 'author_id'].value_counts().items():
        brand_support_count[brand] += int(cnt)

    # Collect first 5 rows
    if i == 0:
        sample_rows = chunk.head(5).to_dict(orient='records')

print(f"Done. Total rows: {total_rows:,}")
print(f"  Inbound  (customers): {inbound_count:,}")
print(f"  Outbound (brands):    {outbound_count:,}")
print(f"  Null text:            {null_text:,}")
print(f"  Null author_id:       {null_author:,}")

# Top 25 brands by support tweet volume
top25 = sorted(brand_support_count.items(), key=lambda x: x[1], reverse=True)[:25]
print("\nTop 25 brands by support tweet volume:")
print(f"{'Rank':<5} {'Brand':<35} {'Support tweets':>15}")
print("-" * 58)
for rank, (brand, cnt) in enumerate(top25, 1):
    print(f"{rank:<5} {brand:<35} {cnt:>15,}")

# Save full brand table
out = {
    'total_rows': total_rows,
    'inbound_count': inbound_count,
    'outbound_count': outbound_count,
    'null_text': int(null_text),
    'null_author': int(null_author),
    'top25_brands': [{'rank': r+1, 'brand': b, 'support_tweets': c} for r, (b, c) in enumerate(top25)],
    'sample_rows': sample_rows,
}
Path('results').mkdir(exist_ok=True)
Path('results/full_dataset_audit.json').write_text(json.dumps(out, indent=2, default=str))
print("\nFull audit written to results/full_dataset_audit.json")
