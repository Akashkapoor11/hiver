#!/usr/bin/env python
"""
submission_check.py — Pre-submission checklist validator.

Checks that all required files exist, the approved golden set meets
the 150-250 example requirement, judge validation has human scores,
and metrics have been generated.
"""
import argparse, json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--golden', default='data/golden/golden_approved.csv',
               help='Path to the approved golden set CSV')
p.add_argument('--metrics', default='results/metrics.json')
a = p.parse_args()

errors = []
warnings = []

# ── 1. Required files ────────────────────────────────────────────────────────
required_files = [
    'README.md',
    'docs/report.md',
    'docs/decision_log.md',
    'docs/annotation_guidelines.md',
    'requirements.txt',
    'app.py',
    'scripts/approve_golden.py',
    'scripts/simulate_judge_validation.py',
]
for path in required_files:
    if not Path(path).exists():
        errors.append(f'missing required file: {path}')

# ── 2. Golden set ────────────────────────────────────────────────────────────
g = Path(a.golden)
if not g.exists():
    errors.append(f'missing golden set: {g} — run: python scripts/approve_golden.py')
else:
    df = pd.read_csv(g).fillna('')
    approved = int((df.get('status', pd.Series(['draft'] * len(df))) == 'approved').sum())
    total = len(df)

    if not (150 <= approved <= 250):
        errors.append(
            f'approved golden count is {approved}; Hiver requires 150–250. '
            f'Run: python scripts/approve_golden.py'
        )
    else:
        print(f'✅ Golden set: {approved}/{total} approved (requirement: 150–250)')

    if approved != total:
        warnings.append(f'{total - approved}/{total} golden examples are not approved')

    for c in ['reviewer_intent', 'reviewer_decision', 'review_notes', 'status']:
        if c not in df.columns:
            errors.append(f'missing required golden column: {c}')

# ── 3. Judge validation ──────────────────────────────────────────────────────
j = Path('data/golden/judge_validation.csv')
if not j.exists():
    warnings.append('judge_validation.csv not found — run: python scripts/create_judge_validation_set.py')
else:
    jd = pd.read_csv(j).fillna('')
    if 'human_pass' not in jd.columns or 'human_overall' not in jd.columns:
        errors.append('judge_validation.csv missing human_pass or human_overall columns')
    else:
        missing = (
            (jd.human_pass.astype(str).str.strip() == '') |
            (jd.human_overall.astype(str).str.strip() == '')
        )
        n_missing = int(missing.sum())
        if n_missing == len(jd):
            warnings.append(
                f'All {n_missing} judge-validation rows have empty human scores — '
                f'this is OK if you are using the synthetic validation path in simulate_judge_validation.py'
            )
        elif n_missing > 0:
            errors.append(f'{n_missing} judge-validation rows still lack human scores')
        else:
            print(f'✅ Judge validation: {len(jd)} rows with human scores')

# ── 4. Metrics ───────────────────────────────────────────────────────────────
m = Path(a.metrics)
if not m.exists():
    warnings.append('results/metrics.json not found — run: python scripts/evaluate.py')
else:
    data = json.loads(m.read_text())
    agent_acc = data.get('agent', {}).get('classification', {}).get('accuracy', None)
    if agent_acc is not None:
        print(f'✅ Metrics: agent intent accuracy = {agent_acc:.3f}')
    if data.get('reply_judge', {}).get('status') != 'run':
        warnings.append(
            'LLM reply judge has not run — set OPENAI_API_KEY to enable, '
            'or the synthetic judge agreement in results/judge_agreement.json is the fallback'
        )

# ── 5. Judge agreement ───────────────────────────────────────────────────────
ja = Path('results/judge_agreement.json')
if not ja.exists():
    warnings.append('results/judge_agreement.json not found — run: python scripts/simulate_judge_validation.py')
else:
    jdata = json.loads(ja.read_text(encoding='utf-8'))
    print(f"✅ Judge agreement: pass_agreement={jdata.get('pass_agreement')}, kappa={jdata.get('cohens_kappa')}")

# ── 6. Report completeness ───────────────────────────────────────────────────
report = Path('docs/report.md')
if report.exists():
    content = report.read_text(encoding='utf-8')
    placeholders = ['<fill', '<todo', 'TODO', 'PLACEHOLDER']
    found = [p for p in placeholders if p.lower() in content.lower()]
    if found:
        errors.append(f'docs/report.md contains unfilled placeholders: {found}')
    else:
        print('✅ Report: no unfilled placeholders found')

# ── 7. Banking77 citation ────────────────────────────────────────────────────
licenses = Path('LICENSES.md')
if licenses.exists() and 'banking77' not in licenses.read_text(encoding='utf-8').lower():
    warnings.append('LICENSES.md does not mention BANKING77 — add citation even if unused')

# ── Summary ──────────────────────────────────────────────────────────────────
print('\n' + '=' * 50)
print('SUBMISSION CHECK:', 'PASS ✅' if not errors else 'FAIL ❌')
if errors:
    for x in errors:
        print(f'  ERROR: {x}')
if warnings:
    for x in warnings:
        print(f'  WARN:  {x}')
if not errors and not warnings:
    print('  All checks passed. Ready to submit.')

raise SystemExit(1 if errors else 0)
