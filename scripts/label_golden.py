#!/usr/bin/env python
"""
label_golden.py — Human annotation UI for the Hiver golden set.

Rules:
- NEVER auto-approves any row.
- Only rows clicked through 'Approve & Next' or 'Approve & Stay' count as approved.
- Saves progress to a separate state file so the app can resume after restart.
- golden_approved.csv contains ONLY rows the human has personally approved.
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.intents import INTENTS

# ── Paths ────────────────────────────────────────────────────────────────────
SEED_PATH     = Path('data/golden/golden_seed.csv')
APPROVED_PATH = Path('data/golden/golden_approved.csv')
STATE_PATH    = Path('data/golden/.labeler_state.json')   # remembers last position

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Hiver — Golden Set Labeler',
    page_icon='🏷️',
    layout='wide',
)

# ── Load seed ────────────────────────────────────────────────────────────────
if not SEED_PATH.exists():
    st.error('❌ Seed file not found. Run: python scripts/build_golden_seed.py first.')
    st.stop()

@st.cache_data(show_spinner=False)
def load_seed():
    df = pd.read_csv(SEED_PATH).fillna('')
    # Ensure all required columns exist
    if 'status'            not in df.columns: df['status']            = 'draft'
    if 'reviewer_intent'   not in df.columns: df['reviewer_intent']   = df.get('intent', '')
    if 'reviewer_decision' not in df.columns: df['reviewer_decision'] = df.get('expected_decision', '')
    if 'review_notes'      not in df.columns: df['review_notes']      = ''
    return df

# We load fresh each time (not cached) so edits show up immediately
df = pd.read_csv(SEED_PATH).fillna('')
if 'status'            not in df.columns: df['status']            = 'draft'
if 'reviewer_intent'   not in df.columns: df['reviewer_intent']   = df.get('intent', '')
if 'reviewer_decision' not in df.columns: df['reviewer_decision'] = df.get('expected_decision', '')
if 'review_notes'      not in df.columns: df['review_notes']      = ''

total = len(df)
approved_mask = df['status'] == 'approved'
n_approved = int(approved_mask.sum())

# ── Restore last position ────────────────────────────────────────────────────
def load_state() -> int:
    if STATE_PATH.exists():
        try:
            return int(json.loads(STATE_PATH.read_text(encoding='utf-8')).get('last_idx', 0))
        except Exception:
            pass
    return 0

def save_state(idx: int):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({'last_idx': idx}), encoding='utf-8')

if 'idx' not in st.session_state:
    st.session_state.idx = load_state()

idx = int(st.session_state.idx)
idx = max(0, min(idx, total - 1))

# ── Helper: persist changes ───────────────────────────────────────────────────
def save_row(idx, intent, decision, notes, approve: bool):
    df.loc[idx, 'reviewer_intent']   = intent
    df.loc[idx, 'reviewer_decision'] = decision
    df.loc[idx, 'review_notes']      = notes
    if approve:
        df.loc[idx, 'status'] = 'approved'
    df.to_csv(SEED_PATH, index=False)
    # Write ONLY approved rows to golden_approved.csv
    approved_df = df[df['status'] == 'approved'].copy()
    APPROVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    approved_df.to_csv(APPROVED_PATH, index=False)
    st.cache_data.clear()

# ── Header ───────────────────────────────────────────────────────────────────
st.title('🏷️ Hiver Golden Set Labeler')
st.caption(
    'Read each customer tweet carefully. The AI-proposed values are pre-selected. '
    'Correct them if needed, then click **Approve & Next**. '
    'Only rows YOU approve through this UI will count as human-labelled.'
)

# ── Progress bar ─────────────────────────────────────────────────────────────
n_approved = int((df['status'] == 'approved').sum())
progress = n_approved / total

col_prog1, col_prog2 = st.columns([4, 1])
with col_prog1:
    st.progress(progress, text=f'Progress: {n_approved} / {total} approved')
with col_prog2:
    pct = f'{100*progress:.1f}%'
    if n_approved == total:
        st.success(f'✅ Complete! {pct}')
    else:
        st.metric('Approved', f'{n_approved}/{total}')

# ── Completion banner ─────────────────────────────────────────────────────────
if n_approved == total:
    st.balloons()
    st.success(
        '🎉 **All 200 examples reviewed and approved!**\n\n'
        'Your golden set is complete. Run the evaluation:\n'
        '```\npython scripts/evaluate.py '
        '--golden data/golden/golden_approved.csv '
        '--cases data/processed/applesupport_cases.csv '
        '--results results/\n```'
    )
    st.stop()

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
nav_cols = st.columns([1, 6, 1])

with nav_cols[0]:
    if st.button('◀ Prev', disabled=(idx == 0)):
        st.session_state.idx = idx - 1
        save_state(idx - 1)
        st.rerun()

with nav_cols[1]:
    # Jump to any row — pre-fill with current position
    new_idx = st.number_input(
        f'Jump to row (0–{total-1})',
        min_value=0, max_value=total-1,
        value=idx, step=1, label_visibility='collapsed'
    )
    if new_idx != idx:
        st.session_state.idx = int(new_idx)
        save_state(int(new_idx))
        st.rerun()

with nav_cols[2]:
    if st.button('Next ▶', disabled=(idx == total - 1)):
        st.session_state.idx = idx + 1
        save_state(idx + 1)
        st.rerun()

# ── Find next unapproved ─────────────────────────────────────────────────────
unapproved_indices = df[df['status'] != 'approved'].index.tolist()
if unapproved_indices:
    next_unapproved = min(unapproved_indices, key=lambda i: abs(i - idx))
    if st.button(f'⏭ Jump to next unapproved (row {next_unapproved})'):
        st.session_state.idx = next_unapproved
        save_state(next_unapproved)
        st.rerun()

# ── Current row ──────────────────────────────────────────────────────────────
r = df.iloc[idx]
row_status = r.get('status', 'draft')

st.divider()
status_badge = '✅ APPROVED' if row_status == 'approved' else '⏳ DRAFT'
st.subheader(f'Row {idx + 1} of {total}   {status_badge}')

# ── Content columns ──────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown('### 💬 Customer message')
    st.info(str(r.get('customer_text', '')))

    st.markdown('### 📩 Historical AppleSupport reply')
    st.success(str(r.get('support_text', '')))

with right:
    st.markdown('### 🏷️ Intent label')
    intent_keys = list(INTENTS.keys())
    current_intent = str(r.get('reviewer_intent', r.get('intent', intent_keys[0])))
    if current_intent not in intent_keys:
        current_intent = intent_keys[0]

    intent = st.selectbox(
        'Gold intent',
        options=intent_keys,
        index=intent_keys.index(current_intent),
        format_func=lambda x: f'{x}  —  {INTENTS[x]["description"]}',
        key=f'intent_{idx}'
    )

    st.markdown('### ⚡ Escalation decision')
    current_decision = str(r.get('reviewer_decision', r.get('expected_decision', 'auto_handle')))
    decision = st.radio(
        'Gold escalation decision',
        options=['auto_handle', 'escalate'],
        index=0 if current_decision == 'auto_handle' else 1,
        horizontal=True,
        key=f'decision_{idx}'
    )

    st.markdown('### 📝 Reviewer notes (optional)')
    notes = st.text_area(
        'Why did you choose this label? Any ambiguity?',
        value=str(r.get('review_notes', '')),
        height=80,
        key=f'notes_{idx}',
        placeholder='e.g. "Borderline iCloud/sync — chose sync because backup is the core issue"'
    )

# ── Diff highlight ───────────────────────────────────────────────────────────
proposed_intent    = str(r.get('intent', ''))
proposed_decision  = str(r.get('expected_decision', ''))

if intent != proposed_intent:
    st.warning(f'⚠️ You changed intent: `{proposed_intent}` → `{intent}`')
if decision != proposed_decision:
    st.warning(f'⚠️ You changed decision: `{proposed_decision}` → `{decision}`')

# ── Action buttons ────────────────────────────────────────────────────────────
st.divider()
btn1, btn2, btn3, btn4 = st.columns(4)

with btn1:
    if st.button('✅ Approve & Next', type='primary', use_container_width=True):
        save_row(idx, intent, decision, notes, approve=True)
        next_idx = min(idx + 1, total - 1)
        st.session_state.idx = next_idx
        save_state(next_idx)
        st.rerun()

with btn2:
    if st.button('✅ Approve & Stay', use_container_width=True):
        save_row(idx, intent, decision, notes, approve=True)
        save_state(idx)
        st.rerun()

with btn3:
    if st.button('💾 Save draft (no approval)', use_container_width=True):
        save_row(idx, intent, decision, notes, approve=False)
        st.info('Saved as draft — not counted as approved.')
        save_state(idx)

with btn4:
    if st.button('⏭ Skip (no save)', use_container_width=True):
        next_idx = min(idx + 1, total - 1)
        st.session_state.idx = next_idx
        save_state(next_idx)
        st.rerun()

# ── Sidebar: summary ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header('📊 Session Summary')
    st.metric('Total rows', total)
    st.metric('Approved', n_approved)
    st.metric('Remaining', total - n_approved)
    st.metric('Current row', idx + 1)

    st.divider()
    st.markdown('**Intent distribution (approved)**')
    if n_approved > 0:
        approved_df = df[df['status'] == 'approved']
        counts = approved_df['reviewer_intent'].value_counts()
        for intent_name, count in counts.items():
            st.markdown(f'`{intent_name}`: {count}')

    st.divider()
    st.caption(
        'golden_approved.csv contains ONLY rows you personally approved. '
        'No auto-approval, no simulation.'
    )
