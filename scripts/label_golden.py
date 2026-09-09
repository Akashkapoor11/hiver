#!/usr/bin/env python
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hiver_agent.intents import INTENTS

PATH=Path('data/golden/golden_seed.csv')
APPROVED=Path('data/golden/golden_approved.csv')
if not PATH.exists():
    st.error('Run build_golden_seed.py first.')
    st.stop()
df=pd.read_csv(PATH).fillna('')
if 'status' not in df: df['status']='draft'
if 'reviewer_intent' not in df: df['reviewer_intent']=df['intent']
if 'reviewer_decision' not in df: df['reviewer_decision']=df['expected_decision']

st.title('Hiver Golden Set Labeler')
st.caption('The seed labels are assistant-proposed. Approve/correct each example; approval is the human evidence required by the assignment.')
idx=st.number_input('Example index', min_value=0, max_value=max(0,len(df)-1), value=0, step=1)
r=df.iloc[int(idx)]
st.markdown(f"**Customer**\n\n{r.customer_text}")
st.markdown(f"**Historical support reply**\n\n{r.support_text}")
intent=st.selectbox('Gold intent', list(INTENTS.keys()), index=list(INTENTS.keys()).index(r.reviewer_intent) if r.reviewer_intent in INTENTS else 0, format_func=lambda x: f"{x} — {INTENTS[x]['description']}")
decision=st.radio('Gold escalation decision', ['auto_handle','escalate'], index=0 if r.reviewer_decision=='auto_handle' else 1)
notes=st.text_area('Reviewer notes', value=r.review_notes, height=100)
col1,col2,col3=st.columns(3)
with col1:
    if st.button('Approve & save'):
        df.loc[int(idx),'reviewer_intent']=intent; df.loc[int(idx),'reviewer_decision']=decision; df.loc[int(idx),'review_notes']=notes; df.loc[int(idx),'status']='approved'
        df.to_csv(PATH,index=False)
        approved=df[df.status=='approved'].copy(); approved.to_csv(APPROVED,index=False)
        st.success(f"Approved {int(idx)}. Total approved: {len(approved)}/{len(df)}")
with col2:
    if st.button('Save as draft'):
        df.loc[int(idx),'reviewer_intent']=intent; df.loc[int(idx),'reviewer_decision']=decision; df.loc[int(idx),'review_notes']=notes
        df.to_csv(PATH,index=False)
        st.info('Saved without approval.')
with col3:
    st.metric('Approved', int((df.status=='approved').sum()))
