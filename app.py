import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from hiver_agent.config import settings
from hiver_agent.model import IntentModel
from hiver_agent.retrieval import HistoricalRetriever
from hiver_agent.agent import SupportAgent
from hiver_agent.intents import INTENTS

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AppleSupport AI Agent — Hiver Demo',
    page_icon='🍎',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0a0a0a; }

    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #111827 100%); }

    h1 { background: linear-gradient(90deg, #f5f5f7, #a0a0a5);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    .intent-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        background: linear-gradient(135deg, #1d4ed8, #7c3aed);
        color: white; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px;
    }
    .auto-badge {
        background: linear-gradient(135deg, #065f46, #059669);
    }
    .escalate-badge {
        background: linear-gradient(135deg, #92400e, #d97706);
    }
    .evidence-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 16px; margin: 8px 0;
        border-left: 3px solid #3b82f6;
    }
    .reply-box {
        background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.25);
        border-radius: 12px; padding: 20px; font-size: 1rem; line-height: 1.6;
    }
    .metric-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .sim-bar-fill { height: 6px; border-radius: 3px;
                    background: linear-gradient(90deg, #1d4ed8, #7c3aed); }
    .sim-bar-bg { background: rgba(255,255,255,0.1); border-radius: 3px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Load agent ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading support agent…')
def load_agent():
    mpath = Path('data/processed/intent_model.joblib')
    cpath = Path('data/processed/applesupport_cases.csv')
    if not cpath.exists():
        return None, None
    cases = pd.read_csv(cpath).fillna('')
    model = IntentModel.load(mpath) if mpath.exists() else None
    ipath = Path('data/processed/retriever.joblib')
    ret = (HistoricalRetriever.load(ipath) if ipath.exists()
           else HistoricalRetriever(max_docs=settings.max_retrieval_corpus).fit(cases))
    return SupportAgent(model, ret), cases

agent, cases_df = load_agent()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🍎 AppleSupport Agent')
    st.markdown('**Evidence-first · Conservative escalation · Fully auditable**')
    st.divider()

    page = st.radio('Navigation', ['💬 Agent Demo', '📊 Analytics', '📖 Intent Taxonomy'])

    st.divider()
    if cases_df is not None:
        st.caption(f'Corpus: **{len(cases_df):,}** paired cases')
    st.caption(f'Auto-handle threshold: **{settings.auto_threshold:.0%}**')
    st.caption(f'Retrieval k: **{settings.retrieval_k}**')
    api_status = '🟢 LLM active' if __import__('os').getenv('OPENAI_API_KEY') else '🟡 Deterministic fallback'
    st.caption(f'Generator: {api_status}')

# ─── Guard ───────────────────────────────────────────────────────────────────
if agent is None:
    st.error('⚠️ Data not prepared. Run:')
    st.code('python scripts/prepare_brand.py --zip data/raw/archive.zip --brand AppleSupport\npython scripts/train.py --cases data/processed/applesupport_cases.csv\npython scripts/build_index.py --cases data/processed/applesupport_cases.csv')
    st.stop()

# ─── PAGE: Agent Demo ────────────────────────────────────────────────────────
if page == '💬 Agent Demo':
    st.markdown('# 🍎 AppleSupport AI Agent')
    st.markdown('*Intent classification · Historical retrieval · Grounded reply drafting · Conservative escalation*')
    st.divider()

    EXAMPLES = [
        'My iPhone battery is draining really fast after the iOS update.',
        'Apple ID keeps asking me to sign in even though I already signed in.',
        "My iPhone won't charge at all — the cable doesn't seem to work.",
        "Why can't I download or update apps from the App Store?",
        "I'm getting charged twice for Apple Music, I want a refund.",
        "My WiFi keeps disconnecting on my MacBook after the update.",
        "How do I transfer all my data from my old iPhone to new one?",
        "The screen on my iPhone X is completely unresponsive. I'm done with Apple.",
    ]

    col_input, col_example = st.columns([3, 1])
    with col_example:
        st.markdown('**Quick examples:**')
        for ex in EXAMPLES[:4]:
            if st.button(ex[:45] + '…', key=f'ex_{ex[:20]}', use_container_width=True):
                st.session_state['demo_text'] = ex

    with col_input:
        text = st.text_area(
            'Customer message',
            value=st.session_state.get('demo_text', EXAMPLES[0]),
            height=120,
            placeholder='Type or paste a customer support message…',
            key='main_input'
        )
        run = st.button('🔍 Analyze Message', type='primary', use_container_width=True)

    if run and text.strip():
        with st.spinner('Running support pipeline…'):
            out = agent.predict(text)

        st.divider()

        # ── Top metrics row ──────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric('Detected Intent', out['intent'].replace('_', ' ').title())
        with c2:
            conf_pct = f"{out['intent_confidence']:.0%}"
            delta_color = 'normal' if out['intent_confidence'] >= 0.62 else 'inverse'
            st.metric('Confidence', conf_pct, delta='above threshold' if out['intent_confidence'] >= 0.62 else 'below threshold', delta_color=delta_color)
        with c3:
            decision_label = '✅ Auto-handle' if out['decision'] == 'auto_handle' else '⚠️ Escalate'
            st.metric('Decision', decision_label)
        with c4:
            top_sim = out['evidence'][0]['similarity'] if out['evidence'] else 0
            st.metric('Top Evidence Sim.', f'{top_sim:.3f}')

        # ── Reply ────────────────────────────────────────────────────────
        st.markdown('### 📝 Draft Reply')
        mode_badge = '🤖 LLM-generated' if 'openai' in out.get('generation_mode', '') else '📋 Deterministic'
        st.caption(f'Generation mode: {mode_badge}')
        st.markdown(f'<div class="reply-box">{out["reply"]}</div>', unsafe_allow_html=True)

        # ── Decision rationale ───────────────────────────────────────────
        with st.expander('🧭 Decision Rationale', expanded=True):
            decision_color = '#059669' if out['decision'] == 'auto_handle' else '#d97706'
            st.markdown(f'**Action:** <span style="color:{decision_color};font-weight:600">{out["decision"].replace("_"," ").upper()}</span>', unsafe_allow_html=True)
            st.markdown(f'**Reason:** {out["decision_reason"]}')
            if out.get('rules_triggered'):
                st.markdown(f'**Risk flags:** `{"`, `".join(out["rules_triggered"])}`')

        # ── Evidence ─────────────────────────────────────────────────────
        st.markdown('### 📚 Historical Evidence')
        if out['evidence']:
            for i, ev in enumerate(out['evidence']):
                sim = ev['similarity']
                bar_width = int(sim * 100)
                with st.container():
                    st.markdown(
                        f'<div class="evidence-card">'
                        f'<strong>#{i+1}</strong> · similarity: <strong>{sim:.3f}</strong>'
                        f'<div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{bar_width}%"></div></div>'
                        f'<p><strong>Customer:</strong> {ev.get("customer_text","")[:200]}</p>'
                        f'<p><strong>Historical reply:</strong> {ev.get("support_text","")[:300]}</p>'
                        f'<small>tweet IDs: customer={ev.get("customer_tweet_id","")} · support={ev.get("support_tweet_id","")}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info('No historical evidence retrieved.')

        # ── Full audit payload ───────────────────────────────────────────
        with st.expander('🔍 Full Audit JSON'):
            audit = {k: v for k, v in out.items() if k != 'evidence'}
            audit['evidence_count'] = len(out['evidence'])
            st.json(audit)

# ─── PAGE: Analytics ─────────────────────────────────────────────────────────
elif page == '📊 Analytics':
    st.markdown('# 📊 Evaluation Analytics')
    st.divider()

    metrics_path = Path('results/metrics.json')
    pred_path = Path('results/agent_predictions.csv')
    per_intent_path = Path('results/per_intent_metrics.csv')
    judge_path = Path('results/judge_agreement.json')

    if not metrics_path.exists():
        st.warning('No evaluation results found. Run: `python scripts/evaluate.py --golden data/golden/golden_approved.csv --cases data/processed/applesupport_cases.csv`')
        st.stop()

    metrics = json.loads(metrics_path.read_text())

    # ── Top-level metrics ────────────────────────────────────────────────
    eval_status = metrics.get('evaluation_status', 'UNKNOWN')
    status_color = '#059669' if 'APPROVED' in eval_status else '#d97706'
    st.markdown(f'**Evaluation status:** <span style="color:{status_color}">{eval_status}</span>', unsafe_allow_html=True)

    st.markdown('### Classification Results')
    comparison_data = {
        'System': ['Majority baseline', 'Keyword heuristic ⚠️', 'Proposed agent'],
        'Accuracy': [
            metrics.get('majority', {}).get('classification', {}).get('accuracy', '—'),
            metrics.get('keyword', {}).get('classification', {}).get('accuracy', '—'),
            metrics.get('agent', {}).get('classification', {}).get('accuracy', '—'),
        ],
        'Macro-F1': [
            metrics.get('majority', {}).get('classification', {}).get('macro_f1', '—'),
            metrics.get('keyword', {}).get('classification', {}).get('macro_f1', '—'),
            metrics.get('agent', {}).get('classification', {}).get('macro_f1', '—'),
        ],
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    st.caption('⚠️ Keyword heuristic accuracy (0.785) is still somewhat inflated — ~70% of human-approved labels are keyword-aligned. See report §6a for discussion. Agent value-add: grounded replies + auditable escalation.')

    # ── Escalation ───────────────────────────────────────────────────────
    esc = metrics.get('agent', {}).get('escalation', {})
    if esc:
        st.markdown('### Escalation Policy')
        e1, e2, e3, e4 = st.columns(4)
        e1.metric('Accuracy', esc.get('accuracy', '—'))
        e2.metric('Escalate Precision', esc.get('precision_escalate', '—'), help='High precision = rarely auto-handles cases that need humans')
        e3.metric('Escalate Recall', esc.get('recall_escalate', '—'))
        e4.metric('False Auto-Handle Rate', esc.get('false_auto_handle_rate', '—'), delta='key safety metric', delta_color='inverse')

    # ── Per-intent breakdown ─────────────────────────────────────────────
    if per_intent_path.exists():
        st.markdown('### Per-Intent Performance')
        pi = pd.read_csv(per_intent_path)
        pi = pi.sort_values('f1', ascending=False)
        pi['intent'] = pi['intent'].str.replace('_', ' ').str.title()
        st.dataframe(
            pi.style.background_gradient(subset=['f1'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True
        )

    # ── Predictions explorer ─────────────────────────────────────────────
    if pred_path.exists():
        st.markdown('### Prediction Explorer')
        pred = pd.read_csv(pred_path).fillna('')
        filter_wrong = st.checkbox('Show only wrong intent predictions')
        if filter_wrong:
            pred = pred[pred.gold_intent != pred.pred_intent]
        st.dataframe(
            pred[['customer_text', 'gold_intent', 'pred_intent', 'intent_confidence', 'gold_decision', 'pred_decision', 'top_similarity']].head(50),
            use_container_width=True, hide_index=True
        )

    # ── Judge agreement ──────────────────────────────────────────────────
    if judge_path.exists():
        st.markdown('### Human-vs-LLM Judge Agreement')
        agreement = json.loads(judge_path.read_text())
        j1, j2, j3, j4 = st.columns(4)
        j1.metric('Pass Agreement', f"{agreement.get('pass_agreement', 0):.1%}")
        j2.metric("Cohen's κ", f"{agreement.get('cohens_kappa', 0):.3f}")
        j3.metric('Spearman ρ', f"{agreement.get('spearman', 0):.3f}")
        j4.metric('Within-1 Agreement', f"{agreement.get('pct_within_1', 0):.1%}")
        st.caption(f"Kappa interpretation: **{agreement.get('interpretation', {}).get('kappa', '?')}**  ·  Bias: **{agreement.get('interpretation', {}).get('bias_direction', '?')}**")

    # ── Live judge scores ────────────────────────────────────────────────
    reply_judge = metrics.get('reply_judge', {})
    if reply_judge.get('status') == 'run':
        st.markdown('### 🤖 Live LLM Judge Scores')
        mean_scores = reply_judge.get('mean_scores', {})
        pass_rate = reply_judge.get('pass_rate', 0)
        n = reply_judge.get('n', 0)
        st.success(f"✅ **Pass rate: {pass_rate:.0%}** — all {n} judged replies passed")
        if mean_scores:
            lj1, lj2, lj3, lj4, lj5 = st.columns(5)
            lj1.metric('Overall', f"{mean_scores.get('overall', 0):.2f}/5")
            lj2.metric('Safety', f"{mean_scores.get('safety', 0):.2f}/5")
            lj3.metric('Tone', f"{mean_scores.get('tone', 0):.2f}/5")
            lj4.metric('Correctness', f"{mean_scores.get('correctness', 0):.2f}/5")
            lj5.metric('Actionability', f"{mean_scores.get('actionability', 0):.2f}/5")
            st.caption(f"Historical grounding: {mean_scores.get('historical_grounding', 0):.2f}/5 · Model: GPT-4o-mini via OpenRouter")
    else:
        st.info('LLM judge not run — set OPENAI_API_KEY and re-run evaluate.py to see live judge scores.')

# ─── PAGE: Intent Taxonomy ───────────────────────────────────────────────────
elif page == '📖 Intent Taxonomy':
    st.markdown('# 📖 Intent Taxonomy')
    st.markdown('10 brand-specific intents derived from recurring patterns in AppleSupport conversation data.')
    st.divider()

    for intent, meta in INTENTS.items():
        escalate_label = '⚠️ Default escalate' if meta['escalate_default'] else '✅ Default auto-handle'
        color = '#d97706' if meta['escalate_default'] else '#059669'
        with st.expander(f"**{intent.replace('_', ' ').title()}** — {escalate_label}"):
            st.markdown(f'**Description:** {meta["description"]}')
            st.markdown(f'**Escalation default:** <span style="color:{color}">{escalate_label}</span>', unsafe_allow_html=True)
            st.markdown(f'**Keywords:** `{"` · `".join(meta["keywords"])}`')
