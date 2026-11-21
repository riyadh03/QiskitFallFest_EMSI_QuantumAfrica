import streamlit as st

from components import cards as C
from components import charts as CH
from components import theme as T
from utils import data as D
from utils import registry as R

st.title('Architecture benchmark')

exps = {e['key']: e for e in R.experiments()}
ARCH = ['reference', 'matched_rbf', 'starter_qnn', 'qnn_benchmark', 'qks', 'quantum_kernel', 'hybrid_fusion']
ARCH = [k for k in ARCH if exps.get(k, {}).get('metrics')]
sel = st.multiselect('Architectures', ARCH, default=ARCH, format_func=T.name, label_visibility='collapsed')
if not sel:
    st.stop()
st.caption('Validation, 150 fields. QNN L=1 was trained on 200 rows only; QKS features are computed in closed form.')

# ---------------------------------------------------------------- scores
f1 = [exps[k]['metrics']['macro_f1'] for k in sel]
f1_sd = [(exps[k].get('metrics_sd') or {}).get('macro_f1') for k in sel]
acc = [exps[k]['metrics']['accuracy'] for k in sel]
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(CH.bars(sel, f1, 'Macro-F1', 'Macro-F1', errors=f1_sd, yrange=[0.5, 1.02]), width='stretch')
with c2:
    st.plotly_chart(CH.bars(sel, acc, 'Accuracy', 'Accuracy', yrange=[0.5, 1.02]), width='stretch')

# ---------------------------------------------------------------- recall
full = [k for k in sel if exps[k]['metrics'].get('recall') and None not in exps[k]['metrics']['recall']]
st.plotly_chart(CH.grouped_recall([(k, exps[k]['metrics']['recall']) for k in full], 'Per-class recall'),
                width='stretch')
partial = [k for k in sel if k not in full]
if partial:
    st.caption('Wheat recall only (others not recorded): ' + ' · '.join(
        f"{T.name(k)} {exps[k]['metrics']['recall'][2]:.0%}" for k in partial))

# ---------------------------------------------------------------- cost
c1, c2 = st.columns(2)
fit = [(k, exps[k].get('fit')) for k in sel if exps[k].get('fit')]
pred = [(k, exps[k].get('predict')) for k in sel if exps[k].get('predict')]
with c1:
    st.plotly_chart(CH.bars([k for k, _ in fit], [v for _, v in fit], 'Training time (log)', 'Seconds',
                            text_fmt='.2f', log=True), width='stretch')
with c2:
    st.plotly_chart(CH.bars([k for k, _ in pred], [v for _, v in pred], 'Inference time, 150 fields (log)',
                            'Seconds', text_fmt='.3f', log=True), width='stretch')

# ---------------------------------------------------------------- table
rows = []
for k in sel:
    e = exps[k]
    cfg, res = e['config'], e['resources']
    rows.append({'Model': T.name(k), 'Family': T.FAMILY_LABEL[T.family(k)],
                 'Features': cfg.get('Features', '—'), 'Qubits': str(cfg.get('Qubits', '—')),
                 'Size': str(res.get('Trainable angles') or res.get('Trainable parameters')
                          or (f"{res['Support vectors']} SVs" if res.get('Support vectors') else None)
                          or res.get('Branches') or '—'),
                 'Macro-F1': round(e['metrics']['macro_f1'], 4)})
st.dataframe(rows, hide_index=True, width='stretch')

# ---------------------------------------------------------------- side by side
cm_keys = [k for k in sel if exps[k]['metrics'].get('confusion')]
if cm_keys:
    c1, c2 = st.columns(2)
    pick_a = c1.selectbox('Left', cm_keys, index=cm_keys.index('matched_rbf') if 'matched_rbf' in cm_keys else 0,
                          format_func=T.name)
    pick_b = c2.selectbox('Right', cm_keys, index=cm_keys.index('hybrid_fusion') if 'hybrid_fusion' in cm_keys else 0,
                          format_func=T.name)
    for side, (col, k) in enumerate([(c1, pick_a), (c2, pick_b)]):
        with col:
            cm = exps[k]['metrics']['confusion']
            dec = 0 if all(float(v).is_integer() for r in cm for v in r) else 1
            st.plotly_chart(CH.confusion(cm, T.name(k), T.color(k), decimals=dec), width='stretch',
                            key=f'side_cm_{side}')

# ---------------------------------------------------------------- uncertainty
audit = D.load('audit')
qs = D.load('qk_summary') or {}
rows = [{'B − A': n.replace('_', ' '), 'Δ macro-F1': round(c['macro_f1'], 4),
         '95% CI': f"[{c['macro_f1_ci'][0]:+.3f}, {c['macro_f1_ci'][1]:+.3f}]"}
        for n, c in (qs.get('comparisons') or {}).items()]
if audit:
    rows += [{'B − A': n.replace('_', ' '), 'Δ macro-F1': round(c['per_spatial_block']['macro_f1']['delta'], 4),
              '95% CI': '[{:+.3f}, {:+.3f}]'.format(*c['per_spatial_block']['macro_f1']['ci95'])}
             for n, c in audit['paired_bootstrap']['comparisons'].items()]
if rows:
    with st.expander('Are the differences significant? (spatial-block bootstrap)'):
        st.dataframe(rows, hide_index=True, width='stretch')
        st.caption('Only PCA-8 models vs the reference exclude 0. QK, RBF and fusion are not distinguishable on '
                   'macro-F1.')

C.note('<b>Verdict:</b> fusion 0.953 ≈ quantum kernel 0.940–0.947 ≈ matched RBF 0.932 (1–3 fields apart). '
       'The RBF is the cheapest. QNN and QKS trail by 0.10–0.15.', 'hybrid')
