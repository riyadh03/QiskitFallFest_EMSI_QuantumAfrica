import plotly.graph_objects as go
import streamlit as st

from components import cards as C
from components import charts as CH
from components import theme as T
from utils import data as D

st.markdown('<div class="cl-eyebrow">Quantum Cube</div>', unsafe_allow_html=True)
st.title('Quantum Cube CropLab')
st.markdown('<p class="cl-lead">Classifying <b>alfalfa, maize and wheat</b> fields in Doukkala, Morocco, from 13 '
            'Sentinel-2 features — with quantum, classical and hybrid models.</p>', unsafe_allow_html=True)

arch = D.asset('assets/architecture-diagram.png')
if arch:
    st.image(arch, width='stretch')

manifest = D.load('manifest') or {}
rows = manifest.get('rows', {})
fm = ((D.load('export') or {}).get('fusion') or {}).get('metrics') or {}
qk = D.qk_run('quantum_kernel')
rbf = D.qk_run('matched_rbf')
hw = D.load('hw_fusion') or {}

k = st.columns(4)
with k[0]:
    C.kpi('Fusion (submitted)', C.fmt(fm.get('macro_f1'), 3), 'macro-F1 · validation', T.HYBRID)
with k[1]:
    C.kpi('Quantum kernel', C.fmt(qk and qk['metrics']['macro_f1'], 3), 'macro-F1 · wheat recall 92%', T.QUANTUM)
with k[2]:
    C.kpi('Matched RBF', C.fmt(rbf and rbf['metrics']['macro_f1'], 3), 'macro-F1 · wheat recall 84%', T.CLASSICAL)
with k[3]:
    C.kpi('IBM ibm_kingston', f"{len(hw.get('jobs', []))} jobs", f"{hw.get('fields', '—')} fields on hardware",
          T.HARDWARE)

st.write('')
st.markdown('##### Submitted pipeline')
C.flow([
    ('13 features', 'Sentinel-2 bands + NDVI', T.SLATE),
    ('PCA(8)', 'fitted on training only', T.SLATE),
    ('Quantum kernel', '8 qubits → SVM', T.QUANTUM),
    ('RBF kernel', 'classical → SVM', T.CLASSICAL),
    ('Fusion', '0.53·QK + 0.47·RBF', T.HYBRID),
    ('Crop', 'alfalfa · maize · wheat', T.NAVY),
])

st.write('')
left, right = st.columns([1, 1])
with left:
    cc = D.load('class_counts')
    if cc is not None:
        fig = go.Figure()
        for split, colour in [('Training', T.SLATE), ('Validation', T.QUANTUM)]:
            fig.add_bar(name=split, x=[T.CLASS_LABELS[c] for c in cc['target']], y=cc[split], marker_color=colour,
                        text=cc[split], textposition='outside')
        fig.update_layout(title='Fields per crop', yaxis_title='Fields', barmode='group', height=300,
                          legend=CH.LEGEND, margin=dict(l=50, r=10, t=48, b=20),
                          yaxis_range=[0, cc['Training'].max() * 1.2])
        st.plotly_chart(fig, width='stretch')
with right:
    st.markdown(f"""
| Split | Fields | Use |
|---|---:|---|
| Training | {rows.get('train', '—')} | fit models |
| Validation | {rows.get('validation', '—')} | select & report |
| Test | {rows.get('test', '—')} | labels private |
""")
    st.caption('Metric: macro-F1 over the 3 crops. Main results use exact simulation (≤ 8 qubits).')
