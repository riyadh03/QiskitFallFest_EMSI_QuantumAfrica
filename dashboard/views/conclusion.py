import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import cards as C
from components import charts as CH
from components import theme as T
from utils import data as D

st.title('Conclusion & hardware')

audit = D.load('audit') or {}
hwf = D.load('hw_fusion') or {}
hws = D.load('hw_summary') or {}
shots = D.load('shots')
reports = D.hardware_reports()

# ---------------------------------------------------------------- architecture
arch = D.asset('assets/architecture-diagram.png')
if arch:
    st.image(arch, width='stretch')

# ---------------------------------------------------------------- findings
c = st.columns(3)
with c[0]:
    C.note('<b>Representation first.</b> PCA-8 + kernel: 0.856 → 0.94.', 'info')
with c[1]:
    C.note('<b>Quantum kernel fixes wheat.</b> Recall 84% → 92%, the only significant gain.', 'info')
with c[2]:
    C.note('<b>No quantum advantage.</b> γ = 0 and a classical twin do as well.', 'limit')

left, right = st.columns(2)
with left:
    scan = (D.load('export') or {}).get('alpha_scan')
    if scan:
        fig = go.Figure(go.Scatter(x=[r['alpha'] for r in scan], y=[r['correct'] for r in scan], mode='lines',
                                   line=dict(color=T.HYBRID, width=3, shape='hv'),
                                   hovertemplate='α = %{x:.2f}<br>%{y}/150<extra></extra>'))
        fig.add_vline(x=0.53, line=dict(color=T.HYBRID, dash='dot'), annotation_text='α = 0.53')
        fig.update_layout(title='Fusion weight α: fields correct', xaxis_title='α (0 = RBF, 1 = quantum kernel)',
                          yaxis_title='Correct /150', height=330, yaxis_range=[136, 145],
                          margin=dict(l=56, r=16, t=48, b=40))
        st.plotly_chart(fig, width='stretch')
        st.caption('143/150 only for α ∈ [0.52, 0.54]: +1 field over the quantum kernel alone.')
with right:
    vs = audit.get('validation_scores') or {}
    if vs:
        names = {'quantum_kernel': 'quantum_kernel', 'matched_rbf': 'matched_rbf', 'fusion_qk_rbf': 'hybrid_fusion'}
        keys = [k for k in names if k in vs['reported_selected_on_validation']]
        fig = go.Figure()
        fig.add_bar(name='Tuned on validation', x=[T.short(names[k]) for k in keys],
                    y=[vs['reported_selected_on_validation'][k]['macro_f1'] for k in keys],
                    marker_color=[T.color(names[k]) for k in keys],
                    text=[f"{vs['reported_selected_on_validation'][k]['macro_f1']:.3f}" for k in keys],
                    textposition='outside')
        fig.add_bar(name='Tuned on training CV only', x=[T.short(names[k]) for k in keys],
                    y=[vs['selected_on_train_cv_only'][k]['macro_f1'] for k in keys],
                    marker=dict(color=[T.color(names[k]) for k in keys], opacity=0.35),
                    text=[f"{vs['selected_on_train_cv_only'][k]['macro_f1']:.3f}" for k in keys],
                    textposition='outside')
        fig.update_layout(title='Audit: tuning on validation is optimistic', barmode='group', yaxis_title='Macro-F1',
                          yaxis_range=[0.85, 0.98], height=330, legend=CH.LEGEND, margin=dict(l=56, r=16, t=48, b=20))
        st.plotly_chart(fig, width='stretch')
        st.caption('Expect a lower test score than 0.953.')

# ---------------------------------------------------------------- hardware
st.markdown('##### IBM Quantum hardware')
st.caption('SVMs trained locally. On the QPU we measured kernel rows only (compute-uncompute circuits); the RBF '
           'branch always runs on CPU.')
k = st.columns(4)
with k[0]:
    C.kpi('Backend', reports[0]['backend'] if reports else '—', 'Heron r2 · 156 qubits', T.HARDWARE)
with k[1]:
    C.kpi('Circuits', f"{sum(r['circuits'] for r in reports):,}", f"{len(reports)} jobs", T.HARDWARE)
with k[2]:
    C.kpi('Fields', hwf.get('fields', '—'), 'validation, pooled', T.HARDWARE)
with k[3]:
    C.kpi('QPU time', f"{sum((r.get('usage_seconds') or 0) for r in reports)} s", 'all jobs', T.HARDWARE)

st.dataframe([{'Job ID': r['job_id'], 'Fields': ', '.join(map(str, r['fields'])), 'Shots': r['shots'],
               'Circuits': r['circuits'], 'QPU s': r.get('usage_seconds'),
               'λ': round(float(np.mean(r['shrink_factor'])), 2) if r.get('shrink_factor') else None,
               'Used': '✓' if r.get('twirling') else '✗ calibration cancelled'} for r in reports],
             hide_index=True, width='stretch')

left, right = st.columns(2)
corr = hwf.get('correct') or {}
n = hwf.get('fields', 0)
with left:
    if corr:
        stages = ['Exact sim', 'Ideal shots', 'QPU']
        qk_vals = [(hws.get('correct') or {}).get(x) for x in ('exact', 'ideal_shots', 'hardware_corrected')]
        fu_vals = [corr.get(x) for x in ('fusion_exact', 'fusion_ideal_shots', 'fusion_hardware_corrected')]
        fig = go.Figure()
        fig.add_bar(name=T.name('quantum_kernel'), x=stages, y=qk_vals, marker_color=T.QUANTUM,
                    text=[f'{v}/{n}' for v in qk_vals], textposition='outside')
        fig.add_bar(name=T.name('hybrid_fusion'), x=stages, y=fu_vals, marker_color=T.HYBRID,
                    text=[f'{v}/{n}' for v in fu_vals], textposition='outside')
        fig.add_hline(y=corr.get('rbf_only_local'), line=dict(color=T.CLASSICAL, dash='dash'),
                      annotation_text=f"RBF on CPU {corr.get('rbf_only_local')}/{n}", annotation_position='top left')
        fig.update_layout(title=f'Correct on the {n} hardware fields', barmode='group', yaxis_title='Correct',
                          yaxis_range=[0, n + 2], height=340, legend=CH.LEGEND, margin=dict(l=56, r=16, t=48, b=20))
        st.plotly_chart(fig, width='stretch')
with right:
    if shots:
        fig = go.Figure()
        for kind, colour, label in [('ideal', T.QUANTUM, 'Ideal sampler'),
                                    ('hardware_like', T.HARDWARE, f"Noisy sim (λ = {shots['lam']})")]:
            r = [x for x in shots['rows'] if x['model'] == kind]
            fig.add_scatter(x=[x['shots'] for x in r], y=[x['macro_f1_mean'] for x in r], name=label,
                            mode='lines+markers', line=dict(color=colour, width=3))
        fig.add_hline(y=shots['exact_macro_f1'], line=dict(color=T.QUANTUM, dash='dot'),
                      annotation_text=f"exact {shots['exact_macro_f1']:.2f}")
        fig.update_xaxes(type='log', title='Shots per kernel entry')
        fig.update_layout(title='Shot noise (simulated, 150 fields)', yaxis_title='Macro-F1', height=340,
                          legend=dict(CH.LEGEND, y=-0.22), margin=dict(l=56, r=16, t=48, b=20))
        st.plotly_chart(fig, width='stretch')
st.caption('On the QPU the fusion falls back on the RBF: it never beats the RBF alone. Even 100 000 shots stay below '
           'exact simulation.')

per = hwf.get('per_field') or []
if per:
    with st.expander('Per-field hardware predictions'):
        mark = lambda p, t: f"{'✅' if p == t else '❌'} {p}"
        st.dataframe([{'Field': x['field'], 'True': x['true'], 'RBF (CPU)': mark(x['rbf_only'], x['true']),
                       'Fusion · exact': mark(x['fusion_exact'], x['true']),
                       'QK · QPU': mark(x['qk_hardware_corrected'], x['true']),
                       'Fusion · QPU': mark(x['fusion_hardware_corrected'], x['true'])} for x in per],
                     hide_index=True, width='stretch')

st.divider()
c1, c2 = st.columns([1, 6])
with c1:
    logo = D.asset('assets/team_logo.png')
    if logo:
        st.image(logo, width=90)
with c2:
    st.markdown('**Quantum Cube** — representation and kernel choice matter more than the circuit; the quantum kernel '
                'fixes wheat, and today\'s hardware noise erases the gain.')
    st.caption('Qiskit Fall Fest · EMSI · Quantum Africa · Data: Ouassanouan et al., Figshare v2, CC BY 4.0.')
