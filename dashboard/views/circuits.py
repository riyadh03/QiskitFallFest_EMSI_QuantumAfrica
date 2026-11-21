import plotly.graph_objects as go
import streamlit as st

from components import cards as C
from components import charts as CH
from components import theme as T
from utils import data as D

st.title('Circuit benchmark')
st.caption('Same PCA-8 inputs, SVM and fusion rule — only the feature-map circuit changes.')

bench = D.load('circuits')
exp = D.load('export') or {}
res = {r['architecture']: r for r in exp.get('circuit_resources', [])}
if not bench:
    C.note(f"<code>{D.path('circuits')}</code> not found.", 'warn')
    st.stop()

TRACK_COLOR = {'Reference': T.NAVY, 'A · Feature map': '#7C3AED', 'B · Re-uploading': '#0EA5E9',
               'C · Topology': T.QUANTUM, 'D · Bandwidth': '#93C5FD'}


def track(name):
    n = name.lower()
    if n.startswith('current'):
        return 'Reference'
    if 'zzfeaturemap' in n or 'ry-rx' in n:
        return 'A · Feature map'
    if 're-uploading' in n:
        return 'B · Re-uploading'
    if 'bandwidth' in n:
        return 'D · Bandwidth'
    return 'C · Topology'


def short(name):
    return (name.replace('Current (Linear chain, a=0.2, g=0.01)', 'Linear chain (reference)')
            .replace(' Entanglement', '').replace('Bandwidth scaling ', '').replace('(reps=1, linear)', '')
            .replace('Data Re-uploading (2 layers + RY(pi/4) mixing)', 'Re-uploading ×2'))


rows = []
for r in bench:
    rr = res.get(r['Architecture'], {})
    rows.append(dict(name=short(r['Architecture']), track=track(r['Architecture']), qk_f1=r['QK_F1'],
                     hy_f1=r['Hybrid_F1'], correct=r['Hybrid_Correct'], wheat=r['Wheat_Rec'],
                     depth=rr.get('depth'), twoq=rr.get('two_qubit_gates')))

c1, c2 = st.columns([2, 1])
with c1:
    pick = st.radio('Track', ['All'] + list(TRACK_COLOR)[1:], horizontal=True, label_visibility='collapsed')
with c2:
    metric = st.radio('Score', ['qk_f1', 'hy_f1'], horizontal=True, label_visibility='collapsed',
                      format_func=lambda m: {'qk_f1': 'Kernel alone', 'hy_f1': 'In the fusion'}[m])
shown = [r for r in rows if pick == 'All' or r['track'] in (pick, 'Reference')]

fig = go.Figure()
for tr, colour in TRACK_COLOR.items():
    pts = [r for r in shown if r['track'] == tr]
    if pts:
        fig.add_bar(y=[r['name'] for r in pts], x=[r[metric] for r in pts], orientation='h', name=tr,
                    marker_color=colour, text=[f"{r[metric]:.3f}" for r in pts], textposition='outside',
                    cliponaxis=False, hovertemplate='%{y}<br>%{x:.4f}<extra></extra>')
h = 150 + 34 * len(shown)
fig.update_layout(title='Macro-F1 by circuit design', xaxis_title='Macro-F1', height=h,
                  legend=dict(CH.LEGEND, y=-70 / h), barmode='overlay',
                  yaxis=dict(autorange='reversed'), margin=dict(l=180, r=40, t=48, b=20),
                  xaxis_range=[min(0.25, min(r[metric] for r in shown) - 0.05), 1.02])
st.plotly_chart(fig, width='stretch')

table = [{'Track': r['track'], 'Design': r['name'], 'Depth': str(r['depth']) if r['depth'] is not None else '—',
          '2-qubit gates': str(r['twoq']) if r['twoq'] is not None else '—', 'Kernel F1': round(r['qk_f1'], 4),
          'Fusion F1': round(r['hy_f1'], 4), 'Fusion /150': r['correct'], 'Wheat recall': f"{r['wheat']:.0%}"}
         for r in shown]
st.dataframe(table, hide_index=True, width='stretch')
st.caption('Logical depth before transpilation. "Star" is no longer defined in challenge/circuits.py, so its depth '
           'is not recorded.')

c1, c2 = st.columns(2)
with c1:
    known = [r for r in rows if r['twoq'] is not None]
    fig = go.Figure()
    for tr, colour in TRACK_COLOR.items():
        pts = [r for r in known if r['track'] == tr]
        if pts:
            fig.add_scatter(x=[r['twoq'] for r in pts], y=[r['qk_f1'] for r in pts], mode='markers', name=tr,
                            marker=dict(size=12, color=colour, line=dict(width=1, color='white')),
                            text=[r['name'] for r in pts],
                            hovertemplate='%{text}<br>%{x} two-qubit gates<br>F1 %{y:.3f}<extra></extra>')
    fig.update_layout(title='More entangling gates ≠ better kernel', xaxis_title='Two-qubit gates',
                      yaxis_title='Kernel macro-F1', height=360, legend=dict(CH.LEGEND, y=-0.22),
                      margin=dict(l=56, r=16, t=48, b=20))
    st.plotly_chart(fig, width='stretch')
with c2:
    g = D.load('grid_quantum')
    if g is not None:
        piv = g.groupby(['a', 'gamma'])['f1'].max().reset_index().pivot(index='gamma', columns='a', values='f1')
        fig = go.Figure(go.Heatmap(z=piv.values, x=[str(a) for a in piv.columns], y=[str(v) for v in piv.index],
                                   colorscale=[[0, '#EFF6FF'], [1, T.QUANTUM]], text=piv.values,
                                   texttemplate='%{text:.3f}', showscale=False,
                                   hovertemplate='a = %{x}, γ = %{y}<br>%{z:.4f}<extra></extra>'))
        fig.update_layout(title='Grid search: best F1 per (a, γ)', xaxis_title='Angle scale a',
                          yaxis_title='RZZ strength γ', height=360, margin=dict(l=56, r=16, t=48, b=40))
        st.plotly_chart(fig, width='stretch')

img = D.asset('outputs/dashboard/qk_feature_map.png')
if img:
    st.image(img, caption='Reference circuit: H · RZ(0.4·x) · RZZ(0.02·xᵢxᵢ₊₁) — depth 9, 7 RZZ.', width='stretch')

C.note('<b>Takeaway:</b> the angle scale a decides; entanglement topology and depth do not help.', 'hybrid')
