import plotly.graph_objects as go
import streamlit as st

from components import cards as C
from components import charts as CH
from components import theme as T
from utils import data as D
from utils import robustness as RB

st.title('Robustness')
st.caption('Models trained on clean data, tested on noisy copies of the 150 validation fields. Synthetic noise is a '
           'stress test, not a simulation of a real sensor fault. Sensor bias was not tested.')

c1, c2, c3 = st.columns([2.2, 1, 1.3])
with c1:
    protocol = st.radio('Protocol', list(RB.PROTOCOLS), horizontal=True,
                        format_func=lambda p: RB.PROTOCOLS[p]['short'])
spec = RB.PROTOCOLS[protocol]
with c2:
    metric = st.radio('Metric', ['macro_f1'] + (['accuracy'] if spec['has_accuracy'] else []), horizontal=True,
                      format_func=lambda m: {'macro_f1': 'Macro-F1', 'accuracy': 'Accuracy'}[m])
with c3:
    view = st.radio('Scale', ['absolute', 'relative'], horizontal=True,
                    format_func=lambda v: {'absolute': 'Absolute', 'relative': 'Δ vs clean'}[v])

series = RB.series(protocol, metric)
if not series:
    C.note(f"<code>{D.path(spec['source'])}</code> not found.", 'warn')
    st.stop()
keys = st.multiselect('Models', [s['key'] for s in series], default=[s['key'] for s in series],
                      format_func=lambda k: next(s.get('label') or T.name(k) for s in series if s['key'] == k),
                      label_visibility='collapsed')
series = [s for s in series if s['key'] in keys]
if not series:
    st.stop()

label = {'macro_f1': 'Macro-F1', 'accuracy': 'Accuracy'}[metric]
plot = RB.degradation(series) if view == 'relative' else series
st.plotly_chart(CH.lines(plot, f'{label} vs noise σ', 'Noise σ',
                         label if view == 'absolute' else f'Δ {label} vs clean', height=420, xvals=series[0]['x']),
                width='stretch')
short = {'paired': 'σ × training std per band · 5 draws, same noise for every model (hover shows ± sd).',
         'reliability': 'σ × training std per band · 5 draws (hover shows ± sd).',
         'pca_noise': 'Noise in standardized PCA space · 1 draw · fusion at α = 0.5. Not comparable with the other '
                      'two protocols.'}[protocol]
st.caption(short)
C.source(D.path(spec['source']))

# ---------------------------------------------------------------- band drop
if protocol == 'reliability':
    fd = RB.band_drop()
    if fd:
        fig = go.Figure()
        for m in ['reference', 'matched_rbf', 'quantum_kernel']:
            fig.add_bar(name=T.name(m), x=[r['feature'] for r in fd], y=[r[m] for r in fd], marker_color=T.color(m))
        fig.update_layout(title='Band drop: Δ macro-F1 when one band is replaced by its training mean',
                          barmode='group', xaxis_title='Missing band', yaxis_title='Δ macro-F1', height=380,
                          legend=dict(CH.LEGEND, y=-0.25), margin=dict(l=56, r=16, t=48, b=20))
        st.plotly_chart(fig, width='stretch')
