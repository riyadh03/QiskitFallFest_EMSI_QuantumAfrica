import plotly.graph_objects as go
import streamlit as st

from components import cards as C
from components import charts as CH
from components import theme as T
from utils import registry as R
from utils import robustness as RB

st.title('Experiments')

exps = R.experiments()
by_key = {e['key']: e for e in exps}
options = [e['key'] for g, _ in R.GROUPS for e in exps if e['group'] == g]
default = st.query_params.get('exp', 'hybrid_fusion')
key = st.pills('Experiment', options, default=default if default in options else options[-1],
               format_func=T.short, label_visibility='collapsed') or options[-1]
st.query_params['exp'] = key
e = by_key[key]
colour = T.color(key)

st.markdown(f"#### {e['title']}")
st.markdown(f"{C.family_pill(key)}{C.pill(e['phase'], T.SLATE)}{C.pill('Validation · 150 fields', T.SLATE)}",
            unsafe_allow_html=True)
st.caption(e['findings'][0])

# ---------------------------------------------------------------- results
m = e.get('metrics')
sd = e.get('metrics_sd') or {}
if m:
    k = st.columns(4)
    with k[0]:
        C.kpi('Macro-F1', C.fmt(m['macro_f1'], 4), f"± {sd['macro_f1']:.3f} · 3 seeds" if 'macro_f1' in sd else '',
              colour)
    with k[1]:
        C.kpi('Accuracy', C.fmt(m['accuracy'], 4), f"± {sd['accuracy']:.3f} · 3 seeds" if 'accuracy' in sd else '',
              colour)
    with k[2]:
        C.kpi('Training', C.seconds(e.get('fit')))
    with k[3]:
        C.kpi('Inference · 150 fields', C.seconds(e.get('predict')))

    left, right = st.columns(2)
    with left:
        if m.get('confusion'):
            dec = 0 if all(float(v).is_integer() for row in m['confusion'] for v in row) else 1
            st.plotly_chart(CH.confusion(m['confusion'], 'Confusion matrix' + (' (mean of 3 seeds)' if dec else ''),
                                         colour, decimals=dec), width='stretch')
        else:
            C.note('Confusion matrix: not recorded.', 'warn')
    with right:
        rec = m.get('recall') or [None] * 3
        prec = m.get('precision') or [None] * 3
        labels = [T.CLASS_LABELS[c] for c in T.CLASSES]
        fig = go.Figure()
        if any(p is not None for p in prec):
            fig.add_bar(name='Precision', x=labels, y=prec, marker_color=CH._hex_rgba(colour, 0.4),
                        text=[f'{p:.0%}' for p in prec], textposition='outside')
        fig.add_bar(name='Recall', x=labels, y=[r if r is not None else 0 for r in rec], marker_color=colour,
                    text=[f'{r:.0%}' if r is not None else 'n/r' for r in rec], textposition='outside')
        fig.update_layout(title='Per-class precision & recall', barmode='group', yaxis_range=[0, 1.12], height=320,
                          legend=CH.LEGEND, margin=dict(l=50, r=10, t=48, b=20))
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, width='stretch')
else:
    C.note('No saved metrics for this experiment.', 'warn')

# ---------------------------------------------------------------- configuration + robustness
left, right = st.columns([1, 1.25])
with left:
    st.markdown('##### Configuration')
    C.meta_panel(e['config'])
with right:
    st.markdown('##### Robustness')
    protos = e.get('robustness') or []
    if not protos:
        st.caption('No perturbation test for this model.')
    else:
        p = protos[0] if len(protos) == 1 else st.radio(
            'Protocol', protos, horizontal=True, label_visibility='collapsed',
            format_func=lambda x: RB.PROTOCOLS[x]['short'])
        series = RB.series(p)
        for s in series:
            if s['key'] != key:
                s['dash'] = 'dot'
        st.plotly_chart(CH.lines(series, RB.PROTOCOLS[p]['short'], 'Noise σ', 'Macro-F1', height=330,
                                 xvals=series[0]['x'] if series else None), width='stretch')

if e['limitations']:
    st.caption('⚠ ' + e['limitations'][0])

# ---------------------------------------------------------------- metadata
with st.expander('Execution metadata & sources'):
    meta = {k_: v for k_, v in e['meta'].items() if k_ != 'Environment' or v}
    meta.update({k_: v for k_, v in e['resources'].items() if v is not None})
    C.meta_panel(meta)
    C.source(*e['sources'])
