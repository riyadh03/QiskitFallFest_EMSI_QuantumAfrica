"""Colours, model identities and global styling. One colour per model everywhere in the app.

Colour meaning: blue = quantum, green = classical, orange = hybrid, slate = hardware / neutral.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = '#1E293B'
SLATE = '#64748B'
BORDER = '#E2E8F0'
SURFACE = '#F8FAFC'

QUANTUM = '#2563EB'
CLASSICAL = '#16A34A'
HYBRID = '#EA580C'
HARDWARE = '#334155'

CLASSES = ['alfalfa', 'maize', 'wheat']
CLASS_LABELS = {'alfalfa': 'Alfalfa', 'maize': 'Maize', 'wheat': 'Wheat'}

# key -> (display name, short name, family, colour). Families drive the colour; shades keep models distinguishable.
MODELS = {
    'reference':        ('Supplied reference SVM',        'Reference SVM',    'classical', '#65A30D'),
    'matched_head':     ('Starter MLP control',           'MLP control',      'classical', '#84CC16'),
    'matched_rbf':      ('Matched RBF-SVM (PCA-8)',       'Matched RBF',      'classical', CLASSICAL),
    'classical_twin':   ('Classical twin kernel',         'Classical twin',   'classical', '#0F766E'),
    'starter_qnn':      ('Starter QNN (4 qubits)',        'Starter QNN',      'quantum',   '#1E3A8A'),
    'qnn_benchmark':    ('Variational QNN (L=1)',         'QNN L=1',          'quantum',   '#3B82F6'),
    'qks':              ('Quantum Kitchen Sinks',         'QKS',              'quantum',   '#0EA5E9'),
    'quantum_kernel':   ('8-qubit quantum kernel',        'Quantum kernel',   'quantum',   QUANTUM),
    'no_entanglement':  ('Quantum kernel, γ = 0',         'QK no entangl.',   'quantum',   '#93C5FD'),
    'hybrid_fusion':    ('Decision fusion (submitted)',   'Fusion',           'hybrid',    HYBRID),
}
FAMILY_LABEL = {'classical': 'Classical', 'quantum': 'Quantum', 'hybrid': 'Hybrid'}
FAMILY_COLOR = {'classical': CLASSICAL, 'quantum': QUANTUM, 'hybrid': HYBRID}


def name(key):
    return MODELS[key][0]


def color(key):
    return MODELS[key][3]


def short(key):
    return MODELS[key][1]


def family(key):
    return MODELS[key][2]


def _template():
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family='Inter, Segoe UI, Helvetica, Arial, sans-serif', size=14, color=NAVY),
        title=dict(font=dict(size=17, color=NAVY), x=0, xanchor='left', y=0.97),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(gridcolor='#EEF2F7', linecolor=BORDER, zerolinecolor=BORDER, ticks='outside', tickcolor=BORDER,
                   title=dict(font=dict(size=14, color=SLATE))),
        yaxis=dict(gridcolor='#EEF2F7', linecolor=BORDER, zerolinecolor=BORDER, ticks='outside', tickcolor=BORDER,
                   title=dict(font=dict(size=14, color=SLATE))),
        legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor=BORDER, borderwidth=1, font=dict(size=13)),
        margin=dict(l=60, r=24, t=64, b=56),
        hoverlabel=dict(bgcolor='white', font=dict(color=NAVY), bordercolor=BORDER),
        colorway=[QUANTUM, CLASSICAL, HYBRID, '#0EA5E9', '#65A30D', '#93C5FD', HARDWARE],
    )
    return t


pio.templates['croplab'] = _template()
pio.templates.default = 'croplab'

CSS = f"""
<style>
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
  h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
  h1 {{ font-weight: 750; }}
  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  .cl-card {{ background: white; border: 1px solid {BORDER}; border-radius: 14px; padding: 10px 14px;
              box-shadow: 0 1px 2px rgba(15,23,42,0.04); height: 100%; }}
  .cl-kpi-label {{ color: {SLATE}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em;
                   font-weight: 600; }}
  .cl-kpi-value {{ font-size: 1.45rem; font-weight: 750; line-height: 1.15; margin-top: 4px; }}
  .cl-kpi-sub {{ color: {SLATE}; font-size: 0.76rem; margin-top: 4px; }}
  .cl-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
              font-weight: 650; margin-right: 6px; border: 1px solid; }}
  .cl-eyebrow {{ color: {QUANTUM}; font-weight: 700; font-size: 0.8rem; text-transform: uppercase;
                 letter-spacing: .1em; margin-bottom: -6px; }}
  .cl-lead {{ font-size: 1.0rem; color: #334155; line-height: 1.6; }}
  .cl-note {{ border-left: 4px solid; border-radius: 8px; padding: 8px 12px; margin: 4px 0 10px 0;
              background: {SURFACE}; color: #334155; font-size: 0.88rem; line-height: 1.45; }}
  .cl-meta {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 6px 14px; }}
  .cl-meta div span {{ display: block; color: {SLATE}; font-size: 0.76rem; text-transform: uppercase;
                       letter-spacing: .05em; font-weight: 600; }}
  .cl-meta div b {{ font-weight: 600; color: {NAVY}; font-size: 0.86rem; word-break: break-word; }}
  .cl-muted {{ color: {SLATE}; }}
  .cl-source {{ color: {SLATE}; font-size: 0.8rem; margin-top: -6px; }}
  .cl-source code {{ font-size: 0.78rem; }}
  .cl-flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; }}
  .cl-flow .step {{ flex: 1 1 140px; background: white; border: 1px solid {BORDER}; border-radius: 12px;
                    padding: 8px 10px; border-top: 4px solid {SLATE}; font-size: 0.85rem; }}
  .cl-flow .step b {{ display: block; font-size: 0.95rem; }}
  .cl-flow .step small {{ color: {SLATE}; }}
  .cl-flow .arrow {{ align-self: center; color: {SLATE}; font-size: 1.3rem; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)
