"""Plotly figures shared by all pages. Class order is always alfalfa, maize, wheat.

Models are identified by a coloured legend square, never by long labels on the x-axis.
"""
import numpy as np
import plotly.graph_objects as go

from . import theme as T

LEGEND = dict(orientation='h', y=-0.12, x=0, yanchor='top', font=dict(size=12))


def _hex_rgba(hex_colour, alpha):
    h = hex_colour.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def confusion(cm, title, colour=T.QUANTUM, normalize=False, decimals=0, height=320):
    """Rows = true class, columns = predicted class. Counts in cells, optional row-normalized percentages."""
    cm = np.asarray(cm, float)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    z = pct if normalize else cm
    labels = [T.CLASS_LABELS[c] for c in T.CLASSES]
    text = [[(f'{cm[i, j]:.{decimals}f}' + (f'<br><span style="font-size:11px">{pct[i, j]:.0%}</span>' if normalize else ''))
             for j in range(3)] for i in range(3)]
    zmax = 1 if normalize else max(cm.max(), 1)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text,
        colorscale=[[0, '#FFFFFF'], [1, colour]], zmin=0, zmax=zmax, showscale=False, xgap=3, ygap=3,
        hovertemplate='True %{y} → predicted %{x}<br>%{text}<extra></extra>'))
    for i in range(3):  # white text on dark cells, navy on light cells
        for j in range(3):
            fig.add_annotation(x=labels[j], y=labels[i], text=text[i][j], showarrow=False,
                               font=dict(color='white' if z[i, j] / zmax > 0.55 else T.NAVY, size=16))
    fig.update_layout(title=title, height=height, xaxis_title='Predicted', yaxis_title='True',
                      yaxis_autorange='reversed', margin=dict(l=70, r=10, t=48, b=48))
    fig.update_xaxes(side='bottom', showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def bars(keys, values, title, yaxis, errors=None, text_fmt='.3f', yrange=None, height=360, log=False):
    """One bar per model, identified by its colour in the legend (no x-axis labels)."""
    fig = go.Figure()
    for i, (k, v) in enumerate(zip(keys, values)):
        err = None if errors is None or not errors[i] else dict(type='data', array=[errors[i]], color=T.SLATE)
        fig.add_bar(x=[T.short(k)], y=[v], name=T.name(k), marker_color=T.color(k), error_y=err,
                    text=[f'{v:{text_fmt}}'], textposition='outside', cliponaxis=False,
                    hovertemplate=T.name(k) + '<br>%{y:' + text_fmt + '}<extra></extra>')
    fig.update_layout(title=title, yaxis_title=yaxis, height=height, bargap=0.25, legend=LEGEND,
                      margin=dict(l=56, r=16, t=48, b=20))
    fig.update_xaxes(showticklabels=False)
    if log:
        fig.update_yaxes(type='log')
    if yrange:
        fig.update_yaxes(range=yrange)
    return fig


def grouped_recall(rows, title, height=380):
    """rows: list of (model key, [recall alfalfa, maize, wheat])."""
    fig = go.Figure()
    for key, rec in rows:
        fig.add_bar(name=T.name(key), x=[T.CLASS_LABELS[c] for c in T.CLASSES], y=rec, marker_color=T.color(key),
                    text=[f'{r:.0%}' for r in rec], textposition='outside', textfont=dict(size=11),
                    hovertemplate=T.name(key) + '<br>%{x}: %{y:.0%}<extra></extra>')
    fig.update_layout(title=title, barmode='group', yaxis_title='Recall', height=height,
                      yaxis_range=[0, 1.1], legend=LEGEND, margin=dict(l=56, r=16, t=48, b=20))
    fig.update_yaxes(tickformat='.0%')
    return fig


def architecture(height=430):
    """Submitted pipeline (challenge/submission.py) as an editable Plotly diagram; the dashed frame marks what ran on IBM."""
    fig = go.Figure()
    # (x0, y0, x1, y1, colour, title, subtitle)
    boxes = [
        (0.0, 3.7, 1.7, 6.3, T.SLATE, '13 features', 'Sentinel-2<br>12 bands + NDVI'),
        (2.1, 3.7, 4.1, 6.3, T.SLATE, 'Shared encoder', 'Scaler → PCA(8) → Scaler<br>fit on 914 training fields'),
        (4.6, 5.6, 7.3, 9.4, T.QUANTUM, '8-qubit feature map', 'H · RZ(2a·xᵢ) · RZZ(2γ·xᵢxᵢ₊₁)<br>a = 0.2, γ = 0.01, depth 9'),
        (7.7, 5.6, 9.9, 9.4, T.QUANTUM, 'Fidelity kernel', 'K = |⟨ψ(x)|ψ(x′)⟩|²<br>→ SVM (C = 100)'),
        (4.6, 0.6, 9.9, 4.4, T.CLASSICAL, 'Classical branch', 'RBF kernel (γ = 0.113) → SVM (C = 10)<br>same 8 components'),
        (10.4, 3.2, 12.7, 6.8, T.HYBRID, 'Decision fusion', 'α·softmax(d<sub>QK</sub>)<br>+ (1−α)·softmax(d<sub>RBF</sub>)<br>α = 0.53'),
        (13.2, 3.7, 14.8, 6.3, T.NAVY, 'Prediction', 'alfalfa<br>maize · wheat'),
    ]
    for x0, y0, x1, y1, c, t, s in boxes:
        fig.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=c, width=2),
                      fillcolor=_hex_rgba(c, 0.07), layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 0.55, text=f'<b>{t}</b>', showarrow=False,
                           font=dict(size=14, color=c))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 - 0.45, text=s, showarrow=False,
                           font=dict(size=11.5, color=T.NAVY))
    # QPU frame around the quantum branch
    fig.add_shape(type='rect', x0=4.4, y0=5.3, x1=10.1, y1=10.3, line=dict(color=T.HARDWARE, width=1.5, dash='dash'))
    fig.add_annotation(x=7.25, y=9.95, text='Kernel rows also measured on IBM ibm_kingston (inference only)',
                       showarrow=False, font=dict(size=11, color=T.HARDWARE))
    arrows = [((1.7, 5.0), (2.1, 5.0), T.SLATE), ((4.1, 5.6), (4.6, 7.5), T.QUANTUM), ((4.1, 4.4), (4.6, 2.5), T.CLASSICAL),
              ((7.3, 7.5), (7.7, 7.5), T.QUANTUM), ((9.9, 7.5), (10.4, 6.0), T.QUANTUM),
              ((9.9, 2.5), (10.4, 4.0), T.CLASSICAL), ((12.7, 5.0), (13.2, 5.0), T.HYBRID)]
    for (x0, y0), (x1, y1), c in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2, arrowcolor=c, text='')
    fig.update_xaxes(visible=False, range=[-0.2, 15.0])
    fig.update_yaxes(visible=False, range=[0.2, 10.6])
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=8, b=4), showlegend=False, dragmode=False)
    return fig


def lines(series, title, xaxis, yaxis, height=380, yrange=None, xvals=None):
    """series: list of dicts(key, x, y, sd=None, dash=None, label=None). Plain lines; sd only in the hover."""
    fig = go.Figure()
    for s in series:
        label = s.get('label') or T.name(s['key'])
        has_sd = s.get('sd') is not None
        fig.add_trace(go.Scatter(
            x=s['x'], y=s['y'], name=label, mode='lines+markers',
            line=dict(color=T.color(s['key']), width=3, dash=s.get('dash')), marker=dict(size=7),
            customdata=s.get('sd'),
            hovertemplate=label + '<br>σ = %{x}<br>%{y:.3f}' + (' ± %{customdata:.3f}' if has_sd else '')
            + '<extra></extra>'))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, height=height,
                      legend=dict(LEGEND, y=-0.22), margin=dict(l=56, r=16, t=48, b=20))
    if yrange:
        fig.update_yaxes(range=yrange)
    if xvals is not None:
        fig.update_xaxes(tickvals=xvals)
    return fig
