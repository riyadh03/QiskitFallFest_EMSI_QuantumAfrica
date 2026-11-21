"""Reusable HTML blocks: KPI cards, notes, pills, metadata panels and source captions."""
from html import escape

import streamlit as st

from . import theme as T

NOT_RECORDED = 'Not recorded'


def fmt(v, digits=4, suffix=''):
    if v is None or v == '':
        return NOT_RECORDED
    if isinstance(v, float):
        return f'{v:.{digits}f}{suffix}'
    return f'{v}{suffix}'


def seconds(v):
    if v is None:
        return NOT_RECORDED
    return f'{v * 1000:.0f} ms' if v < 1 else f'{v:.1f} s'


def kpi(label, value, sub='', accent=T.NAVY):
    st.markdown(f"""<div class="cl-card"><div class="cl-kpi-label">{escape(label)}</div>
        <div class="cl-kpi-value" style="color:{accent}">{escape(str(value))}</div>
        <div class="cl-kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)


def pill(text, colour):
    return (f'<span class="cl-pill" style="color:{colour};border-color:{colour}33;background:{colour}12">'
            f'{escape(text)}</span>')


def family_pill(key):
    fam = T.family(key)
    return pill(T.FAMILY_LABEL[fam], T.FAMILY_COLOR[fam])


def note(text, kind='info'):
    colour = {'info': T.QUANTUM, 'warn': '#D97706', 'ok': T.CLASSICAL, 'limit': '#DC2626', 'hybrid': T.HYBRID}[kind]
    st.markdown(f'<div class="cl-note" style="border-left-color:{colour}">{text}</div>', unsafe_allow_html=True)


def source(*paths):
    files = ', '.join(f'<code>{escape(p)}</code>' for p in paths)
    st.markdown(f'<div class="cl-source">Source: {files}</div>', unsafe_allow_html=True)


def meta_panel(items: dict):
    """Compact metadata grid. Values that are None show as 'Not recorded'."""
    cells = ''.join(f'<div><span>{escape(k)}</span><b>{escape(fmt(v) if not isinstance(v, str) else v)}</b></div>'
                    for k, v in items.items())
    st.markdown(f'<div class="cl-card"><div class="cl-meta">{cells}</div></div>', unsafe_allow_html=True)


def header(eyebrow, title, lead=None):
    st.markdown(f'<div class="cl-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    if lead:
        st.markdown(f'<p class="cl-lead">{lead}</p>', unsafe_allow_html=True)


def flow(steps):
    """steps: list of (title, subtitle, colour). Simple horizontal pipeline that wraps on narrow screens."""
    parts = []
    for i, (t, s, c) in enumerate(steps):
        if i:
            parts.append('<div class="arrow">→</div>')
        parts.append(f'<div class="step" style="border-top-color:{c}"><b>{escape(t)}</b><small>{s}</small></div>')
    st.markdown(f'<div class="cl-flow">{"".join(parts)}</div>', unsafe_allow_html=True)
