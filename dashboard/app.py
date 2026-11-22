"""Q³ CropLab — interactive research explorer for the Moroccan Crop Classification Challenge.

Launch from the repository root:
    streamlit run dashboard/app.py

The app only reads saved results (outputs/, results/, configs/). It never trains a model or runs a circuit.
Generate the one missing file once with:  python -m dashboard.export
"""
import streamlit as st

from components import theme
from utils import data as D

st.set_page_config(page_title='Q³ CropLab', page_icon='🌾', layout='wide',
                   initial_sidebar_state='expanded')
theme.apply()

pages = [
    st.Page('views/overview.py', title='Overview', icon=':material/home:', default=True),
    st.Page('views/experiments.py', title='Experiments', icon=':material/science:'),
    st.Page('views/robustness.py', title='Robustness', icon=':material/shield:'),
    st.Page('views/architectures.py', title='Architecture benchmark', icon=':material/leaderboard:'),
    st.Page('views/circuits.py', title='Circuit benchmark', icon=':material/memory:'),
    st.Page('views/conclusion.py', title='Conclusion & hardware', icon=':material/flag:'),
]

with st.sidebar:
    logo = D.asset('assets/team_logo.png')
    if logo:
        st.image(logo, width=92)
    st.markdown('### Q³ CropLab')
    st.caption('Moroccan Crop Classification Challenge · Qiskit Fall Fest · EMSI · Quantum Africa')

nav = st.navigation(pages, position='sidebar')

with st.sidebar:
    st.divider()
    st.markdown(
        f"<span style='color:{theme.QUANTUM}'>■</span> Quantum &nbsp; "
        f"<span style='color:{theme.CLASSICAL}'>■</span> Classical &nbsp; "
        f"<span style='color:{theme.HYBRID}'>■</span> Hybrid", unsafe_allow_html=True)
    st.caption('All scores are on the 150-field validation split unless stated otherwise. '
               'Test labels are private; no test accuracy is shown.')
    if not D.exists('export'):
        st.warning('outputs/dashboard/dashboard_export.json is missing. Run `python -m dashboard.export` once.')

nav.run()
