import streamlit as st
from st_cytoscape import cytoscape
import pandas as pd

import datetime

try:
    from alphastats.gui.utils.preprocessing_helper import main_preprocessing
    from alphastats.gui.utils.ui_helper import sidebar_info

except ModuleNotFoundError:
    from utils.preprocessing_helper import main_preprocessing
    from utils.ui_helper import sidebar_info


sidebar_info()

st.markdown("### Preprocessing")
st.markdown('Select either the predefined workflow where you can only enable/disable steps or create a custom workflow, that allows switching steps around.')


tab1, tab2  = st.tabs(["Predefined workflow", "Custom workflow"])

with tab1:
    if "workflow" not in st.session_state:
        st.session_state.workflow = ["remove contaminations", "remove samples", "subset data", "filter data completeness", "log2 transform", "normalization", "imputation"]

    elements = [
        {
            'data': {'id': i, 'label': label},
            "selectable": True
        } for i, label in enumerate(st.session_state.workflow)
    ]

    for i in range(len(st.session_state.workflow)-1):
        elements.append({'data': {'id': f'{i}_{i+1}', 'source': i, 'target': i+1}, 'selectable': False})

    stylesheet = [
    {"selector": "node", "style": {
        "label": "data(label)",
        'shape':'roundrectangle',
        "width": 200,
        "height": 60,
        "text-valign": "center",
        "text-halign": "center"
        }},
    {
        "selector": "edge",
        "style": {
            "width": 3,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
        },
    },
    ]

    selected = cytoscape(
        elements,
        stylesheet,
        layout='grid',
        selection_type='single',
        width=f'{len(st.session_state.workflow)*230}px',
        key="graph")

    main_preprocessing()

with tab2:
    "Custom workflows coming soon"
