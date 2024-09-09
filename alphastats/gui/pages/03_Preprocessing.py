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
    available_steps = ["remove contaminations", "remove samples", "subset data", "filter data completeness", "log2 transform", "normalization", "imputation"]
    if "workflow" not in st.session_state:
        st.session_state.workflow = ["remove contaminations", "subset data", "log2 transform"]

    elements = [
        {
            'group': 'nodes',
            'data': {
                'id': i,
                'label': label,
            },
            "selectable": True,
            "classes": ['active'] if label in st.session_state.workflow else ['inactive']
        } for i, label in enumerate(available_steps)
    ]

    for label1, label2 in zip(st.session_state.workflow[:-1], st.session_state.workflow[1:]):
        i = available_steps.index(label1)
        j = available_steps.index(label2)
        elements.append({'group':'edges', 'data': {'id': f'{i}_{j}', 'source': i, 'target': j}, 'selectable': False})

    stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            'shape':'roundrectangle',
            "width": 200,
            "height": 60,
            "text-valign": "center",
            "text-halign": "center"
        },
    },
    {
        "selector": "node.active",
        "style": {
            "background-color": 'lightgreen',
            "opacity": '1',
        },
    },
    {
        "selector": "node.inactive",
        "style": {
            "background-color": 'grey',
            "opacity": '0.3',
        },
    },
    {
        "selector": "node.selected",
        "style": {
            "background-color": 'red'
        },
    },
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
        layout={'name': 'grid', 'rows': 1},
        selection_type='single',
        width=f'{len(available_steps)*230}px',
        key="graph")

    main_preprocessing()

    # TODO: Add comparison plot of indensity distribution before and after preprocessing

with tab2:
    "Custom workflows coming soon"
