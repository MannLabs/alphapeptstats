import streamlit as st
import pandas as pd

import datetime

try:
    from alphastats.gui.utils.preprocessing_helper import main_preprocessing, draw_predefined_workflow
    from alphastats.gui.utils.ui_helper import sidebar_info

except ModuleNotFoundError:
    from utils.preprocessing_helper import main_preprocessing, draw_predefined_workflow
    from utils.ui_helper import sidebar_info


sidebar_info()

if 'workflow' not in st.session_state:
    st.session_state.workflow = ["remove contaminations", "subset data", "log2 transform"]

st.markdown("### Preprocessing")
st.markdown('Select either the predefined workflow where you can only enable/disable steps or create a custom workflow, that allows switching steps around.')


tab1, tab2  = st.tabs(["Predefined workflow", "Custom workflow"])

with tab1:

    c1, c2 = st.columns([1, 1])

    with c2:
        main_preprocessing()

    with c1:
        st.write("### Flowchart of currenlty selected workflow:")

        selected_nodes = draw_predefined_workflow(st.session_state.workflow)

    # TODO: Add comparison plot of indensity distribution before and after preprocessing

with tab2:
    "Custom workflows coming soon"
