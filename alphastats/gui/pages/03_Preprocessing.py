import streamlit as st
import pandas as pd


from alphastats.gui.utils.preprocessing_helper import (
    draw_workflow,
    configure_preprocessing,
    display_preprocessing_info,
    update_workflow,
    run_preprocessing,
    reset_preprocessing,
    PREPROCESSING_STEPS,
)
from alphastats.gui.utils.ui_helper import sidebar_info


sidebar_info()

if "workflow" not in st.session_state:
    st.session_state["workflow"] = [
        PREPROCESSING_STEPS.REMOVE_CONTAMINATIONS,
        PREPROCESSING_STEPS.SUBSET,
        PREPROCESSING_STEPS.LOG2_TRANSFORM,
    ]

st.markdown("### Preprocessing")
c1, c2 = st.columns([1, 1])

with c2:
    if "dataset" not in st.session_state:
        settings = {k: True for k in st.session_state.workflow}
    else:
        settings = configure_preprocessing(dataset=st.session_state["dataset"])

    new_workflow = update_workflow(settings)

    if new_workflow != st.session_state.workflow:
        st.session_state.workflow = new_workflow

with c1:
    st.write("#### Flowchart of preprocessing workflow:")

    selected_nodes = draw_workflow(st.session_state.workflow)

    if "dataset" not in st.session_state:
        st.info("Import data first to configure and run preprocessing")

    else:
        c11, c12 = st.columns([1, 1])

        with c11:
            if st.button("Run preprocessing"):
                run_preprocessing(settings, st.session_state["dataset"])
                display_preprocessing_info(
                    st.session_state["dataset"].preprocessing_info
                )

        with c12:
            if st.button("Reset all Preprocessing steps"):
                reset_preprocessing(st.session_state["dataset"])
                display_preprocessing_info(
                    st.session_state["dataset"].preprocessing_info
                )

# TODO: Add comparison plot of indensity distribution before and after preprocessing
