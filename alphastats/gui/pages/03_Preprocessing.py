import pandas as pd
import streamlit as st

from alphastats.gui.utils.preprocessing_helper import (
    PREPROCESSING_STEPS,
    configure_preprocessing,
    display_preprocessing_info,
    draw_workflow,
    reset_preprocessing,
    run_preprocessing,
    update_workflow,
)
from alphastats.gui.utils.ui_helper import StateKeys, init_session_state, sidebar_info

init_session_state()
sidebar_info()

if StateKeys.WORKFLOW not in st.session_state:
    st.session_state[StateKeys.WORKFLOW] = [
        PREPROCESSING_STEPS.REMOVE_CONTAMINATIONS,
        PREPROCESSING_STEPS.SUBSET,
        PREPROCESSING_STEPS.REPLACE_ZERO,
        PREPROCESSING_STEPS.LOG2_TRANSFORM,
    ]

st.markdown("### Preprocessing")
c1, c2 = st.columns([1, 1])

with c2:
    if StateKeys.DATASET in st.session_state:
        settings = configure_preprocessing(dataset=st.session_state[StateKeys.DATASET])
        new_workflow = update_workflow(settings)
        if new_workflow != st.session_state[StateKeys.WORKFLOW]:
            st.session_state[StateKeys.WORKFLOW] = new_workflow

with c1:
    st.write("#### Flowchart of preprocessing workflow:")

    selected_nodes = draw_workflow(st.session_state[StateKeys.WORKFLOW])

    if StateKeys.DATASET not in st.session_state:
        st.info("Import data first to configure and run preprocessing")

    else:
        c11, c12 = st.columns([1, 1])
        if c11.button("Run preprocessing", key="_run_preprocessing"):
            run_preprocessing(settings, st.session_state[StateKeys.DATASET])
            # TODO show more info about the preprocessing steps
            display_preprocessing_info(
                st.session_state[StateKeys.DATASET].preprocessing_info
            )

        if c12.button("Reset all Preprocessing steps", key="_reset_preprocessing"):
            reset_preprocessing(st.session_state[StateKeys.DATASET])
            display_preprocessing_info(
                st.session_state[StateKeys.DATASET].preprocessing_info
            )

# TODO: Add comparison plot of intensity distribution before and after preprocessing
