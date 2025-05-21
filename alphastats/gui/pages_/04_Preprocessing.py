import streamlit as st

from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.gui.utils.preprocessing_helper import (
    configure_preprocessing,
    display_preprocessing_info,
    draw_workflow,
    reset_preprocessing,
    run_preprocessing,
    update_workflow,
)
from alphastats.gui.utils.state_keys import StateKeys
from alphastats.gui.utils.state_utils import init_session_state
from alphastats.gui.utils.ui_helper import sidebar_info

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()


st.markdown("## Preprocessing")

if StateKeys.DATASET not in st.session_state:
    st.info("Import data first.")
    st.stop()

c1, _, c2 = st.columns([0.3, 0.1, 0.45])


dataset = st.session_state[StateKeys.DATASET]

with c1:
    st.write("##### Select preprocessing steps")
    settings = configure_preprocessing(dataset=dataset)
    new_workflow = update_workflow(settings)
    if new_workflow != st.session_state[StateKeys.WORKFLOW]:
        st.session_state[StateKeys.WORKFLOW] = new_workflow

    is_preprocessing_done = dataset.preprocessing_info[
        PreprocessingStateKeys.PREPROCESSING_DONE
    ]

    if is_preprocessing_done:
        st.success("Preprocessing finished successfully!", icon="✅")

    c11, c12 = st.columns([1, 1])
    if c11.button(
        "Run preprocessing", key="_run_preprocessing", disabled=is_preprocessing_done
    ):
        run_preprocessing(settings, dataset)
        st.rerun()

    if c12.button(
        "❌ Reset preprocessing",
        key="_reset_preprocessing",
        disabled=not is_preprocessing_done,
    ):
        reset_preprocessing(dataset)
        st.rerun()

with c2:
    selected_nodes = draw_workflow(st.session_state[StateKeys.WORKFLOW])

    st.markdown("##### Current preprocessing status")
    display_preprocessing_info(dataset.preprocessing_info)

# TODO add help to individual steps with more info
# TODO: Add comparison plot of intensity distribution before and after preprocessing
