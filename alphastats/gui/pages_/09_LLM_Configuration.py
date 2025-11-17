import streamlit as st

from alphastats.gui.utils.llm_config_helper import (
    add_model_config,
    display_model_configurations,
)
from alphastats.gui.utils.state_keys import StateKeys
from alphastats.gui.utils.state_utils import init_session_state
from alphastats.gui.utils.ui_helper import sidebar_info

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()

st.markdown("## LLM Configuration")

st.info(
    """
    Configure multiple LLM models for use in analysis interpretation.
    Each model configuration can have its own API key and settings.

    Configured models will be available for selection on the LLM interpretation page.
    """
)

configured_models = st.session_state.get(StateKeys.LLM_CONFIGURATIONS, [])

if not configured_models:
    st.markdown("### No models configured yet")
    st.write("Click the button below to add your first model configuration.")
else:
    st.markdown(f"### Configured Models ({len(configured_models)})")
    display_model_configurations()

st.markdown("---")

add_model_config()
