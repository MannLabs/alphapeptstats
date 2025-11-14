"""Functions to initialize and empty the session state."""

import uuid
from copy import deepcopy

import streamlit as st

from alphastats.gui.utils.state_keys import DefaultStates, StateKeys

keys_to_preserve = [StateKeys.LLM_CONFIGURATIONS]


def empty_session_state() -> None:
    """Remove all variables to avoid conflicts."""
    for key in st.session_state:
        if key not in keys_to_preserve:
            del st.session_state[key]
    st.empty()


INIT_STATES = {
    StateKeys.USER_SESSION_ID: str(uuid.uuid4()),
    StateKeys.WORKFLOW: DefaultStates.WORKFLOW.copy(),
    StateKeys.SAVED_ANALYSES: {},
    StateKeys.LLM_CHATS: {},
    StateKeys.LLM_CONFIGURATIONS: [],
    StateKeys.ANNOTATION_STORE: {},
    StateKeys.SELECTED_ANALYSIS: None,
    StateKeys.PROMPT_EXPERIMENTAL_DESIGN: None,
    StateKeys.PROMPT_PROTEIN_DATA: None,
    StateKeys.PROMPT_INSTRUCTIONS: None,
    StateKeys.ENRICHMENT_COLUMNS: [],
}


def init_session_state() -> None:
    """Initialize the session state if not done yet."""
    for key, value in INIT_STATES.items():
        if key not in st.session_state:
            try:
                st.session_state[key] = deepcopy(value)
            except AttributeError:
                st.session_state[key] = value
