"""Functions to initialize and empty the session state."""

import uuid

import streamlit as st

from alphastats.gui.utils.state_keys import DefaultStates, StateKeys
from alphastats.llm.llm_integration import Models


def empty_session_state() -> None:
    """Remove all variables to avoid conflicts."""
    for key in st.session_state:
        del st.session_state[key]
    st.empty()


INIT_STATES = {
    StateKeys.USER_SESSION_ID: str(uuid.uuid4()),
    StateKeys.ORGANISM: 9606,  # human
    StateKeys.WORKFLOW: DefaultStates.WORKFLOW.copy(),
    StateKeys.SAVED_ANALYSES: {},
    StateKeys.LLM_CHATS: {},
    StateKeys.ANNOTATION_STORE: {},
    StateKeys.MAX_TOKENS: 10000,
    StateKeys.MODEL_NAME: (
        Models.GPT4O
    ),  # TDOO: change to None: this is just for convenience now
    StateKeys.SELECTED_ANALYSIS: None,
    StateKeys.PROMPT_EXPERIMENTAL_DESIGN: None,
    StateKeys.PROMPT_PROTEIN_DATA: None,
    StateKeys.PROMPT_INSTRUCTIONS: None,
}


def init_session_state() -> None:
    """Initialize the session state if not done yet."""
    for key, value in INIT_STATES.items():
        if key not in st.session_state:
            st.session_state[key] = value
