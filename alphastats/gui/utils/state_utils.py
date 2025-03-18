"""Functions to initialize and empty the session state."""

import uuid

import streamlit as st

from alphastats.gui.utils.state_keys import DefaultStates, StateKeys


def empty_session_state() -> None:
    """Remove all variables to avoid conflicts."""
    for key in st.session_state:
        del st.session_state[key]
    st.empty()


def init_session_state() -> None:
    """Initialize the session state if not done yet."""
    if StateKeys.USER_SESSION_ID not in st.session_state:
        st.session_state[StateKeys.USER_SESSION_ID] = str(uuid.uuid4())

    if StateKeys.ORGANISM not in st.session_state:
        st.session_state[StateKeys.ORGANISM] = 9606  # human

    if StateKeys.WORKFLOW not in st.session_state:
        st.session_state[StateKeys.WORKFLOW] = DefaultStates.WORKFLOW.copy()

    if StateKeys.SAVED_ANALYSES not in st.session_state:
        st.session_state[StateKeys.SAVED_ANALYSES] = {}

    if StateKeys.LLM_CHATS not in st.session_state:
        st.session_state[StateKeys.LLM_CHATS] = {}

    if StateKeys.ANNOTATION_STORE not in st.session_state:
        st.session_state[StateKeys.ANNOTATION_STORE] = {}

    if StateKeys.MAX_TOKENS not in st.session_state:
        st.session_state[StateKeys.MAX_TOKENS] = 10000
