"""Functions to initialize and empty the session state."""

import uuid

import streamlit as st

from alphastats.gui.utils.state_keys import DefaultStates, StateKeys


def empty_session_state() -> None:
    """Remove all variables to avoid conflicts."""
    for key in st.session_state:
        del st.session_state[key]
    st.empty()


def init_session_state() -> None:  # noqa: C901
    """Initialize the session state if not done yet."""
    if StateKeys.USER_SESSION_ID not in st.session_state:
        st.session_state[StateKeys.USER_SESSION_ID] = str(uuid.uuid4())

    if StateKeys.ORGANISM not in st.session_state:
        st.session_state[StateKeys.ORGANISM] = 9606  # human

    if StateKeys.WORKFLOW not in st.session_state:
        st.session_state[StateKeys.WORKFLOW] = DefaultStates.WORKFLOW.copy()

    if StateKeys.ANALYSIS_LIST not in st.session_state:
        st.session_state[StateKeys.ANALYSIS_LIST] = []

    if StateKeys.LLM_INTEGRATION not in st.session_state:
        st.session_state[StateKeys.LLM_INTEGRATION] = {}

    if StateKeys.ANNOTATION_STORE not in st.session_state:
        st.session_state[StateKeys.ANNOTATION_STORE] = {}

    if StateKeys.SELECTED_GENES_UP not in st.session_state:
        st.session_state[StateKeys.SELECTED_GENES_UP] = None

    if StateKeys.SELECTED_GENES_DOWN not in st.session_state:
        st.session_state[StateKeys.SELECTED_GENES_DOWN] = None

    if StateKeys.SELECTED_UNIPROT_FIELDS not in st.session_state:
        st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = (
            DefaultStates.SELECTED_UNIPROT_FIELDS.copy()
        )

    if StateKeys.MAX_TOKENS not in st.session_state:
        st.session_state[StateKeys.MAX_TOKENS] = 10000

    if StateKeys.RECENT_CHAT_WARNINGS not in st.session_state:
        st.session_state[StateKeys.RECENT_CHAT_WARNINGS] = []
