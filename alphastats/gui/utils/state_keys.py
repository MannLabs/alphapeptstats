"""Keys and helper functions for the session state."""

import uuid

import streamlit as st
from dataset.keys import ConstantsClass
from gui.utils.ui_helper import DefaultStates


class StateKeys(metaclass=ConstantsClass):
    """Keys for accessing the session state."""

    USER_SESSION_ID = "user_session_id"
    DATASET = "dataset"

    WORKFLOW = "workflow"

    ANALYSIS_LIST = "analysis_list"

    # LLM
    OPENAI_API_KEY = "openai_api_key"  # pragma: allowlist secret
    MODEL_NAME = "model_name"
    LLM_INPUT = "llm_input"
    LLM_INTEGRATION = "llm_integration"
    ANNOTATION_STORE = "annotation_store"
    SELECTED_GENES_UP = "selected_genes_up"
    SELECTED_GENES_DOWN = "selected_genes_down"
    SELECTED_UNIPROT_FIELDS = "selected_uniprot_fields"
    MAX_TOKENS = "max_tokens"
    RECENT_CHAT_WARNINGS = "recent_chat_warnings"

    ORGANISM = "organism"  # TODO: this is essentially a constant


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
