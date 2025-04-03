"""Functions to initialize and empty the session state."""

import streamlit as st

from alphastats.gui.utils.state_keys import INIT_STATES


def empty_session_state() -> None:
    """Remove all variables to avoid conflicts."""
    for key in st.session_state:
        del st.session_state[key]
    st.empty()


def init_session_state() -> None:
    """Initialize the session state if not done yet."""
    for key, value in INIT_STATES.items():
        if key not in st.session_state:
            st.session_state[key] = value
