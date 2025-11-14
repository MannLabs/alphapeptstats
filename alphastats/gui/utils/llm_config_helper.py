"""Helper functions to manage LLM model configurations in Streamlit GUI."""

from __future__ import annotations

import streamlit as st

from alphastats.gui.utils.llm_helper import OLLAMA_BASE_URL, llm_connection_test
from alphastats.gui.utils.state_keys import (
    LLMKeys,
    StateKeys,
)
from alphastats.llm.llm_integration import (
    Model,
)


def _mask_api_key(api_key: str | None) -> str:
    """Mask an API key for display, showing first 3 and last 3 characters.

    Args:
        api_key: The API key to mask

    Returns:
        Masked API key string

    """
    if not api_key or len(api_key) < 6:  # noqa: PLR2004
        return "*" * len(api_key)
    return f"{api_key[:3]}{(len(api_key)-6)*'*'}{api_key[-3:]}"


@st.fragment
def add_model_config() -> None:
    """Display form to add a new model configuration."""
    import uuid

    with st.form("add_model_config", clear_on_submit=True):
        st.markdown("#### Add New Model Configuration")

        model_name = st.selectbox(
            "Select Model",
            options=Model.get_available_models(),
            help="Choose the LLM model to configure",
        )

        model = Model(model_name)
        requires_api_key = model.requires_api_key()
        is_vertex_model = model_name.startswith("vertex")

        # Always show API key field with appropriate label and help text
        if is_vertex_model:
            api_key_label = "Vertex Project ID"  # pragma: allowlist secret
            api_key_help = "Enter your Google Cloud Project ID for Vertex AI"  # pragma: allowlist secret
        else:
            api_key_label = "API Key"  # pragma: allowlist secret
            api_key_help = (
                "Enter the API key for this model"
                + (  # pragma: allowlist secret
                    "" if requires_api_key else " (leave empty if not needed)"
                )
            )

        api_key = st.text_input(
            api_key_label,
            type="password",
            help=api_key_help,
        )

        # Always show base URL field with smart defaults
        if is_vertex_model:
            base_url_label = "Vertex Location"
            base_url_help = "Enter the Google Cloud region (e.g., us-central1)"
            default_url = "us-central1"
        else:
            base_url_label = "Base URL"
            base_url_help = (
                "Enter the base URL for the API endpoint (leave empty to use default)"
            )
            default_url = OLLAMA_BASE_URL if "ollama" in model_name.lower() else ""

        base_url = st.text_input(
            base_url_label,
            value=default_url,
            help=base_url_help,
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=1000,
            max_value=200000,
            value=10000,
            step=1000,
            help="Maximum number of tokens for context window",
        )

        submitted = st.form_submit_button("➕ Add Configuration")  # noqa: RUF001

        if submitted:
            if requires_api_key and not api_key.strip():
                st.error("API key is required for this model")
                return

            config = {
                "id": str(uuid.uuid4()),
                "model_name": model_name,
                "api_key": api_key.strip(),
                "base_url": base_url.strip(),
                "max_tokens": max_tokens,
                "test_status": "not_tested",
                "last_tested": None,
            }

            st.session_state[StateKeys.LLM_CONFIGURATIONS].append(config)
            st.success(f"✅ Added configuration for {model_name}")
            st.rerun()


@st.fragment
def display_model_configurations() -> None:  # noqa: C901 too complex
    """Display list of configured models with options to remove and test."""
    configurations = st.session_state.get(StateKeys.LLM_CONFIGURATIONS, [])
    st.write(configurations)
    if not configurations:
        return

    for config in configurations:
        with st.expander(f"🤖 {config['model_name']}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Model:** {config['model_name']}")

                st.markdown(f"**API Key:** {_mask_api_key(config.get('api_key'))}")

                st.markdown(f"**Base URL:** {config.get('base_url')}")

                st.markdown(f"**Max Tokens:** {config.get('max_tokens'):,}")

                test_status = config.get("test_status", "not_tested")
                if test_status == "success":
                    st.success("✅ Connection test passed")
                elif test_status == "failed":
                    st.error("❌ Connection test failed")
                    if config.get("test_error"):
                        st.caption(f"Error: {config['test_error']}")

            with col2:
                if st.button("🔌 Test", key=f"test_{config['id']}"):
                    with st.spinner("Testing connection..."):
                        error = llm_connection_test(
                            model_name=config["model_name"],
                            base_url=config.get("base_url") or None,
                            api_key=config.get("api_key") or None,
                        )
                        if error:
                            config["test_status"] = "failed"
                            config["test_error"] = error
                            st.error(f"❌ Connection failed: {error}")
                        else:
                            config["test_status"] = "success"
                            config["test_error"] = None
                            st.success("✅ Connection successful")
                        st.rerun()

                if st.button("🗑️ Remove", key=f"remove_{config['id']}"):
                    st.session_state[StateKeys.LLM_CONFIGURATIONS].remove(config)

                    # Check if this config is used in any analysis
                    used_in_analyses = []
                    for analysis_key, chat in st.session_state[
                        StateKeys.LLM_CHATS
                    ].items():
                        if chat.get(LLMKeys.LLM_CONFIGURATION_ID) == config["id"]:
                            used_in_analyses.append(analysis_key)

                    if used_in_analyses:
                        st.warning(
                            f"⚠️ This configuration was used in {len(used_in_analyses)} analysis/analyses. "
                            "Those analyses will need to be reconfigured."
                        )

                    st.success(f"Removed configuration for {config['model_name']}")
                    st.rerun()
