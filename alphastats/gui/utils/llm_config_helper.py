"""Helper functions to manage LLM model configurations in Streamlit GUI."""

from __future__ import annotations

import streamlit as st

from alphastats.gui.utils.llm_helper import OLLAMA_BASE_URL, llm_connection_test
from alphastats.gui.utils.state_keys import (
    LLMKeys,
    ModelKeys,
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


def get_test_status_icon(test_status: str) -> str:
    """Get icon for configuration test status.

    Args:
        test_status: The test status ("success", "failed", "not_tested")

    Returns:
        Icon string for the status

    """
    status_icons = {
        "success": "✅",
        "failed": "❌",
        "not_tested": "⚠️",
    }
    return status_icons.get(test_status, "❓")


def format_model_config_for_display(config: dict) -> str:
    """Format configuration for display in selectbox.

    Args:
        config: Configuration dictionary

    Returns:
        Formatted string with model name and status icon

    """
    test_status = config.get(ModelKeys.TEST_STATUS, "not_tested")
    icon = get_test_status_icon(test_status)

    return f"{config[ModelKeys.MODEL_NAME]} [max_tokens={config[ModelKeys.MAX_TOKENS]} base_url={config.get(ModelKeys.BASE_URL)} test status: {icon} {test_status}]"


def get_model_config_by_id(config_id: str) -> dict | None:
    """Retrieve configuration by ID.

    Args:
        config_id: Configuration UUID

    Returns:
        Configuration dictionary or None if not found

    """
    configurations = st.session_state.get(StateKeys.LLM_CONFIGURATIONS, [])
    return next((c for c in configurations if c["id"] == config_id), None)


def is_model_config_in_use(config_id: str) -> tuple[bool, list[str]]:
    """Check if configuration is used by any LLM chat.

    Args:
        config_id: Configuration UUID

    Returns:
        Tuple of (is_in_use, list_of_analysis_keys)

    """
    llm_chats = st.session_state.get(StateKeys.LLM_CHATS, {})
    analyses_using_config = []
    for analysis_key, chat_state in llm_chats.items():
        if chat_state.get(LLMKeys.LLM_CONFIGURATION_ID) == config_id:
            analyses_using_config.append(analysis_key)
    return len(analyses_using_config) > 0, analyses_using_config


@st.fragment
def add_model_config() -> None:
    """Display form to add a new model configuration."""
    import uuid

    with st.form("add_model_config", clear_on_submit=True):
        st.markdown("#### Add New Model Configuration")

        model_name = st.selectbox(
            "Select Model",
            options=Model.get_available_models(),
            help="Choose the LLM model to configure. You may add custom models.",
            accept_new_options=True,
        )
        st.info(
            "You can add custom models by typing their identifier (needs to be supported by LiteLLM) into the selection box and press 'Add:'."
            "Note: only the ones in the dropdown are officially supported and tested."
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
                ModelKeys.ID: str(uuid.uuid4()),
                ModelKeys.MODEL_NAME: model_name,
                ModelKeys.API_KEY: api_key.strip(),
                ModelKeys.BASE_URL: base_url.strip(),
                ModelKeys.MAX_TOKENS: max_tokens,
                ModelKeys.TEST_STATUS: "not_tested",
                "last_tested": None,
            }

            st.session_state[StateKeys.LLM_CONFIGURATIONS].append(config)
            st.success(f"✅ Added configuration for {model_name}")
            st.rerun()


@st.fragment
def display_model_configurations() -> None:
    """Display list of configured models with options to remove and test."""
    configurations = st.session_state.get(StateKeys.LLM_CONFIGURATIONS, [])

    if not configurations:
        return

    for config in configurations:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Model: {config[ModelKeys.MODEL_NAME]}**")

            st.markdown(f"API Key: {_mask_api_key(config.get(ModelKeys.API_KEY))}")

            st.markdown(f"Base URL: {config.get(ModelKeys.BASE_URL)}")

            st.markdown(f"Max Tokens: {config.get(ModelKeys.MAX_TOKENS):,}")

            test_status = config.get(ModelKeys.TEST_STATUS, "not_tested")
            if test_status == "success":
                st.success("✅ Connection test passed")
            elif test_status == "failed":
                st.error("❌ Connection test failed")
                if config.get("test_error"):
                    st.caption(f"Error: {config['test_error']}")

        with col2:
            if st.button("🔌 Test", key=f"test_{config[ModelKeys.ID]}"):
                with st.spinner("Testing connection..."):
                    error = llm_connection_test(
                        model_name=config[ModelKeys.MODEL_NAME],
                        base_url=config.get(ModelKeys.BASE_URL) or None,
                        api_key=config.get(ModelKeys.API_KEY) or None,
                    )
                    if error:
                        config[ModelKeys.TEST_STATUS] = "failed"
                        config["test_error"] = error
                        st.error(f"❌ Connection failed: {error}")
                    else:
                        config[ModelKeys.TEST_STATUS] = "success"
                        config["test_error"] = None
                        st.success("✅ Connection successful")
                    st.rerun()

            # Check if configuration is in use before allowing deletion
            in_use, analyses_using = is_model_config_in_use(config[ModelKeys.ID])

            if in_use:
                st.warning(
                    f"⚠️ Configuration is in use by {len(analyses_using)} analyses, which will break after removal of the model."
                )
            if st.button("🗑️ Remove", key=f"remove_{config[ModelKeys.ID]}"):
                st.session_state[StateKeys.LLM_CONFIGURATIONS].remove(config)
                st.success(f"✅ Removed configuration for {config['model_name']}")
                st.rerun()
