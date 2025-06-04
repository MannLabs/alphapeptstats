from unittest import skip
from unittest.mock import patch

import pytest

from alphastats.gui.utils.llm_helper import (
    get_display_proteins_html,
    llm_connection_test,
    set_api_key,
)
from alphastats.gui.utils.state_keys import StateKeys


@pytest.fixture
def mock_streamlit():
    """Fixture to mock streamlit module."""
    with (
        patch("streamlit.info") as mock_info,
        patch("streamlit.error") as mock_error,
        patch("streamlit.success") as mock_success,
        patch("streamlit.toast") as mock_toast,
        patch("streamlit.session_state", {}) as mock_session_state,
    ):
        yield {
            "info": mock_info,
            "error": mock_error,
            "success": mock_success,
            "toast": mock_toast,
            "session_state": mock_session_state,
        }


@patch("streamlit.session_state", new_callable=dict)
def test_display_proteins_upregulated(mock_session_state):
    """Test displaying upregulated proteins."""
    annotation_store = {
        "P12345": {"primaryAccession": "P12345"},
        "Q67890;P56789": {"primaryAccession": "Q67890"},
    }
    feature_to_repr_map = {"P12345": "P12345", "Q67890;P56789": "Q67890"}
    protein_ids = ["P12345", "Q67890;P56789"]
    result = get_display_proteins_html(
        protein_ids,
        is_upregulated=True,
        annotation_store=annotation_store,
        feature_to_repr_map=feature_to_repr_map,
    )

    expected_html = (
        "<ul><a href = https://www.uniprot.org/uniprotkb/P12345>"
        + '<li style="color: green;">P12345</li></a>'
        + "<a href = https://www.uniprot.org/uniprotkb/Q67890>"
        + '<li style="color: green;">Q67890</li></a></ul>'
    )

    assert result == expected_html


@patch("streamlit.session_state", new_callable=dict)
def test_display_proteins_downregulated(mock_session_state):
    """Test displaying downregulated proteins."""
    annotation_store = {"P12345": {"primaryAccession": "P12345"}}
    feature_to_repr_map = {"P12345": "P12345", "Q67890;P56789": "Q67890"}
    protein_ids = ["P12345"]
    result = get_display_proteins_html(
        protein_ids,
        is_upregulated=False,
        annotation_store=annotation_store,
        feature_to_repr_map=feature_to_repr_map,
    )

    expected_html = (
        "<ul><a href = https://www.uniprot.org/uniprotkb/P12345>"
        + '<li style="color: red;">P12345</li></a></ul>'
    )

    assert result == expected_html


def test_display_proteins_empty_list(mock_streamlit):
    """Test displaying empty protein list."""
    assert (
        get_display_proteins_html(
            [], is_upregulated=True, annotation_store={}, feature_to_repr_map={}
        )
        == "<ul></ul>"
    )


@pytest.mark.parametrize(
    "api_key,expected_message",
    [
        ("abc123xyz", "API key set: 'abc***xyz'"),
        # (
        #     None,
        #     "Please enter an LLM API key or provide it in a secrets.toml file in the alphastats/gui/.streamlit directory like `api_key = <key>`",
        # ),
    ],
)
def test_set_api_key_direct(mock_streamlit, api_key, expected_message):
    """Test setting API key directly."""
    set_api_key(api_key)

    if api_key:
        mock_streamlit["info"].assert_called_once_with(expected_message)
        assert mock_streamlit["session_state"][StateKeys.OPENAI_API_KEY] == api_key
    else:
        mock_streamlit["info"].assert_called_with(expected_message)


@patch("streamlit.secrets")
@patch("pathlib.Path.exists")
@skip("functionality was commented out")
def test_set_api_key_from_secrets(mock_exists, mock_st_secrets, mock_streamlit):
    """Test loading API key from secrets.toml."""
    mock_exists.return_value = True

    mock_st_secrets.__getitem__.return_value = (
        "test_secret_key"  # pragma: allowlist secret
    )

    set_api_key()

    mock_streamlit["toast"].assert_called_with(
        "API key loaded from secrets.toml.", icon="✅"
    )
    assert (
        mock_streamlit["session_state"][StateKeys.OPENAI_API_KEY]
        == "test_secret_key"  # pragma: allowlist secret
    )


@patch("pathlib.Path.exists")
@skip("functionality was commented out")
def test_set_api_key_missing_secrets(mock_exists, mock_streamlit):
    """Test handling missing secrets.toml."""
    mock_exists.return_value = False

    set_api_key()

    mock_streamlit["info"].assert_called_with(
        "Please enter an LLM API key or provide it in a secrets.toml file in the "
        "alphastats/gui/.streamlit directory like `api_key = <key>`"
    )


@patch("alphastats.gui.utils.llm_helper.LLMIntegration")
@patch("alphastats.gui.utils.llm_helper.LLMClientWrapper")
def test_llm_connection_test_success(mock_client_wrapper, mock_llm):
    """Test successful LLM connection."""
    assert llm_connection_test("some_model") is None

    mock_client_wrapper.assert_called_once_with(
        "some_model", base_url=None, api_key=None
    )
    mock_llm.assert_called_once_with(mock_client_wrapper.return_value, load_tools=False)


@patch("alphastats.gui.utils.llm_helper.LLMIntegration")
@patch("alphastats.gui.utils.llm_helper.LLMClientWrapper")
def test_llm_connection_test_failure(mock_client_wrapper, mock_llm, mock_streamlit):
    """Test failed LLM connection."""
    mock_llm.return_value.chat_completion.side_effect = ValueError("API Error")

    assert llm_connection_test("some_model") == "API Error"

    mock_client_wrapper.assert_called_once_with(
        "some_model", base_url=None, api_key=None
    )
    mock_llm.assert_called_once_with(mock_client_wrapper.return_value, load_tools=False)
