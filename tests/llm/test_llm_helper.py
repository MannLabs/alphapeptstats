from unittest.mock import patch

import pytest

from alphastats.gui.utils.llm_helper import (
    get_display_proteins_html,
    llm_connection_test,
    set_api_key,
)
from alphastats.gui.utils.ui_helper import StateKeys


@pytest.fixture
def mock_streamlit():
    """Fixture to mock streamlit module."""
    with patch("streamlit.info") as mock_info, patch(
        "streamlit.error"
    ) as mock_error, patch("streamlit.success") as mock_success, patch(
        "streamlit.session_state", {}
    ) as mock_session_state:
        yield {
            "info": mock_info,
            "error": mock_error,
            "success": mock_success,
            "session_state": mock_session_state,
        }


def test_display_proteins_upregulated(mock_streamlit):
    """Test displaying upregulated proteins."""
    protein_ids = ["P12345", "Q67890"]
    result = get_display_proteins_html(protein_ids, is_upregulated=True)

    expected_html = (
        "<ul><a href = https://www.uniprot.org/uniprotkb?query=P12345>"
        + '<li style="color: green;">P12345</li></a>'
        + "<a href = https://www.uniprot.org/uniprotkb?query=Q67890>"
        + '<li style="color: green;">Q67890</li></a></ul>'
    )

    assert result == expected_html


def test_display_proteins_downregulated(mock_streamlit):
    """Test displaying downregulated proteins."""
    protein_ids = ["P12345"]
    result = get_display_proteins_html(protein_ids, is_upregulated=False)

    expected_html = (
        "<ul><a href = https://www.uniprot.org/uniprotkb?query=P12345>"
        + '<li style="color: red;">P12345</li></a></ul>'
    )

    assert result == expected_html


def test_display_proteins_empty_list(mock_streamlit):
    """Test displaying empty protein list."""
    assert get_display_proteins_html([], is_upregulated=True) == "<ul></ul>"


@pytest.mark.parametrize(
    "api_key,expected_message",
    [
        ("abc123xyz", "OpenAI API key set: abc***xyz"),
        (
            None,
            "Please enter an OpenAI key or provide it in a secrets.toml file in the alphastats/gui/.streamlit directory like `openai_api_key = <key>`",
        ),
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
def test_set_api_key_from_secrets(mock_exists, mock_st_secrets, mock_streamlit):
    """Test loading API key from secrets.toml."""
    mock_exists.return_value = True

    mock_st_secrets.__getitem__.return_value = (
        "test_secret_key"  # pragma: allowlist secret
    )

    set_api_key()

    mock_streamlit["info"].assert_called_with(
        "OpenAI API key loaded from secrets.toml."
    )
    assert (
        mock_streamlit["session_state"][StateKeys.OPENAI_API_KEY]
        == "test_secret_key"  # pragma: allowlist secret
    )


@patch("pathlib.Path.exists")
def test_set_api_key_missing_secrets(mock_exists, mock_streamlit):
    """Test handling missing secrets.toml."""
    mock_exists.return_value = False

    set_api_key()

    mock_streamlit["info"].assert_called_with(
        "Please enter an OpenAI key or provide it in a secrets.toml file in the "
        "alphastats/gui/.streamlit directory like `openai_api_key = <key>`"
    )


@patch("alphastats.gui.utils.llm_helper.LLMIntegration")
def test_llm_connection_test_success(mock_llm):
    """Test successful LLM connection."""
    assert llm_connection_test("some_model") is None

    mock_llm.assert_called_once_with(
        "some_model", base_url=None, api_key=None, load_tools=False
    )


@patch("alphastats.gui.utils.llm_helper.LLMIntegration")
def test_llm_connection_test_failure(mock_llm, mock_streamlit):
    """Test failed LLM connection."""
    mock_llm.return_value.chat_completion.side_effect = ValueError("API Error")

    assert llm_connection_test("some_model") == "API Error"

    mock_llm.assert_called_once_with(
        "some_model", base_url=None, api_key=None, load_tools=False
    )
