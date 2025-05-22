"""Tests for the SessionManager class."""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from streamlit.runtime.state import SessionStateProxy

from alphastats.gui.utils.session_manager import SessionManager
from alphastats.gui.utils.state_keys import LLMKeys, StateKeys

EXPECTED_STATE = {
    StateKeys.USER_SESSION_ID: "some_id",
    StateKeys.DATASET: "some_dataset",
}


class MockSessionState(SessionStateProxy):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return self.__dict__


@pytest.fixture
def mock_session_state():
    """Return a mock session state."""

    return MockSessionState(
        ignored_key="ignored_value",
        **EXPECTED_STATE,
    )


@pytest.fixture
def mock_llm_integration():
    return MagicMock()


@pytest.fixture
def mock_session_state_with_llm(mock_llm_integration):
    """Return a mock session state."""

    return MockSessionState(
        ignored_key="ignored_value",
        **EXPECTED_STATE,
        **{
            StateKeys.LLM_CHATS: {
                "some_datetime": {LLMKeys.LLM_INTEGRATION: mock_llm_integration}
            }
        },
    )


def test_copy(mock_session_state):
    """Test that the session state is copied correctly."""

    already_present_dict = {"already_present_key": "already_present_value"}  # preserved
    target = {
        **already_present_dict,
        StateKeys.DATASET: "some_other_dataset",  # overwritten
    }

    session_manager = SessionManager()

    # when
    session_manager._clean_copy(mock_session_state.to_dict(), target)
    assert target == {**EXPECTED_STATE, **already_present_dict}


def test_get_saved_sessions(tmp_path):
    """Test that the saved sessions are returned in reverse order."""

    (tmp_path / "session_20230101-000000.cpkl").touch()
    (tmp_path / "session_20230102-000000.cpkl").touch()

    # when
    sessions = SessionManager.get_saved_sessions(tmp_path)
    assert sessions == ["session_20230102-000000.cpkl", "session_20230101-000000.cpkl"]


def test_save(mock_session_state, tmp_path):
    """Test that the session state is saved to a file."""

    session_manager = SessionManager(str(tmp_path))

    file_path = session_manager.save(mock_session_state, "some_name")

    assert Path(file_path).exists()
    assert "some_name" in file_path


@patch("alphastats.gui.utils.session_manager.empty_session_state")
@patch("alphastats.gui.utils.session_manager.init_session_state")
def test_save_and_load(
    mock_empty_session_state, mock_init_session_state, mock_session_state, tmp_path
):
    """Test that the session state is saved and loaded correctly."""

    session_manager = SessionManager(str(tmp_path))

    file_path = session_manager.save(mock_session_state)

    llm_state = {
        StateKeys.MODEL_NAME: "some_model_name",
        StateKeys.OPENAI_API_KEY: "some_key",  # pragma: allowlist secret
        StateKeys.BASE_URL: "some_url",
        "some_key_that_will_get_overwritten": "some_value",
    }

    session_state_before_load = llm_state.copy()

    # when
    session_manager.load(Path(file_path).name, session_state_before_load)

    assert session_state_before_load == {**EXPECTED_STATE, **llm_state}
    mock_empty_session_state.assert_called_once()
    mock_init_session_state.assert_called_once()


@patch("alphastats.gui.utils.session_manager.empty_session_state")
@patch("alphastats.gui.utils.session_manager.init_session_state")
@patch("alphastats.gui.utils.session_manager.LLMClientWrapper")
def test_save_and_load_with_llm(
    mock_client_wrapper,
    mock_init_session_state,
    mock_empty_session_state,
    mock_session_state_with_llm,
    tmp_path,
):
    """Test that the session state is saved and loaded correctly when an LLM chat is present."""

    session_manager = SessionManager(str(tmp_path))

    file_path = session_manager.save(mock_session_state_with_llm)

    llm_state = {
        StateKeys.MODEL_NAME: "some_model_name",
        StateKeys.OPENAI_API_KEY: "some_key",  # pragma: allowlist secret
        StateKeys.BASE_URL: "some_url",
        "some_key_that_will_get_overwritten": "some_value",
    }

    session_state_before_load = llm_state.copy()

    # when
    session_manager.load(Path(file_path).name, session_state_before_load)

    assert session_state_before_load == {
        **EXPECTED_STATE,
        **llm_state,
        **{
            StateKeys.LLM_CHATS: {
                "some_datetime": {
                    LLMKeys.LLM_INTEGRATION: mock.ANY,
                    LLMKeys.BASE_URL: "some_url",
                    LLMKeys.MODEL_NAME: "some_model_name",
                }
            }
        },
    }

    mock_empty_session_state.assert_called_once()
    mock_init_session_state.assert_called_once()
    mock_client_wrapper.assert_called_once_with(
        model_name="some_model_name",
        api_key="some_key",  # pragma: allowlist secret
        base_url="some_url",
    )
