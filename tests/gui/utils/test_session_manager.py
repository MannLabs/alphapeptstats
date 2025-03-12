"""Tests for the SessionManager class."""

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.runtime.state import SessionStateProxy

from alphastats.gui.utils.session_manager import SessionManager
from alphastats.gui.utils.state_keys import StateKeys

EXPECTED_STATE = {
    StateKeys.USER_SESSION_ID: "some_id",
    StateKeys.DATASET: "some_dataset",
}


@pytest.fixture
def mock_session_state():
    """Return a mock session state."""

    class MockSessionState(SessionStateProxy):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to_dict(self):
            return self.__dict__

    return MockSessionState(ignored_key="ignored_value", **EXPECTED_STATE)


def test_copy(mock_session_state):
    """Test that the session state is copied correctly."""

    already_present_dict = {"already_present_key": "already_present_value"}  # preserved
    target = already_present_dict | {
        StateKeys.DATASET: "some_other_dataset"
    }  # overwritten

    session_manager = SessionManager()

    # when
    session_manager._copy(mock_session_state.to_dict(), target)
    assert target == EXPECTED_STATE | already_present_dict


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

    file_path = session_manager.save(mock_session_state)

    assert Path(file_path).exists()


@patch("alphastats.gui.utils.session_manager.empty_session_state")
@patch("alphastats.gui.utils.session_manager.init_session_state")
def test_save_and_load(
    mock_empty_session_state, mock_init_session_state, mock_session_state, tmp_path
):
    """Test that the session state is saved and loaded correctly."""

    session_manager = SessionManager(str(tmp_path))

    file_path = session_manager.save(mock_session_state)

    target = {}

    # when
    session_manager.load(Path(file_path).name, target)

    assert target == EXPECTED_STATE
    mock_empty_session_state.assert_called_once()
    mock_init_session_state.assert_called_once()
