from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from alphastats.gui.utils.state_keys import StateKeys

from .conftest import APP_FOLDER, create_dataset_alphapept

TESTED_PAGE = f"{APP_FOLDER}/pages_/04_Preprocessing.py"


@patch("streamlit.page_link")  # to avoid errors with the relative paths
def test_page_04_loads_without_input(mock_page_link: MagicMock):
    """Test if the page loads without any input and inititalizes the session state with the correct values."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    assert not at.exception


def test_page_04_loads_with_input():
    """Test if the page loads with input and serves the processing interface correctly."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    at.session_state[StateKeys.DATASET] = create_dataset_alphapept()
    at.run()

    assert not at.exception
    assert at.columns[0].selectbox.len == 3
    assert at.button.len == 3


def test_page_04_runs_preprocessreset_alphapept():
    """Test if the page preprocesses and resets preprocessing without exceptions."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    at.session_state[StateKeys.DATASET] = create_dataset_alphapept()
    at.run()

    at.button(key="_run_preprocessing").click()
    at.run()

    at.button(key="_reset_preprocessing").click()
    at.run()

    assert not at.exception
