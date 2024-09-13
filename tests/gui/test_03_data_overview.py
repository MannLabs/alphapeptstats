from streamlit.testing.v1 import AppTest
from pathlib import Path
from unittest.mock import MagicMock, patch
from .conftest import create_dataset_alphapept, APP_FOLDER

APP_FILE = f"{APP_FOLDER}/pages/03_Data Overview.py"

def test_page_03_loads_without_input():
    """Test if the page loads without any input and inititalizes the session state with the correct values."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    assert not at.exception


def test_page_03_loads_with_input():
    """Test if the page loads with input and inititalizes the session state with the correct values."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    at.session_state["dataset"] = create_dataset_alphapept()
    at.run()

    assert not at.exception
