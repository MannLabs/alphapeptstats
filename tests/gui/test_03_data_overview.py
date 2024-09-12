from streamlit.testing.v1 import AppTest
from pathlib import Path
from unittest.mock import MagicMock, patch
from .test_helper import create_dataset_alphapept
import pandas as pd
from io import BytesIO


def print_session_state(apptest: AppTest):
    for k, v in apptest.session_state.filtered_state.items():
        print(
            f"{k}:    {str(type(v))}   {str(v)[:20] if type(v) not in [int, list, str] else v}"
        )


APP_FOLDER = Path(__file__).parent / Path("../../alphastats/gui/")
APP_FILE = f"{APP_FOLDER}/pages/03_Data Overview.py"
TEST_INPUT_FILES = f"{APP_FOLDER}/../../testfiles"


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