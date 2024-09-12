from alphastats import DataSet
from streamlit.testing.v1 import AppTest
from pathlib import Path
from unittest.mock import MagicMock, patch
from .conftest import create_dataset_alphapept
import pandas as pd
from io import BytesIO


APP_FOLDER = Path(__file__).parent / Path("../../alphastats/gui/")
APP_FILE = f"{APP_FOLDER}/pages/03_Preprocessing.py"
TEST_INPUT_FILES = f"{APP_FOLDER}/../../testfiles"


def test_page_04_loads_without_input():
    """Test if the page loads without any input and inititalizes the session state with the correct values."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    assert not at.exception


def test_page_04_loads_with_input():
    """Test if the page loads with input and inititalizes the session state with the correct values."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    at.session_state["dataset"] = create_dataset_alphapept()
    at.run()

    assert not at.exception
    assert at.columns[3].selectbox.len == 6
    assert at.button.len == 2


def test_page_04_runs_preprocessreset_alphapept():
    """Test if the page preprocesses and resets preprocessing without exceptions."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    at.session_state["dataset"] = create_dataset_alphapept()
    at.run()

    at.button(key="_run_preprocessing").click()
    at.run()

    at.button(key="_reset_preprocessing").click()
    at.run()

    assert not at.exception
