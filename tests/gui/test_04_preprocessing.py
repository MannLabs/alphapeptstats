from pathlib import Path

from streamlit.testing.v1 import AppTest

from alphastats.DataSet import DataSet
from alphastats.load_data import load_data

APP_FOLDER = Path(__file__).parent / Path("../../alphastats/gui/")
APP_FILE = f"{APP_FOLDER}/pages/03_Preprocessing.py"
TEST_INPUT_FILES = f"{APP_FOLDER}/../../testfiles"


def print_session_state(apptest: AppTest):
    for k, v in apptest.session_state.filtered_state.items():
        print(
            f"{k}:    {str(type(v))}   {str(v)[:20] if type(v) not in [int, list, str] else v}"
        )


def create_dataset_alphapept():
    loader = load_data(
        file=TEST_INPUT_FILES + "/alphapept/results_proteins.csv", type="alphapept"
    )
    metadata_path = TEST_INPUT_FILES + "/alphapept/metadata.csv"
    return DataSet(
        loader=loader,
        metadata_path=metadata_path,
        sample_column="sample",
    )


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
