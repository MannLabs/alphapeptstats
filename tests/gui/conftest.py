from streamlit.testing.v1 import AppTest
from alphastats.load_data import load_data
from pathlib import Path
from alphastats import DataSet


APP_FOLDER = Path(__file__).parent / Path("../../alphastats/gui/")
TEST_INPUT_FILES = f"{APP_FOLDER}/../../testfiles"


def print_session_state(apptest: AppTest):
    """Prints the session state of the AppTest object.
    Not used productively, but for debugging purposes."""
    for k, v in apptest.session_state.filtered_state.items():
        print(
            f"{k}:    {str(type(v))}   {str(v)[:20] if type(v) not in [int, list, str] else v}"
        )


def create_dataset_alphapept():
    """Creates a dataset object from the alphapept testfiles."""
    loader = load_data(
        file=TEST_INPUT_FILES + "/alphapept/results_proteins.csv", type="alphapept"
    )
    metadata_path = TEST_INPUT_FILES + "/alphapept/metadata.csv"
    return DataSet(
        loader=loader,
        metadata_path=metadata_path,
        sample_column="sample",
    )
