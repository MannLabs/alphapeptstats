from io import BytesIO
from pathlib import Path

from streamlit.testing.v1 import AppTest

from alphastats.DataSet import DataSet
from alphastats.load_data import load_data

# TODO: Turn the helpers into fixtures

APP_FOLDER = Path(__file__).parent / "../../alphastats/gui/"
TEST_INPUT_FILES_PATH = APP_FOLDER / "../../testfiles"


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
        file=str(TEST_INPUT_FILES_PATH / "alphapept/results_proteins.csv"),
        type="alphapept",
    )
    metadata_path = TEST_INPUT_FILES_PATH / "alphapept/metadata.csv"
    return DataSet(
        loader=loader,
        metadata_path_or_df=str(metadata_path),
        sample_column="sample",
    )


def data_buf(file_path: str):
    """Helper function to open a data file from the testfiles folder and return a BytesIO object.

    Additionally add filename as attribute."""
    with open(TEST_INPUT_FILES_PATH / file_path, "rb") as f:
        buf = BytesIO(f.read())
        buf.name = file_path.split("/")[-1]
        return buf


def metadata_buf(file_path: str):
    """Helper function to open a metadata file from the testfiles folder and return a BytesIO object.

    Additionally add filename as attribute and set the metadatafile in the session state."""
    with open(TEST_INPUT_FILES_PATH / file_path, "rb") as f:
        buf = BytesIO(f.read())
        buf.name = file_path.split("/")[-1]
        return buf
