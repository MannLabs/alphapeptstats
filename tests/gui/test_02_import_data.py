from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


def print_session_state(apptest: AppTest):
    for k, v in apptest.session_state.filtered_state.items():
        print(
            f"{k}:    {str(type(v))}   {str(v)[:20] if type(v) not in [int, list, str] else v}"
        )


APP_FOLDER = Path(__file__).parent / Path("../../alphastats/gui/")
TESTED_PAGE = f"{APP_FOLDER}/pages/02_Import Data.py"
TEST_INPUT_FILES_PATH = APP_FOLDER / "../../testfiles"


def test_page_02_loads_without_input():
    """Test if the page loads without any input and inititalizes the session state with the correct values."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    assert not at.exception

    assert at.session_state.organism == 9606
    assert at.session_state.user_session_id is not None
    assert at.session_state.gene_to_prot_id == {}


@patch("streamlit.file_uploader")
def test_patched_page_02_loads_without_input(mock_file_uploader: MagicMock):
    """Test if the page loads without any input and inititalizes the session state with the correct value when the file_uploader is patched."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    assert not at.exception

    assert at.session_state.organism == 9606
    assert at.session_state.user_session_id is not None
    assert at.session_state.gene_to_prot_id == {}


@patch(
    "streamlit.page_link"
)  # page link is mocked to avoid errors with the relative paths
def test_page_02_loads_example_data(mock_page_link: MagicMock):
    """Test if the page loads the example data and has the correct session state afterwards."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    # User clicks Load Sample Data button
    at.button(key="_load_example_data").click().run()

    assert not at.exception

    assert at.session_state.metadata_columns == [
        "sample",
        "disease",
        "Drug therapy (procedure) (416608005)",
        "Lipid-lowering therapy (134350008)",
    ]
    assert str(type(at.session_state.dataset)) == "<class 'alphastats.DataSet.DataSet'>"
    assert (
        str(type(at.session_state.loader))
        == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    )
    assert "plotting_options" in at.session_state
    assert "statistic_options" in at.session_state


def _data_buf(file_path: str):
    """Helper function to open a data file from the testfiles folder and return a BytesIO object.

    Additionally add filename as attribute."""
    with open(TEST_INPUT_FILES_PATH / file_path, "rb") as f:
        buf = BytesIO(f.read())
        buf.name = file_path.split("/")[-1]
        return buf


def _metadata_buf(file_path: str, at: AppTest):
    """Helper function to open a metadata file from the testfiles folder and return a BytesIO object.

    Additionally add filename as attribute and set the metadatafile in the session state."""
    with open(TEST_INPUT_FILES_PATH / file_path, "rb") as f:
        buf = BytesIO(f.read())
        buf.name = file_path.split("/")[-1]
        return buf


@patch("streamlit.file_uploader")
@patch(
    "streamlit.page_link"
)  # page link is mocked to avoid errors with the relative paths
def test_page_02_loads_maxquant_testfiles(
    mock_page_link: MagicMock, mock_file_uploader: MagicMock
):
    """Test if the page loads the MaxQuant testfiles and has the correct session state afterwards.

    No input to the dropdown menus is simulated, hence the default detected values are used.
    Two states are tested:
    1. Files are uploaded but not processed yet
    2. Files are uploaded and processed"""
    DATA_FILE = "maxquant/proteinGroups.txt"
    METADATA_FILE = "maxquant/metadata.xlsx"

    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    # User selects MaxQuant from the dropdown menu
    at.selectbox(key="_software").select("MaxQuant")
    mock_file_uploader.side_effect = [None]
    at.run()

    # User uploads the data file
    mock_file_uploader.side_effect = [_data_buf(DATA_FILE), None]
    at.run()

    # User uploads the metadata file
    mock_file_uploader.side_effect = [
        _data_buf(DATA_FILE),
        _metadata_buf(METADATA_FILE, at),
    ]
    at.run()

    assert not at.exception

    # User clicks the Load Data button
    mock_file_uploader.side_effect = [
        _data_buf(DATA_FILE),
        _metadata_buf(METADATA_FILE, at),
    ]
    at.button(key="_create_dataset").click()
    at.run()

    assert not at.exception

    dataset = at.session_state.dataset
    assert dataset._gene_names == "Gene names"
    assert dataset.index_column == "Protein IDs"
    assert dataset._intensity_column == "LFQ intensity [sample]"
    assert dataset.rawmat.shape == (312, 2611)
    assert dataset.software == "MaxQuant"
    assert dataset.sample == "sample"
    assert (
        str(type(at.session_state.loader))
        == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    )
    assert "plotting_options" in at.session_state
    assert "statistic_options" in at.session_state
