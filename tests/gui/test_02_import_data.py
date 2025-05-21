from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from alphastats.gui.utils.state_keys import StateKeys

from .conftest import APP_FOLDER, data_buf, metadata_buf

TESTED_PAGE = f"{APP_FOLDER}/pages/02_Import Data.py"


def test_page_02_loads_without_input():
    """Test if the page loads without any input and inititalizes the session state with the correct values."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    assert not at.exception

    assert at.session_state[StateKeys.USER_SESSION_ID] is not None


@patch("streamlit.file_uploader")
def test_patched_page_02_loads_without_input(mock_file_uploader: MagicMock):
    """Test if the page loads without any input and inititalizes the session state with the correct value when the file_uploader is patched."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    assert not at.exception

    assert at.session_state[StateKeys.USER_SESSION_ID] is not None


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

    assert at.session_state[StateKeys.DATASET].metadata.columns.to_list() == [
        "sample_",
        "disease",
        "Drug therapy (procedure) (416608005)",
        "Lipid-lowering therapy (134350008)",
    ]
    assert (
        str(type(at.session_state[StateKeys.DATASET]))
        == "<class 'alphastats.dataset.dataset.DataSet'>"
    )


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
    mock_file_uploader.side_effect = [data_buf(DATA_FILE), None]
    at.run()

    # User uploads the metadata file
    mock_file_uploader.side_effect = [
        data_buf(DATA_FILE),
        metadata_buf(METADATA_FILE),
    ]
    at.run()

    assert not at.exception

    # User clicks the Load Data button
    mock_file_uploader.side_effect = [
        data_buf(DATA_FILE),
        metadata_buf(METADATA_FILE),
    ]
    at.button(key="_create_dataset").click()
    at.run()

    assert not at.exception

    dataset = at.session_state[StateKeys.DATASET]
    assert dataset._intensity_column == "LFQ intensity [sample]"
    assert dataset.rawmat.shape == (312, 2249)
    assert dataset.software == "MaxQuant"
