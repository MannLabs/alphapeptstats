from streamlit.testing.v1 import AppTest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
from io import BytesIO

def print_session_state(apptest: AppTest):
    for k,v in apptest.session_state.filtered_state.items():
        print(f"{k}:    {str(type(v))}   {str(v)[:20] if type(v) not in [int, list, str] else v}")

APP_FOLDER = Path(__file__).parent / Path("../")
APP_FILE = f"{APP_FOLDER}/pages/02_Import Data.py"
TEST_INPUT_FILES = f"{APP_FOLDER}/../../testfiles"

def test_page_02_loads_without_input():
    """Test if the page loads without any input and inititalizes the session state with the correct values."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    assert not at.exception

    assert at.session_state.organism == 9606
    assert at.session_state.user_session_id == 'test session id'
    assert at.session_state.software == '<select>'
    assert at.session_state.gene_to_prot_id == {}
    assert at.session_state.loader == None

@patch("streamlit.file_uploader")
def test_patched_page_02_loads_without_input(mock_file_uploader: MagicMock):
    """Test if the page loads without any input and inititalizes the session state with the correct value when the file_uploader is patched."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    assert not at.exception

    assert at.session_state.organism == 9606
    assert at.session_state.user_session_id == 'test session id'
    assert at.session_state.software == '<select>'
    assert at.session_state.gene_to_prot_id == {}
    assert at.session_state.loader == None

def test_page_02_loads_sample_data():
    """Test if the page loads the sample data and has the correct session state afterwards."""
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    # User clicks Load Sample Data button
    at.button("load_sample_data").click().run()

    assert not at.exception

    assert at.session_state.metadata_columns == ['sample', 'disease', 'Drug therapy (procedure) (416608005)', 'Lipid-lowering therapy (134350008)']
    assert str(type(at.session_state.dataset)) == "<class 'alphastats.DataSet.DataSet'>"
    assert at.session_state.software == "<select>"
    assert str(type(at.session_state.distribution_plot)) == "<class 'plotly.graph_objs._figure.Figure'>"
    assert str(type(at.session_state.loader)) == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    assert "plotting_options" in at.session_state
    assert "statistic_options" in at.session_state


def _data_buf(path_from_testfiles: str):
    """Helper function to open a data file from the testfiles folder and return a BytesIO object.
    
    Additionally add filename as attribute."""
    with open(f"{TEST_INPUT_FILES}{path_from_testfiles}", "rb") as f:
        buf = BytesIO(f.read())
        buf.name = path_from_testfiles.split('/')[-1]
        return buf


def _metadata_buf(path_from_testfiles: str, at: AppTest):
    """Helper function to open a metadata file from the testfiles folder and return a BytesIO object.
    
    Additionally add filename as attribute and set the metadatafile in the session state."""
    with open(f"{TEST_INPUT_FILES}{path_from_testfiles}", "rb") as f:
        buf = BytesIO(f.read())
        buf.name = path_from_testfiles.split('/')[-1]
        at.session_state.metadatafile = buf
        return buf


@patch("streamlit.file_uploader")
def test_page_02_loads_maxquant_testfiles(mock_file_uploader: MagicMock):
    """Test if the page loads the MaxQuant testfiles and has the correct session state afterwards.
    
    No input to the dropdown menus is simulated, hence the default detected values are used.
    Two states are tested:
    1. Files are uploaded but not processed yet
    2. Files are uploaded and processed"""
    DATA_FILE = "/maxquant/proteinGroups.txt"
    METADATA_FILE = "/maxquant/metadata.xlsx"

    at = AppTest(APP_FILE, default_timeout=200)
    at.run()
    
    # User selects MaxQuant from the dropdown menu
    at.selectbox(key='software').select('MaxQuant')
    mock_file_uploader.side_effect = [None]
    at.run()

    # User uploads the data file
    mock_file_uploader.side_effect = [_data_buf(DATA_FILE),None]
    at.run()

    # User uploads the metadata file
    mock_file_uploader.side_effect = [_data_buf(DATA_FILE),_metadata_buf(METADATA_FILE, at)]
    at.run()

    assert not at.exception

    assert str(type(at.session_state.loader)) == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    assert at.session_state.intensity_column == 'LFQ intensity [sample]'
    assert str(type(at.session_state.metadatafile)) == "<class '_io.BytesIO'>"
    assert at.session_state.software == 'MaxQuant'
    assert at.session_state.index_column == 'Protein IDs'
    assert at.session_state.metadata_columns == ['sample']
    assert at.session_state.sample_column == 'sample'

    # User clicks the Load Data button
    mock_file_uploader.side_effect = [_data_buf(DATA_FILE),_metadata_buf(METADATA_FILE, at)]
    at.button('FormSubmitter:sample_column-Create DataSet').click()
    at.run()

    assert not at.exception
    
    assert at.session_state.dataset.gene_names == "Gene names"
    assert at.session_state.dataset.index_column == "Protein IDs"
    assert at.session_state.dataset.intensity_column == 'LFQ intensity [sample]'
    assert at.session_state.dataset.rawmat.shape == (312, 2611)
    assert at.session_state.dataset.software == 'MaxQuant'
    assert at.session_state.dataset.sample == 'sample'
    assert str(type(at.session_state.distribution_plot)) == "<class 'plotly.graph_objs._figure.Figure'>"
    assert str(type(at.session_state.loader)) == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    assert "plotting_options" in at.session_state
    assert "statistic_options" in at.session_state