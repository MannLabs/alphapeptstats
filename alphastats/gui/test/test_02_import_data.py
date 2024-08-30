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
TEST_FILES = f"{APP_FOLDER}/../../testfiles"

def test_loadpage():
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    assert at.session_state.organism == 9606
    assert at.session_state.user_session_id == 'test session id'
    assert at.session_state.software == '<select>'
    assert at.session_state.gene_to_prot_id == {}
    assert at.session_state.loader == None

@patch("streamlit.file_uploader")
def test_loadmockedpage(mock_file_uploader: MagicMock):
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    assert at.session_state.organism == 9606
    assert at.session_state.user_session_id == 'test session id'
    assert at.session_state.software == '<select>'
    assert at.session_state.gene_to_prot_id == {}
    assert at.session_state.loader == None

def test_sampledata():
    at = AppTest(APP_FILE, default_timeout=200)
    at.run()

    at.button[0].click().run()

    assert at.session_state.metadata_columns == ['sample', 'disease', 'Drug therapy (procedure) (416608005)', 'Lipid-lowering therapy (134350008)']
    assert str(type(at.session_state.dataset)) == "<class 'alphastats.DataSet.DataSet'>"
    assert at.session_state.software == "<select>"
    assert str(type(at.session_state.distribution_plot)) == "<class 'plotly.graph_objs._figure.Figure'>"
    assert str(type(at.session_state.loader)) == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    assert "plotting_options" in at.session_state
    assert "statistic_options" in at.session_state

@patch("streamlit.file_uploader")
def test_mqupload(mock_file_uploader: MagicMock):
    def data_buf():
        with open(f"{TEST_FILES}/maxquant/proteinGroups.txt", "rb") as f:
            buf = BytesIO(f.read())
            buf.name = "proteinGroups.txt"
            return buf
    def metadata_buf():
        with open(f"{TEST_FILES}/maxquant/metadata.xlsx", "rb") as f:
            buf = BytesIO(f.read())
            buf.name = "metadata.xlsx"
            at.session_state.metadatafile = buf
            return buf

    at = AppTest(APP_FILE, default_timeout=200)
    at.run()
    
    at.selectbox(key='software').select('MaxQuant')
    mock_file_uploader.side_effect = [None]
    at.run()

    mock_file_uploader.side_effect = [data_buf(),None]
    at.run()

    mock_file_uploader.side_effect = [data_buf(),metadata_buf()]
    at.run()    
    assert str(type(at.session_state.loader)) == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    assert at.session_state.intensity_column == 'LFQ intensity [sample]'
    assert str(type(at.session_state.metadatafile)) == "<class '_io.BytesIO'>"
    assert at.session_state.software == 'MaxQuant'
    assert at.session_state.index_column == 'Protein IDs'
    assert at.session_state.metadata_columns == ['sample']
    assert at.session_state.sample_column == 'sample'

    mock_file_uploader.side_effect = [data_buf(),metadata_buf()]
    at.button[0].click()
    at.run()
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

if __name__ == "__main__":
    test_mqupload()

