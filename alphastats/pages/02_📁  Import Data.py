import streamlit as st
import os
import pandas as pd
import datetime
import yaml
from typing import Union, Tuple
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.DataSet import DataSet
import logging
from alphastats.AlphaStats import sidebar_info


software_options = {
    "MaxQuant": {
        "import_file": "proteinGroups.txt",
        "intensity_column": ["LFQ intensity [sample]"],
        "index_column": ["Protein IDs"],
        "loader_function": MaxQuantLoader,
    },
    "AlphaPept": {
        "import_file": "results_proteins.csv or results.hdf",
        "intensity_column": ["[sample]_LFQ"],
        "index_column": ["Unnamed: 0"],
        "loader_function": AlphaPeptLoader,
    },
    "DIANN": {
        "import_file": "report.pg_matrix.tsv",
        "intensity_column": ["[sample]"],
        "index_column": ["Protein.Group"],
        "loader_function": DIANNLoader,
    },
    "Fragpipe": {
        "import_file": "combined_protein.tsv",
        "intensity_column": ["[sample] MaxLFQ Intensity "],
        "index_column": ["Protein"],
        "loader_function": FragPipeLoader,
    },
}


def print_software_import_info():
    import_file = software_options.get(st.session_state.software).get("import_file")
    string_output = f"Please upload {import_file} file from {st.session_state.software}."
    return string_output

def select_columns_for_loaders():
    """
    select intensity and index column depending on software
    will be saved in session state
    """
    st.write("Select intensity columns for further analysis")
    st.selectbox(
        "Intensity Column",
        options=software_options.get(st.session_state.software).get("intensity_column"),
         key="intensity_column",
    )

    st.write("Select index column (with ProteinGroups) for further analysis")
    st.selectbox(
            "Index Column",
            options=software_options.get(st.session_state.software).get("index_column"),
            key="index_column",
        )

def load_proteomics_data(uploaded_file, intensity_column, index_column):
    """load software file into loader object from alphastats
    """
    loader = software_options.get(st.session_state.software)["loader_function"](
            uploaded_file, intensity_column, index_column
        )
    return loader

def read_uploaded_file_into_df(file):
    filename = file.name
    if filename.endswith(".xlsx"):
        df = pd.read_excel(file)
    elif filename.endswith(".txt") or filename.endswith(".tsv"):
        df = pd.read_csv(file, delimiter="\t")
    elif filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = None
        logging.warning(
                "WARNING: File could not be read. \nFile has to be a .xslx, .tsv, .csv or .txt file"
            )
        return
    return df

def select_sample_column_metadata(df):
    st.write(
            f"Select column that contains sample IDs matching the sample names described"
            + f"in {software_options.get(st.session_state.software).get('import_file')}"
        )
    with st.form("sample_column"):
        st.selectbox("Sample Column", options=df.columns.to_list(), key = "sample_column")
        submitted = st.form_submit_button("Submit")
    if submitted:
        return True

def display_file(df):
    st.dataframe(df.head(5))

def upload_softwarefile():
    softwarefile = st.file_uploader(
            print_software_import_info(), key="softwarefile"
        )
    if "softwarefile" in st.session_state:
        softwarefile_df = read_uploaded_file_into_df(softwarefile)
        # display head a protein data
        display_file(softwarefile_df)
        select_columns_for_loaders()
        if (
                "intensity_column" in st.session_state
                and "index_column" in st.session_state
        ):
            loader = load_proteomics_data(
                    softwarefile_df,
                    intensity_column=st.session_state.intensity_column,
                    index_column=st.session_state.index_column,
                )
            st.session_state["loader"] = loader

def upload_metadatafile():
    st.file_uploader(
            "Upload metadata file. with information about your samples",
            key="metadatafile",
        )
    if "metadatafile" in st.session_state:
        metadatafile_df = read_uploaded_file_into_df(st.session_state.metadatafile)
        # display metadata
        display_file(metadatafile_df)
        # pick sample column
        if select_sample_column_metadata(metadatafile_df):
            # create dataset
            st.session_state["dataset"] = DataSet(
                    loader=st.session_state.loader,
                    metadata_path=metadatafile_df,
                    sample_column=st.session_state.sample_column
                )
            st.session_state["metadata_columns"] = metadatafile_df.columns.to_list()
    if st.button("Continue without metadata"):
        st.session_state["dataset"] = DataSet(loader=st.session_state.loader)

        

def import_data():
    st.markdown("### Import Data")
    st.selectbox(
            "Select your Proteomics Software",
            options=["<select>", "MaxQuant", "AlphaPept", "DIANN", "Fragpipe"],
            key="software"
        )

    if st.session_state.software != "<select>":
        upload_softwarefile()

    if "loader" in st.session_state:
        upload_metadatafile()

def display_loaded_dataset():
    st.write("## Data was successfully imported")
    st.write("## DataSet has been created")
    st.write(f"Raw data from {st.session_state.dataset.software}")
    display_file(st.session_state.dataset.rawdata)
    st.write(f"Metadata")
    display_file(st.session_state.dataset.metadata)



sidebar_info()

if "dataset" not in st.session_state:
    import_data()

elif st.button('Import new dataset'):
    del st.session_state["dataset"]
    import_data()

else:
    display_loaded_dataset()
    
            
