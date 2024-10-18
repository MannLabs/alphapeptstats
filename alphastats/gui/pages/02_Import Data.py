from typing import List

import streamlit as st

from alphastats import BaseLoader
from alphastats.DataSet import DataSet
from alphastats.gui.utils.import_helper import (
    load_example_data,
    load_proteomics_data,
    show_button_download_metadata_template_file,
    show_loader_columns_selection,
    show_select_sample_column_for_metadata,
    uploaded_file_to_df,
)
from alphastats.gui.utils.options import SOFTWARE_OPTIONS
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    empty_session_state,
    init_session_state,
    sidebar_info,
)


def _finalize_data_loading(
    loader: BaseLoader,
    metadata_columns: List[str],
    dataset: DataSet,
) -> None:
    """Finalize the data loading process."""
    st.session_state[StateKeys.LOADER] = (
        loader  # TODO: Figure out if we even need the loader here, as the dataset has the loader as an attribute.
    )
    st.session_state[StateKeys.METADATA_COLUMNS] = metadata_columns
    st.session_state[StateKeys.DATASET] = dataset

    sidebar_info()

    st.page_link("pages/03_Data Overview.py", label="=> Go to data overview page..")


init_session_state()
sidebar_info()

st.markdown("### Start a new session")
st.write(
    "Start a new session will discard the current one (including all analysis!) and enable importing a new dataset."
)
st.write("To explore AlphaPeptStats you may also load an example dataset.")

c1, c2 = st.columns(2)
if c1.button("Start new Session"):
    empty_session_state()
    st.rerun()


if c2.button("Start new Session with example DataSet", key="_load_example_data"):
    empty_session_state()
    init_session_state()
    loader, metadata_columns, dataset = load_example_data()

    _finalize_data_loading(loader, metadata_columns, dataset)
    st.stop()


st.markdown("### Import Proteomics Data")
if StateKeys.DATASET in st.session_state:
    st.info("DataSet already present.")
    st.page_link("pages/03_Data Overview.py", label="=> Go to data overview page..")
    st.stop()


st.markdown(
    "Create a DataSet with the output of your proteomics software package and the corresponding metadata (optional). "
)

# ########## Select Software
st.markdown("##### 1. Select software and upload data")

default_select_option = "<select>"
options = [default_select_option] + list(SOFTWARE_OPTIONS.keys())

software = st.selectbox(
    "Select your Proteomics Software", options=options, key="_software"
)
if software == default_select_option:
    st.stop()


# ########## Load Software File

softwarefile = st.file_uploader(
    SOFTWARE_OPTIONS.get(software).get("import_file"),
    type=["csv", "tsv", "txt", "hdf"],
)

if softwarefile is None:
    st.stop()

softwarefile_df = uploaded_file_to_df(softwarefile, software)

intensity_column, index_column = show_loader_columns_selection(
    software=software, softwarefile_df=softwarefile_df
)

loader = load_proteomics_data(
    softwarefile_df,
    intensity_column=intensity_column,
    index_column=index_column,
    software=software,
)


# ##########  Load Metadata File
st.markdown("##### 3. Prepare Metadata (optional)")

st.write(
    "Download the template file and add additional information "
    + "to your samples as columns (e.g. 'disease group'). "
    + "Then upload the updated metadata file."
)
show_button_download_metadata_template_file(loader)

metadatafile_upload = st.file_uploader(
    "Upload metadata file with information about your samples",
)

if metadatafile_upload is None:
    st.stop()

metadatafile_df = uploaded_file_to_df(metadatafile_upload)

sample_column = show_select_sample_column_for_metadata(
    metadatafile_df, software, loader
)


# ##########  Create dataset
st.markdown("##### 4. Create DataSet")

dataset = None
metadata_columns = []
c1, c2 = st.columns(2)

if c2.button("Create DataSet without metadata"):
    dataset = DataSet(loader=loader)
    metadata_columns = ["sample"]

if c1.button(
    "Create DataSet with metadata",
    disabled=metadatafile_df is None,
    key="_create_dataset",
):
    if len(metadatafile_df[sample_column].to_list()) != len(
        metadatafile_df[sample_column].unique()
    ):
        raise ValueError("Sample names have to be unique.")

    dataset = DataSet(
        loader=loader,
        metadata_path_or_df=metadatafile_df,
        sample_column=sample_column,
    )
    metadata_columns = metadatafile_df.columns.to_list()

if dataset is not None:
    st.info("DataSet has been created.")
    _finalize_data_loading(loader, metadata_columns, dataset)
