import streamlit as st
import os
import io

from alphastats.gui.utils.options import SOFTWARE_OPTIONS

try:
    from alphastats.gui.utils.import_helper import (
        save_plot_sampledistribution_rawdata,
        display_loaded_dataset,
        load_sample_data,
        empty_session_state, load_softwarefile_df, show_upload_metadatafile, show_select_columns_for_loaders,
        show_loader_columns_selection, load_proteomics_data,
)
    from alphastats.gui.utils.ui_helper import sidebar_info

except ModuleNotFoundError:
    from utils.ui_helper import sidebar_info
    from utils.import_helper import (
        import_data,
        save_plot_sampledistribution_rawdata,
        display_loaded_dataset,
        load_sample_data,
        empty_session_state,
    )

from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

runtime = get_instance()
session_id = get_script_run_ctx().session_id
session_info = runtime._session_mgr.get_session_info(session_id)

user_session_id = session_id
st.session_state["user_session_id"] = user_session_id

if "loader" not in st.session_state:
    st.session_state["loader"] = None

if "gene_to_prot_id" not in st.session_state:
    st.session_state["gene_to_prot_id"] = {}

if "organism" not in st.session_state:
    st.session_state["organism"] = 9606  # human

sidebar_info()

if "dataset" not in st.session_state:
    st.markdown("### Import Proteomics Data")

    st.markdown(
        "Create a DataSet with the output of your proteomics software package and the corresponding metadata (optional). "
    )

# ########## Select Software
default_select_option = "<select>"
options = [default_select_option] + list(SOFTWARE_OPTIONS.keys())

st.selectbox(
    "Select your Proteomics Software",
    options=options,
    key="software",
)

software = st.session_state["software"]

# ########## Load Software File
if software != default_select_option:
    softwarefile = st.file_uploader(
        SOFTWARE_OPTIONS.get(software).get("import_file"),
        type=["csv", "tsv", "txt", "hdf"],
    )

    if softwarefile is not None:
        softwarefile_df = load_softwarefile_df(software, softwarefile)

        intensity_column, index_column = show_loader_columns_selection(software=software, softwarefile_df=softwarefile_df)

        loader = load_proteomics_data(
            softwarefile_df,
            intensity_column=intensity_column,
            index_column=index_column,
            software=software,
        )

        st.session_state["loader"] = loader

### Load Metadata File
if st.session_state["loader"] is not None:
    show_upload_metadatafile(software)


if "dataset" in st.session_state:
    st.info("DataSet has been imported")

    if "distribution_plot" not in st.session_state:
        save_plot_sampledistribution_rawdata()

    display_loaded_dataset()

st.markdown("### Or Load sample Dataset")

if st.button("Load sample DataSet - PXD011839"):
    load_sample_data()
    if "distribution_plot" not in st.session_state:
        save_plot_sampledistribution_rawdata()


st.markdown("### To start a new session:")

if st.button("New Session: Import new dataset"):
    empty_session_state()
    st.rerun()
