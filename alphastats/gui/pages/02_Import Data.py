import streamlit as st
import os
import io

from alphastats import DataSet
from alphastats.gui.utils.options import SOFTWARE_OPTIONS

try:
    from alphastats.gui.utils.import_helper import (
        save_plot_sampledistribution_rawdata,
        display_loaded_dataset,
        load_sample_data,
        empty_session_state, load_softwarefile_df, show_upload_metadatafile,
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

from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

def init_session_state():
    session_id = get_script_run_ctx().session_id

    user_session_id = session_id
    st.session_state["user_session_id"] = user_session_id

    if "loader" not in st.session_state:
        st.session_state["loader"] = None

    if "gene_to_prot_id" not in st.session_state:
        st.session_state["gene_to_prot_id"] = {}

    if "organism" not in st.session_state:
        st.session_state["organism"] = 9606  # human

init_session_state()

sidebar_info()

st.markdown("### Start a new session")
st.write("Start a new session will discard the current one (including all analysis!) and enable importing a new dataset.")
st.write("To explore AlphaPeptStats you may also load an example dataset.")

c1, c2 = st.columns(2)
if c1.button("Start new Session"):
    empty_session_state()

    st.rerun()

if c2.button("Start new Session with example DataSet"):
    empty_session_state()
    init_session_state()
    load_sample_data()
    if "distribution_plot" not in st.session_state:
         save_plot_sampledistribution_rawdata()

if "dataset" not in st.session_state:
    st.markdown("### Import Proteomics Data")

    st.markdown(
        "Create a DataSet with the output of your proteomics software package and the corresponding metadata (optional). "
    )

    # ########## Select Software
    st.markdown("##### 1. Select software and upload data")

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

    # ##########  Load Metadata File
    if st.session_state["loader"] is not None:
        st.markdown("##### 3. Prepare Metadata (optional)")
        metadatafile_df = show_upload_metadatafile()
        show_create_dataset_button(metadatafile_df, software)

        st.markdown("##### 3b. Continue without metadata")
        # TODO make this "4. Create dataset", displaying 1 or 2 buttons depending on if metadata is available
        if st.button("--> Create DataSet without metadata"):
            # TODO idempotency of buttons / or disable
            st.session_state["dataset"] = DataSet(loader=st.session_state.loader)
            st.session_state["metadata_columns"] = ["sample"]

            load_options()

if "dataset" in st.session_state:
    st.markdown("### DataSet Info")
    st.info("DataSet has been imported")

    if "distribution_plot" not in st.session_state:
        save_plot_sampledistribution_rawdata()

    display_loaded_dataset()
