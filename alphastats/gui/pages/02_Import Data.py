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
        show_loader_columns_selection, load_proteomics_data, load_options,
        show_select_sample_column_for_metadata, init_session_state,
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


st.markdown("### Import Proteomics Data")
if "dataset" in st.session_state:
    st.info(f"DataSet already present: {st.session_state['dataset']}")
    st.stop()

if "dataset" not in st.session_state:

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


    if st.session_state["loader"] is not None:
        dataset = None
        metadata_columns = []

        # ##########  Load Metadata File
        st.markdown("##### 3. Prepare Metadata (optional)")

        sample_column = None
        metadatafile_df = show_upload_metadatafile()
        if metadatafile_df is not None:
            sample_column = show_select_sample_column_for_metadata(metadatafile_df, software)

        # ##########  Create dataset
        st.markdown("##### 4. Create DataSet")

        if sample_column:
            if st.button("Create DataSet with metadata"):
                if len(metadatafile_df[sample_column].to_list()) != len(
                        metadatafile_df[sample_column].unique()
                ):
                    raise ValueError("Sample names have to be unique.")

                dataset = DataSet(
                    loader=st.session_state.loader,
                    metadata_path=metadatafile_df,
                    sample_column=sample_column,
                )
                metadata_columns = metadatafile_df.columns.to_list()

        if st.button("Create DataSet without metadata"):
            dataset = DataSet(loader=st.session_state.loader)
            metadata_columns = ["sample"]

        if dataset is not None:
            st.info("DataSet has been created.")
            st.session_state["dataset"] = dataset
            st.session_state["metadata_columns"] = metadata_columns
            load_options()


