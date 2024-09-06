import streamlit as st
import os
import io

try:
    from alphastats.gui.utils.import_helper import (
        import_data,
        save_plot_sampledistribution_rawdata,
        display_loaded_dataset,
        load_sample_data,
        empty_session_state,
    )
    from alphastats.gui.utils.ui_helper import sidebar_info
    from alphastats.DataSet import DataSet
    from alphastats.gui.utils.analysis_helper import (
        get_sample_names_from_software_file,
        read_uploaded_file_into_df,
    )
    from alphastats.gui.utils.software_options import software_options
    from alphastats.gui.utils.ui_helper import sidebar_info, StateKeys
    from alphastats.loader.MaxQuantLoader import MaxQuantLoader

except ModuleNotFoundError:
    from utils.ui_helper import sidebar_info
    from utils.import_helper import (
        import_data,
        save_plot_sampledistribution_rawdata,
        display_loaded_dataset,
        load_sample_data,
        empty_session_state,
    )

    from utils.ui_helper import sidebar_info, StateKeys
    from utils.analysis_helper import (
        get_sample_names_from_software_file,
        read_uploaded_file_into_df,
    )
    from utils.software_options import software_options
    from alphastats import MaxQuantLoader
    from alphastats import DataSet

import pandas as pd
import plotly.express as px
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

runtime = get_instance()
session_id = get_script_run_ctx().session_id
# session_info = runtime._session_mgr.get_session_info(session_id)

user_session_id = session_id
st.session_state[StateKeys.USER_SESSION_ID] = user_session_id

if "loader" not in st.session_state:
    st.session_state[StateKeys.LOADER] = None

if "gene_to_prot_id" not in st.session_state:
    st.session_state[StateKeys.GENE_TO_PROT_ID] = {}

if "organism" not in st.session_state:
    st.session_state[StateKeys.ORGANISM] = 9606  # human

sidebar_info()

if "dataset" not in st.session_state:
    st.markdown("### Import Proteomics Data")

    st.markdown(
        "Create a DataSet with the output of your proteomics software package and the corresponding metadata (optional). "
    )

import_data()

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
