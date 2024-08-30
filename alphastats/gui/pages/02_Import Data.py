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
# session_info = runtime._session_mgr.get_session_info(session_id)

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

import_data()

if "dataset" in st.session_state:
    st.info("DataSet has been imported")

    if "distribution_plot" not in st.session_state:
        save_plot_sampledistribution_rawdata()

    display_loaded_dataset()

st.markdown("### Or Load sample Dataset")

if st.button("Load sample DataSet - PXD011839", key="load_sample_data"):
    load_sample_data()
    if "distribution_plot" not in st.session_state:
        save_plot_sampledistribution_rawdata()


st.markdown("### To start a new session:")

if st.button("New Session: Import new dataset"):
    empty_session_state()
    st.rerun()
