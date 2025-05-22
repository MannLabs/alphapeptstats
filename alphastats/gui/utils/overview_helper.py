import pandas as pd
import streamlit as st

from alphastats.dataset.dataset import DataSet
from alphastats.gui.utils.state_keys import StateKeys
from alphastats.gui.utils.ui_helper import show_button_download_df


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def get_sample_histogram_matrix():
    return st.session_state[StateKeys.DATASET].plot_samplehistograms()


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def get_intensity_distribution_unprocessed():
    return st.session_state[StateKeys.DATASET].plot_sampledistribution(use_raw=True)


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def get_intensity_distribution_processed():
    return st.session_state[StateKeys.DATASET].plot_sampledistribution()


def display_matrix():
    st.markdown("**DataFrame used for analysis** *preview*")

    # TODO why not use the actual matrix here?
    mat = st.session_state[StateKeys.DATASET].mat
    df = pd.DataFrame(
        mat.values,
        index=mat.index.to_list(),
    ).head(10)

    st.dataframe(df)

    show_button_download_df(mat, file_name="analysis_matrix")


def display_loaded_dataset(dataset: DataSet) -> None:
    st.markdown(f"*Preview:* Raw data from {dataset.software}")
    st.dataframe(dataset.rawinput.head(5))

    st.markdown("*Preview:* Metadata")
    st.dataframe(dataset.metadata.head(5))

    st.markdown("*Preview:* Matrix")

    df = pd.DataFrame(
        dataset.rawmat.values,
        index=dataset.rawmat.index.to_list(),
    ).head(5)

    st.dataframe(df)
