import streamlit as st
import pandas as pd

from alphastats.DataSet_Preprocess import PreprocessingStateKeys
from alphastats.DataSet import DataSet
from alphastats.gui.utils.ui_helper import convert_df


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def get_sample_histogram_matrix():
    return st.session_state.dataset.plot_samplehistograms()


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def get_intensity_distribution_unprocessed():
    return st.session_state.dataset.plot_sampledistribution(use_raw=True)


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def get_intensity_distribution_processed():
    return st.session_state.dataset.plot_sampledistribution()


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def get_display_matrix():
    processed_df = pd.DataFrame(
        st.session_state.dataset.mat.values,
        index=st.session_state.dataset.mat.index.to_list(),
    ).head(10)

    return processed_df


def display_matrix():
    text = (
        "Normalization: "
        + str(
            st.session_state.dataset.preprocessing_info[
                PreprocessingStateKeys.NORMALIZATION
            ]
        )
        + ", Imputation: "
        + str(
            st.session_state.dataset.preprocessing_info[
                PreprocessingStateKeys.IMPUTATION
            ]
        )
        + ", Log2-transformed: "
        + str(
            st.session_state.dataset.preprocessing_info[
                PreprocessingStateKeys.LOG2_TRANSFORMED
            ]
        )
    )

    st.markdown("**DataFrame used for analysis** *preview*")
    st.markdown(text)

    df = get_display_matrix()
    csv = convert_df(st.session_state.dataset.mat)

    st.dataframe(df)

    st.download_button(
        "Download as .csv", csv, "analysis_matrix.csv", "text/csv", key="download-csv"
    )


def display_loaded_dataset(dataset: DataSet) -> None:
    st.markdown(f"*Preview:* Raw data from {dataset.software}")
    st.dataframe(dataset.rawinput.head(5))

    st.markdown("*Preview:* Metadata")
    st.dataframe(dataset.metadata.head(5))

    st.markdown("*Preview:* Matrix")

    df = pd.DataFrame(
        dataset.mat.values,
        index=dataset.mat.index.to_list(),
    ).head(5)

    st.dataframe(df)
