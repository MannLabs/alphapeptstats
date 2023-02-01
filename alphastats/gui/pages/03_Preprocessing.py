import streamlit as st
import pandas as pd
from alphastats.gui.utils.ui_helper import sidebar_info
import datetime


def preprocessing():

    st.markdown(
        "Before analyzing your data, consider normalizing and imputing your data as well as the removal of contaminants. "
        + "A more detailed description about the preprocessing methods can be found in the AlphaPeptStats [documentation](https://alphapeptstats.readthedocs.io/en/main/data_preprocessing.html)."
    )

    with st.form("preprocessing"):
        dataset = st.session_state["dataset"]

        remove_contaminations = st.selectbox(
            f"Remove contaminations annotated in {dataset.filter_columns}",
            options=[True, False],
        )

        subset = st.selectbox(
            "Subset data so it matches with metadata. Remove miscellanous samples in rawinput.",
            options=[True, False],
        )

        normalization = st.selectbox(
            "Normalization", options=[None, "zscore", "quantile", "vst", "linear"]
        )

        imputation = st.selectbox(
            "Imputation", options=[None, "mean", "median", "knn", "randomforest"]
        )

        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state.dataset.preprocess(
            remove_contaminations=remove_contaminations,
            subset=subset,
            normalization=normalization,
            imputation=imputation,
        )
        preprocessing = st.session_state.dataset.preprocessing_info
        st.info(
            "Data has been processed. "
            + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
        st.dataframe(
            pd.DataFrame.from_dict(preprocessing, orient="index").astype(str),
            use_container_width=True,
        )


def reset_preprocessing():
    st.session_state.dataset.create_matrix()
    preprocessing = st.session_state.dataset.preprocessing_info
    st.info(
        "Data has been reset. " + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )
    st.dataframe(
        pd.DataFrame.from_dict(preprocessing, orient="index").astype(str),
        use_container_width=True,
    )


def main_preprocessing():

    if "dataset" in st.session_state:
        preprocessing()

        reset_steps = st.button("Reset all Preprocessing steps")
        if reset_steps:
            reset_preprocessing()

    else:
        st.info("Import Data first")


st.markdown("### Preprocessing")
sidebar_info()

main_preprocessing()


def plot_intensity_distribution():
    st.selectbox(
        "Sample", options=st.session_state.dataset.metadata["sample"].to_list()
    )
