import streamlit as st
from alphastats.AlphaStats import sidebar_info


def preprocessing():
    with st.form("preprocessing"):
        dataset = st.session_state["dataset"]
        remove_contaminations = st.selectbox(
                    f"Remove contaminations annotated in {dataset.filter_columns}",
                    options=[True, False],
                )
        subset = st.selectbox(
                    "Subset data so it matches with metadata. Remove miscellanous samples in rawdata.",
                    options=[True, False],
                )
        normalization = st.selectbox(
                    "Normalization", options=[None, "zscore", "quantile", "linear"]
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


def main_preprocessing():
    if "dataset" in st.session_state:
        preprocessing()
    else:
        st.info("Import Data first")


st.markdown("### Preprocessing")
sidebar_info()
#st.sidebar.image("/home/rzwitch/Downloads/randy-streamlit.png", use_column_width=True)
main_preprocessing()