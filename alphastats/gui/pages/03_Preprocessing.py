import streamlit as st
import pandas as pd

import datetime


def preprocessing():

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "Before analyzing your data, consider normalizing and imputing your data as well as the removal of contaminants. "
            + "A more detailed description about the preprocessing methods can be found in the AlphaPeptStats " 
            + "[documentation](https://alphapeptstats.readthedocs.io/en/main/data_preprocessing.html)."
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

            remove_samples = st.multiselect(
                "Remove samples from analysis", 
                options=st.session_state.dataset.metadata[st.session_state.dataset.sample].to_list()
            )

            data_completeness = st.number_input(
                f"Data completeness across samples cut-off \n(0.7 -> protein has to be detected in at least 70% of the samples)",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
            )

            log2_transform = st.selectbox(
                "Log2-transform dataset", options=[True, False],
            )

            normalization = st.selectbox(
                "Normalization", options=[None, "zscore", "quantile", "vst", "linear"]
            )

            imputation = st.selectbox(
                "Imputation", options=[None, "mean", "median", "knn", "randomforest"]
            )

            submitted = st.form_submit_button("Submit")

        if submitted:
            if len(remove_samples) == 0:
                remove_samples = None
            
            st.session_state.dataset.preprocess(
                remove_contaminations=remove_contaminations,
                log2_transform=log2_transform,
                remove_samples = remove_samples,
                data_completeness=data_completeness,
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
        
        st.markdown("#### Batch correction: correct for technical bias")

        with st.form("Batch correction: correct for technical bias"):
            batch = st.selectbox(
                "Batch", 
                options= st.session_state.dataset.metadata.columns.to_list()
            )
            submit_batch_correction = st.form_submit_button("Submit")
        
        if submit_batch_correction:
            st.session_state.dataset.batch_correction(
                batch=batch
            )
            st.info(
                "Data has been processed. "
                + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )

    
    with c2:

        if submitted:
            st.markdown("**Intensity Distribution after preprocessing per sample**")
            fig_processed = st.session_state.dataset.plot_sampledistribution()
            st.plotly_chart(fig_processed.update_layout(plot_bgcolor="white"), use_container_width=True)
        
        else:
            st.markdown("**Intensity Distribution per sample**")
            fig_none_processed = st.session_state.dataset.plot_sampledistribution()
            st.plotly_chart(fig_none_processed.update_layout(plot_bgcolor="white"), use_container_width=True)
        

    reset_steps = st.button("Reset all Preprocessing steps")

    if reset_steps:
        reset_preprocessing()


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

    else:
        st.info("Import Data first")


st.markdown("### Preprocessing")


main_preprocessing()


def plot_intensity_distribution():
    st.selectbox(
        "Sample", options=st.session_state.dataset.metadata["sample"].to_list()
    )