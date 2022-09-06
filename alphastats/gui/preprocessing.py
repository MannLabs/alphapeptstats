import streamlit as st


class Preprocessing:
    def preprocessing(self):
        if self.dataset is None:
            st.write("Load data first.")
        else:
            with st.form("preprocessing"):
                remove_contaminations = st.selectbox(
                    f"Remove contaminations annotated in {self.dataset.filter_columns}",
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
                self.dataset.preprocess(
                    remove_contaminations=remove_contaminations,
                    subset=subset,
                    normalization=normalization,
                    imputation=imputation,
                )
