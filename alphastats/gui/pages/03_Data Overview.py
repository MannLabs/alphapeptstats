
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


@st.cache
def get_display_matrix():

    processed_df = pd.DataFrame(
            st.session_state.dataset.mat.values,
            index=st.session_state.dataset.mat.index.to_list(),
        ).head(10)
    

    csv = convert_df(processed_df)

    return processed_df, csv


def display_matrix():

    text = "Normalization: " + str(st.session_state.dataset.preprocessing_info["Normalization"]) + \
        ", Imputation: " + str(st.session_state.dataset.preprocessing_info["Imputation"])

    st.markdown("### DataFrame used for analysis")
    st.markdown(text)

    df, csv  = get_display_matrix()
    
    st.dataframe(df)

    st.download_button(
        "Download as .csv", csv, "analysis_matrix.csv", "text/csv", key="download-csv"
    )

    

if "dataset" in st.session_state:
    st.markdown("## DataSet overview")

    st.markdown("#### Intensity distribution raw data per sample")
    st.plotly_chart(st.session_state.distribution_plot.update_layout(plot_bgcolor="white"))

    st.markdown("#### Intensity distribution processed data per sample")
    fig_processed = st.session_state.dataset.plot_sampledistribution()
    st.plotly_chart(fig_processed.update_layout(plot_bgcolor="white"))
    
     #   st.markdown("### Intensity distribution processed data per sample")
    #if st.session_state.dataset.preprocessed:

    
    display_matrix()
    
    # Display Missing values / Imputed values
        


    



else:
    st.info("Import Data first")