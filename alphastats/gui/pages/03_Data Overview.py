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

    return processed_df


def display_matrix():

    text = (
        "Normalization: "
        + str(st.session_state.dataset.preprocessing_info["Normalization"])
        + ", Imputation: "
        + str(st.session_state.dataset.preprocessing_info["Imputation"])
    )

    st.markdown("### DataFrame used for analysis **preview**")
    st.markdown(text)

    df = get_display_matrix()
    csv = convert_df(st.session_state.dataset.mat)

    st.dataframe(df)

    st.download_button(
        "Download as .csv", csv, "analysis_matrix.csv", "text/csv", key="download-csv"
    )


if "dataset" in st.session_state:
    st.markdown("## DataSet overview")

    display_matrix()

    st.markdown("#### Intensity distribution raw data per sample")
    st.plotly_chart(
        st.session_state.distribution_plot.update_layout(plot_bgcolor="white")
    )

    st.markdown("#### Intensity distribution data per sample used for analysis")
    fig_processed = st.session_state.dataset.plot_sampledistribution()
    st.plotly_chart(fig_processed.update_layout(plot_bgcolor="white"))


else:
    st.info("Import Data first")
