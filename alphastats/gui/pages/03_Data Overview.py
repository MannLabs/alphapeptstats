import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache_data
def convert_df(df, user_session_id=st.session_state.user_session_id):
    return df.to_csv().encode("utf-8")


@st.cache_data
def get_display_matrix(user_session_id=st.session_state.user_session_id):

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
        + ", Log2-transformed: "
        + str(st.session_state.dataset.preprocessing_info["Log2-transformed"])
    )

    st.markdown("**DataFrame used for analysis** *preview*")
    st.markdown(text)

    df = get_display_matrix()
    csv = convert_df(st.session_state.dataset.mat)

    st.dataframe(df)

    st.download_button(
        "Download as .csv", csv, "analysis_matrix.csv", "text/csv", key="download-csv"
    )


@st.cache_data
def get_sample_histogram_matrix(user_session_id = st.session_state.user_session_id):
    return st.session_state.dataset.plot_samplehistograms()

@st.cache_data
def get_intensity_distribution_processed(user_session_id = st.session_state.user_session_id):
    return st.session_state.dataset.plot_sampledistribution()



if "dataset" in st.session_state:
    st.markdown("## DataSet overview")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("**Intensity distribution raw data per sample**")
        st.plotly_chart(
            st.session_state.distribution_plot.update_layout(plot_bgcolor="white"),
            use_container_width=True,
        )

    with c2:

        st.markdown("**Intensity distribution data per sample used for analysis**")
        st.plotly_chart(
            get_intensity_distribution_processed(user_session_id = st.session_state.user_session_id)
                        .update_layout(plot_bgcolor="white"), use_container_width=True
        )
 
    st.plotly_chart(
        get_sample_histogram_matrix(user_session_id = st.session_state.user_session_id)
                    .update_layout(plot_bgcolor="white"), use_container_width=True
    )


    display_matrix()


else:
    st.info("Import Data first")
