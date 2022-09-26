import streamlit as st
from alphastats.gui.utils.ui_helper import sidebar_info
from alphastats.gui.utils.analysis_helper import get_analysis


def display_df(df):
    st.dataframe(df)


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


st.markdown("### Statistical Analysis")

sidebar_info()

if "dataset" in st.session_state:

    st.selectbox(
        "Statistical Analysis",
        options=list(st.session_state.statistic_options.keys()),
        key="statistic",
    )  # Pass index as ke
    df = get_analysis(
        method=st.session_state.statistic,
        options_dict=st.session_state.statistic_options,
    )
    if df is not None:

        display_df(df)

        filename = st.session_state.statistic + ".csv"
        csv = convert_df(df)
        st.download_button(
            "Download as .csv", csv, filename, "text/csv", key="download-csv"
        )


else:
    st.info("Import Data first")


def show_dataset_overview():
    st.session_state.dataset.print
