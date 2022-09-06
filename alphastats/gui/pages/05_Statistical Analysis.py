from tracemalloc import Statistic
import streamlit as st
from alphastats.gui.utils.ui_helper import sidebar_info

from alphastats.gui.utils.analysis_helper import get_analysis


def display_df(df):
    st.dataframe(df)

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

st.markdown("### Statistical Analysis")

sidebar_info()

if "dataset" in st.session_state:

    from alphastats.gui.utils.options import statistic_options as options_dict

    if "n_rows" not in st.session_state:
        st.session_state.n_rows = 1
    add = st.button(label="add")

    if add:
        st.session_state.n_rows += 1
        st.experimental_rerun()

    for i in range(st.session_state.n_rows):
        # add text inputs here
        statistic = st.selectbox(
            "Statistical Analysis",
            options=list(options_dict.keys()),
            key= "statistic" + str(i),
        )  # Pass index as ke
        df =  get_analysis(method=statistic, options_dict=options_dict)
        if df is not None:
            display_df(df)
            
            filename = statistic + ".csv"
            csv = convert_df(df)
            st.download_button(
            "Download as .csv",
            csv,
            filename,
            "text/csv",
            key='download-csv'
            )


else:
    st.info("Import Data first")


def show_dataset_overview():
    st.session_state.dataset.print
