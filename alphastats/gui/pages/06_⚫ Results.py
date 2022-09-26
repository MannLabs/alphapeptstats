import streamlit as st
import pandas as pd
import io
from alphastats.gui.utils.ui_helper import sidebar_info


def display_plotly_figure(plot):
    st.plotly_chart(plot)


def save_plotly(plot, format):
    # Create an in-memory buffer
    buffer = io.BytesIO()
    # Save the figure as a pdf to the buffer
    plot[1].write_image(file=buffer, format=format)
    st.download_button(
        label="Download as " + format, data=buffer, file_name=plot[0] + "." + format
    )


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


def download_preprocessing_info(plot, count):
    preprocesing_dict = plot[1].preprocessing
    df = pd.DataFrame(preprocesing_dict.items())
    filename = "plot" + plot[0] + "preprocessing_info.csv"
    csv = convert_df(df)
    st.download_button(
        "Download DataSet Info as .csv",
        csv,
        filename,
        "text/csv",
        key="preprocessing" + count,
    )


st.markdown("### Results")

sidebar_info()

if "plot_list" in st.session_state:
    for count, plot in enumerate(st.session_state.plot_list):
        count = str(count)

        st.markdown("\n\n")
        st.write(plot[0])

        display_plotly_figure(plot[1])

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            save_plotly(plot, format="pdf")

        with col2:
            save_plotly(plot, format="svg")

        with col3:
            download_preprocessing_info(plot, count)


else:
    st.info("No analysis performed yet.")
