import streamlit as st
import pandas as pd
import io

try:
    from alphastats.gui.utils.ui_helper import (
        sidebar_info,
        init_session_state,
        convert_df,
    )
except ModuleNotFoundError:
    from utils.ui_helper import sidebar_info


def display_plotly_figure(plot):
    st.plotly_chart(plot)


def save_plotly(plot, name, format):
    # Create an in-memory buffer
    buffer = io.BytesIO()
    # Save the figure as a pdf to the buffer
    plot.write_image(file=buffer, format=format)
    st.download_button(
        label="Download as " + format, data=buffer, file_name=name + "." + format
    )


def download_preprocessing_info(plot, name, count):
    preprocesing_dict = plot.preprocessing
    df = pd.DataFrame(preprocesing_dict.items())
    filename = "plot" + name + "preprocessing_info.csv"
    csv = convert_df(df)
    print("preprocessing" + count)
    st.download_button(
        "Download DataSet Info as .csv",
        csv,
        filename,
        "text/csv",
        key="preprocessing" + count,
    )


init_session_state()
sidebar_info()

st.markdown("### Results")


if "plot_list" in st.session_state:
    for count, plot in enumerate(st.session_state.plot_list):
        print("plot", type(plot), count)
        count = str(count)

        st.markdown("\n\n")
        name = plot[0]
        plot = plot[1]
        if name == "ttest":
            plot = plot.plot
        st.write(name)

        display_plotly_figure(plot)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            save_plotly(plot, name + count, format="pdf")

        with col2:
            save_plotly(plot, name + count, format="svg")

        with col3:
            download_preprocessing_info(plot, name, count)
else:
    st.info("No analysis performed yet.")
