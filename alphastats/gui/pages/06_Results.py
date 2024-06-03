import streamlit as st
import pandas as pd
import io

try:
    from alphastats.gui.utils.ui_helper import sidebar_info
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


@st.cache_data
def convert_df(df, user_session_id=st.session_state.user_session_id):
    return df.to_csv().encode("utf-8")


def download_preprocessing_info(plot, name, count):
    preprocesing_dict = plot.preprocessing
    df = pd.DataFrame(preprocesing_dict.items())
    filename = "plot" + name + "preprocessing_info.csv"
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
