import io
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from gui.utils.analysis import ANALYSIS_OPTIONS

from alphastats.gui.utils.ui_helper import StateKeys, convert_df


def display_figure(plot):
    """
    display plotly or seaborn figure
    """
    try:
        st.plotly_chart(plot.update_layout(plot_bgcolor="white"))
    except Exception:
        st.pyplot(plot)


def save_plot_to_session_state(method, plot):
    """
    save plot with method to session state to retrieve old results
    """
    st.session_state[StateKeys.PLOT_LIST] += [(method, plot)]


def display_df(df):
    mask = df.applymap(type) != bool  # noqa: E721
    d = {True: "TRUE", False: "FALSE"}
    df = df.where(mask, df.replace(d))
    st.dataframe(df)


@st.fragment
def display_plot(method, analysis_result, show_save_button=True) -> None:
    """A fragment to display the plot and download options."""
    display_figure(analysis_result)

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if show_save_button and st.button("Save to results page.."):
            save_plot_to_session_state(method, analysis_result)

    with c2:
        download_figure(method, analysis_result, format="pdf")
        download_figure(method, analysis_result, format="svg")

    with c3:
        download_preprocessing_info(
            method, analysis_result
        )  # TODO this should go elsewhere


def download_figure(method, plot, format):
    """
    download plotly figure
    """

    filename = method + "." + format

    buffer = io.BytesIO()

    try:  # plotly
        plot.write_image(file=buffer, format=format)
    except AttributeError:
        plot.savefig(buffer, format=format)

    st.download_button(label="Download as " + format, data=buffer, file_name=filename)


def download_preprocessing_info(method, plot):
    preprocesing_dict = plot.preprocessing
    df = pd.DataFrame(preprocesing_dict.items())
    filename = "plot" + method + "preprocessing_info.csv"
    csv = convert_df(df)
    st.download_button(
        "Download DataSet Info as .csv",
        csv,
        filename,
        "text/csv",
    )


def gather_parameters_and_do_analysis(
    analysis_name: str,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """Extract plotting options and display.

    Returns a tuple(figure, analysis_object, parameters) where figure is the plot,
    analysis_object is the underlying object, parameters is a dictionary of the parameters used.

    Currently, analysis_object is only not-None for Volcano Plot.
    """
    if (analysis_class := ANALYSIS_OPTIONS.get(analysis_name)) is not None:
        analysis = analysis_class(st.session_state[StateKeys.DATASET])
        analysis.show_widget()
        if st.button("Run analysis .."):
            with st.spinner("Running analysis .."):
                return analysis.gather_parameters_and_do_analysis()

    raise ValueError(f"Analysis method {analysis_name} not found.")
