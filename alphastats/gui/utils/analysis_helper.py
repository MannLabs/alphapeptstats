import io
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from alphastats.gui.utils.analysis import ANALYSIS_OPTIONS, PlottingOptions
from alphastats.gui.utils.ui_helper import StateKeys, convert_df_to_csv
from alphastats.plots.PlotUtils import PlotlyObject


def display_figure(plot):
    """
    display plotly or seaborn figure
    """
    try:
        st.plotly_chart(plot.update_layout(plot_bgcolor="white"))
    except Exception:
        st.pyplot(plot)


@st.fragment
def display_plot(
    method: str,
    plot: PlotlyObject,
    parameters: Optional[Dict] = None,
    show_save_button: bool = True,
    name: str = None,
) -> None:
    """A fragment to display the plot and download options."""
    display_figure(plot)

    c1, c2, c3 = st.columns([1, 1, 1])

    if name is None:
        name = method

    name_pretty = name.replace(" ", "_").lower()
    with c1:
        if show_save_button and st.button("Save to results page.."):
            save_analysis_to_session_state(plot, method, parameters)
            st.info("Saved to results page!")

    with c2:
        download_figure(name_pretty, plot, file_format="pdf")
        download_figure(name_pretty, plot, file_format="svg")

    with c3:
        download_analysis_and_preprocessing_info(method, plot, parameters, name_pretty)


@st.fragment
def display_df(
    method: str,
    df: pd.DataFrame,
    parameters: Optional[Dict] = None,
    show_save_button=True,
    name: str = None,
) -> None:
    """A fragment to display the statistical analysis and download options."""

    mask = df.applymap(type) != bool  # noqa: E721
    df = df.where(mask, df.replace({True: "TRUE", False: "FALSE"}))

    st.dataframe(df)

    if name is None:
        name = method
    name_pretty = name.replace(" ", "_").lower()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if show_save_button and st.button("Save to results page.."):
            save_analysis_to_session_state(df, method, parameters)
            st.info("Saved to results page!")
    with c2:
        csv = convert_df_to_csv(df)

        st.download_button(
            "Download as .csv",
            csv,
            name_pretty + ".csv",
            "text/csv",
            key="download-csv",
        )
    with c3:
        download_analysis_and_preprocessing_info(method, df, parameters, name_pretty)


def download_figure(name: str, plot: PlotlyObject, file_format: str):
    """
    download plotly figure
    """

    filename = name + "." + file_format

    buffer = io.BytesIO()

    try:  # plotly
        plot.write_image(file=buffer, format=file_format)
    except AttributeError:  # TODO figure out what else "plot" can be
        plot.savefig(buffer, format=file_format)

    st.download_button(
        label="Download as ." + file_format, data=buffer, file_name=filename
    )


def download_analysis_and_preprocessing_info(
    method: str,
    analysis_result: Union[PlotlyObject, pd.DataFrame],
    parameters: Dict,
    name: str,
):
    parameters_pretty = {f"analysis_parameter__{k}": v for k, v in parameters.items()}

    if method in PlottingOptions.get_values():
        dict_to_save = {
            **analysis_result.preprocessing,
            **parameters_pretty,
        }  # TODO why is the preprocessing info saved in the plots?
    else:
        dict_to_save = parameters_pretty

    filename = f"analysis_info__{name}.csv"
    csv = convert_df_to_csv(pd.DataFrame(dict_to_save.items()))
    st.download_button(
        "Download analysis info as .csv",
        csv,
        filename,
        "text/csv",
    )


def save_analysis_to_session_state(
    analysis_results: Union[PlotlyObject, pd.DataFrame],
    method: str,
    parameters: Dict,
):
    """Save analysis with method to session state to show on results page."""
    st.session_state[StateKeys.ANALYSIS_LIST] += [
        (
            analysis_results,
            method,
            parameters,
        )
    ]


def gather_parameters_and_do_analysis(
    analysis_method: str,
) -> Tuple[
    Optional[Union[PlotlyObject, pd.DataFrame]], Optional[Any], Optional[Dict[str, Any]]
]:
    """Extract plotting options and display.

    Returns a tuple(figure, analysis_object, parameters) where figure is the plot,
    analysis_object is the underlying object, parameters is a dictionary of the parameters used.

    Currently, analysis_object is only not-None for Volcano Plot.
    """
    if (analysis_class := ANALYSIS_OPTIONS.get(analysis_method)) is not None:
        analysis = analysis_class(st.session_state[StateKeys.DATASET])
        analysis.show_widget()
        if st.button("Run analysis .."):
            with st.spinner("Running analysis .."):
                return analysis.do_analysis()
        return None, None, None

    else:
        raise ValueError(f"Analysis method {analysis_method} not found.")
