import io
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from alphastats.gui.utils.analysis import ANALYSIS_OPTIONS, PlottingOptions
from alphastats.gui.utils.ui_helper import StateKeys, convert_df_to_csv
from alphastats.plots.PlotUtils import PlotlyObject


@st.fragment
def display_plot(
    method: str,
    plot: PlotlyObject,
    parameters: Optional[Dict] = None,
    show_save_button: bool = True,
    name: str = None,
) -> None:
    """A fragment to display a plot and download options."""
    _display(
        method,
        plot,
        parameters=parameters,
        show_save_button=show_save_button,
        name=name,
        display_method=display_figure,
        save_method=_show_buttons_download_figure,
    )


def display_figure(plot):
    """Display plotly or seaborn figure."""
    try:
        st.plotly_chart(plot.update_layout(plot_bgcolor="white"))
    except Exception:
        st.pyplot(plot)


def _show_buttons_download_figure(name_pretty, analysis_result):
    _show_button_download_figure(name_pretty, analysis_result, file_format="pdf")
    _show_button_download_figure(name_pretty, analysis_result, file_format="svg")


def _show_button_download_figure(name: str, plot: PlotlyObject, file_format: str):
    """Download figure."""

    filename = name + "." + file_format

    buffer = io.BytesIO()

    try:  # plotly
        plot.write_image(file=buffer, format=file_format)
    except AttributeError:  # TODO figure out what else "plot" can be
        plot.savefig(buffer, format=file_format)

    st.download_button(
        label="Download as ." + file_format, data=buffer, file_name=filename
    )


@st.fragment
def display_statistical_analysis(
    method: str,
    df: pd.DataFrame,
    parameters: Optional[Dict] = None,
    show_save_button=True,
    name: str = None,
) -> None:
    """A fragment to display a statistical analysis and download options."""
    _display(
        method,
        df,
        parameters=parameters,
        show_save_button=show_save_button,
        name=name,
        display_method=_display_df,
        save_method=_show_button_download_df,
    )


def _display_df(df: pd.DataFrame) -> None:
    """Display a dataframe."""
    mask = df.applymap(type) != bool  # noqa: E721
    df = df.where(mask, df.replace({True: "TRUE", False: "FALSE"}))
    st.dataframe(df)


def _show_button_download_df(
    name_pretty: str, analysis_result: pd.DataFrame, label="Download as .csv"
) -> None:
    """Download a dataframe as .csv."""
    csv = convert_df_to_csv(analysis_result)

    st.download_button(
        label,
        csv,
        name_pretty + ".csv",
        "text/csv",
        key="download-csv",
    )


def _display(
    method: str,
    analysis_result: Union[PlotlyObject, pd.DataFrame],
    *,
    display_method: Callable,
    save_method: Callable,
    parameters: Dict,
    name: str,
    show_save_button: bool,
) -> None:
    """Display analysis results and download options."""
    display_method(analysis_result)

    c1, c2, c3 = st.columns([1, 1, 1])

    if name is None:
        name = method

    name_pretty = name.replace(" ", "_").lower()
    with c1:
        if show_save_button and st.button("Save to results page.."):
            _save_analysis_to_session_state(analysis_result, method, parameters)
            st.success("Saved to results page!")

    with c2:
        save_method(name_pretty, analysis_result)

    with c3:
        _show_button_download_analysis_and_preprocessing_info(
            method, analysis_result, parameters, name_pretty
        )


def _show_button_download_analysis_and_preprocessing_info(
    method: str,
    analysis_result: Union[PlotlyObject, pd.DataFrame],
    parameters: Dict,
    name: str,
):
    """Download analysis info (= analysis and preprocessing parameters and ) as .csv."""
    parameters_pretty = {f"analysis_parameter__{k}": v for k, v in parameters.items()}

    if method in PlottingOptions.get_values():
        dict_to_save = {
            **analysis_result.preprocessing,
            **parameters_pretty,
        }  # TODO why is the preprocessing info saved in the plots?
    else:
        dict_to_save = parameters_pretty

    filename = f"analysis_info__{name}.csv"

    _show_button_download_df(
        filename, pd.DataFrame(dict_to_save.items()), "Download analysis info as .csv"
    )


def _save_analysis_to_session_state(
    analysis_results: Union[PlotlyObject, pd.DataFrame],
    method: str,
    parameters: Dict,
):
    """Save analysis with method and parameters to session state to show on results page."""
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
                result = analysis.do_analysis()
            st.success("Analysis done!")
            return result
        return None, None, None

    else:
        raise ValueError(f"Analysis method {analysis_method} not found.")
