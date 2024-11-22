import io
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from alphastats.dataset.keys import Cols
from alphastats.gui.utils.analysis import (
    ANALYSIS_OPTIONS,
    PlottingOptions,
    StatisticOptions,
)
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    show_button_download_df,
)
from alphastats.llm.uniprot_utils import get_annotations_for_feature
from alphastats.plots.plot_utils import PlotlyObject


@st.fragment
def display_analysis_result_with_buttons(
    df: pd.DataFrame,
    analysis_method: str,
    parameters: Optional[Dict],
    show_save_button=True,
    name: str = None,
) -> None:
    """A fragment to display a statistical analysis and download options."""

    if analysis_method in PlottingOptions.get_values():
        display_function = display_figure
        download_function = _show_buttons_download_figure
    elif analysis_method in StatisticOptions.get_values():
        display_function = _display_df
        download_function = show_button_download_df
    else:
        raise ValueError(f"Analysis method {analysis_method} not found.")

    _display(
        df,
        analysis_method=analysis_method,
        parameters=parameters,
        show_save_button=show_save_button,
        name=name,
        display_function=display_function,
        download_function=download_function,
    )


def _display(
    analysis_result: Union[PlotlyObject, pd.DataFrame],
    *,
    analysis_method: str,
    display_function: Callable,
    download_function: Callable,
    parameters: Dict,
    name: str,
    show_save_button: bool,
) -> None:
    """Display analysis results and download options."""
    display_function(analysis_result)

    c1, c2, c3 = st.columns([1, 1, 1])

    if name is None:
        name = analysis_method

    name_pretty = name.replace(" ", "_").lower()
    with c1:
        if show_save_button and st.button("Save to results page.."):
            _save_analysis_to_session_state(
                analysis_result, analysis_method, parameters
            )
            st.toast("Saved to results page!", icon="✅")

    with c2:
        download_function(
            analysis_result,
            name_pretty,
        )

    with c3:
        _show_button_download_analysis_and_preprocessing_info(
            analysis_method, analysis_result, parameters, name_pretty
        )


def display_figure(plot: PlotlyObject) -> None:
    """Display plotly or seaborn figure."""
    try:
        # calling plot.update_layout is vital here as it enables the savefig function to work
        st.plotly_chart(plot.update_layout(plot_bgcolor="white"))
    except Exception:
        st.pyplot(plot)


def _show_buttons_download_figure(analysis_result: PlotlyObject, name: str) -> None:
    """Show buttons to download figure as .pdf or .svg."""
    # TODO We have to check for all scatter plotly figures, which renderer they use.
    #  Default is webgl, which is good for browser performance, but looks horrendous in svg download
    #  rerendering with svg as renderer could be a method of PlotlyObject to invoke prior to saving as svg
    _show_button_download_figure(analysis_result, name, "pdf")
    _show_button_download_figure(analysis_result, name, "svg")


def _show_button_download_figure(
    plot: PlotlyObject,
    file_name: str,
    file_format: str,
) -> None:
    """Show a button to download a figure."""

    buffer = io.BytesIO()

    try:  # plotly
        plot.write_image(file=buffer, format=file_format)
    except AttributeError:  # seaborn
        plot.savefig(buffer, format=file_format)

    st.download_button(
        label="Download as ." + file_format,
        data=buffer,
        file_name=file_name + "." + file_format,
    )


# TODO: use pandas stylers, rather than changing the data
def _display_df(df: pd.DataFrame) -> None:
    """Display a dataframe."""
    mask = df.applymap(type) != bool  # noqa: E721
    df = df.where(mask, df.replace({True: "TRUE", False: "FALSE"}))
    st.dataframe(df)


def _show_button_download_analysis_and_preprocessing_info(
    method: str,
    analysis_result: Union[PlotlyObject, pd.DataFrame],
    parameters: Dict,
    name: str,
):
    """Download analysis info (= analysis and preprocessing parameters) as .csv."""
    parameters_pretty = {
        f"analysis_parameter__{k}": "None" if v is None else v
        for k, v in parameters.items()
    }

    if method in PlottingOptions.get_values():
        dict_to_save = {
            **analysis_result.preprocessing,
            **parameters_pretty,
        }  # TODO why is the preprocessing info saved in the plots?
    else:
        dict_to_save = parameters_pretty

    show_button_download_df(
        pd.DataFrame(dict_to_save.items()),
        file_name=f"analysis_info__{name}",
        label="Download analysis info as .csv",
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
            st.toast("Analysis done!", icon="✅")
            return result
        return None, None, None

    else:
        raise ValueError(f"Analysis method {analysis_method} not found.")


def gather_uniprot_data(features: list) -> None:
    """
    Gathers UniProt data for a list of features and stores it in the session state.

    Features that are already in the session state are skipped.

    Args:
        features (list): A list of features for which UniProt data needs to be gathered.
    Returns:
        None
    """
    for feature in features:
        if feature in st.session_state[StateKeys.ANNOTATION_STORE]:
            continue
        # TODO: Add some kind of rate limitation to avoid being locked out by uniprot
        st.session_state[StateKeys.ANNOTATION_STORE][feature] = (
            get_annotations_for_feature(feature)
        )


def get_regulated_features(analysis_object: PlotlyObject) -> list:
    """
    Retrieve regulated features from the analysis object.
    This function extracts features that are labeled (i.e., have a non-empty label)
    from the analysis results. It is specifically designed to work with volcano plots.
    Args:
        analysis_object (PlotlyObject): An object containing analysis results,
                                        including feature indices and labels.
    Returns:
        list: A list of regulated features that have non-empty labels.
    """
    # TODO: add a method to the AbstractAnalysis class to retrieve regulated features upon analysis to store in the session state. This function here only works for volcano plots.
    regulated_features = [
        feature
        for feature, label in zip(
            analysis_object.res[Cols.INDEX], analysis_object.res["label"]
        )
        if label != ""
    ]
    return regulated_features
