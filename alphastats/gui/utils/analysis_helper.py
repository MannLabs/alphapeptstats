import io
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from plotly.graph_objects import Figure
from stqdm import stqdm

from alphastats.dataset.keys import Cols
from alphastats.gui.utils.analysis import (
    ANALYSIS_OPTIONS,
    DifferentialExpressionTwoGroupsResult,
    NewAnalysisOptions,
    PlottingOptions,
    ResultComponent,
    StatisticOptions,
)
from alphastats.gui.utils.llm_helper import LLM_ENABLED_ANALYSIS
from alphastats.gui.utils.state_keys import SavedAnalysisKeys, StateKeys
from alphastats.gui.utils.ui_helper import (
    has_llm_support,
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
    editable_annotation: bool = True,
) -> None:
    """A fragment to display a statistical analysis and download options."""

    if analysis_method in PlottingOptions.get_values():
        display_function = display_figure
        download_function = _show_buttons_download_figure
    elif analysis_method in StatisticOptions.get_values():
        display_function = _display_df
        download_function = show_button_download_df
    elif analysis_method in NewAnalysisOptions.get_values():
        display_function = display_results
        download_function = _show_buttons_download_results
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
        editable_annotation=editable_annotation,
    )


def display_results(
    results: ResultComponent, editable_annotation: bool, name: str
) -> None:
    if name is None:
        name = "tmp"
    display_column, widget_column = st.columns((1, 1))
    results.display_object(
        st_display_column=display_column,
        st_widget_column=widget_column,
        data_annotation_editable=editable_annotation,
        display_editable=True,
        name=name,
    )


def _display(
    analysis_result: Union[PlotlyObject, pd.DataFrame, ResultComponent],
    *,
    analysis_method: str,
    display_function: Callable,
    download_function: Callable,
    parameters: Dict,
    name: str,
    show_save_button: bool,
    editable_annotation: bool,
) -> None:
    """Display analysis results and download options."""
    if name is None:
        name = analysis_method

    name_pretty = name.replace(" ", "_").lower()

    if isinstance(analysis_result, ResultComponent):
        display_function(
            analysis_result, editable_annotation=editable_annotation, name=name
        )
    else:
        display_function(analysis_result)

    if analysis_method in LLM_ENABLED_ANALYSIS:
        st.markdown(
            "This analysis can be interpreted with help from the LLM. Save to results to continue."
        )

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if show_save_button and st.button(
            "💾 Save analysis to results page..",
            help="This will save the analysis to the results page"
            + " and allow LLM interpretation."
            if has_llm_support()
            else "",
        ):
            _save_analysis_to_session_state(
                analysis_result, analysis_method, parameters
            )
            st.toast("Saved to results page!", icon="✅")
            if has_llm_support() and isinstance(
                analysis_result, DifferentialExpressionTwoGroupsResult
            ):
                st.page_link("pages_/06_LLM.py", label="➔ Continue with LLM analysis")

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
        st.plotly_chart(plot.update(), key=str(id(plot)))
    except Exception:
        st.pyplot(plot)


def _show_buttons_download_results(analysis_result: ResultComponent, name: str) -> None:
    """Show buttons to download results as pdf, svg or csv."""
    _show_buttons_download_figure(analysis_result.plot, name)
    show_button_download_df(
        analysis_result.dataframe,
        file_name=f"{name}_raw_results.csv",
        label="Download raw results as .csv",
    )
    show_button_download_df(
        analysis_result.annotated_dataframe,
        file_name=f"{name}_anotated_results.csv",
        label="Download annotated results as .csv",
    )


def _show_buttons_download_figure(
    analysis_result: Union[PlotlyObject, Figure], name: str
) -> None:
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
    analysis_results: Union[PlotlyObject, pd.DataFrame, ResultComponent],
    method: str,
    parameters: Dict,
):
    """Save analysis with method and parameters to session state to show on results page."""
    analysis_key = datetime.now()  # could depend on data and parameters one day
    st.session_state[StateKeys.SAVED_ANALYSES][analysis_key] = {
        SavedAnalysisKeys.RESULT: deepcopy(analysis_results),
        SavedAnalysisKeys.METHOD: method,
        SavedAnalysisKeys.PARAMETERS: parameters,
        # TODO number will be given twice if user removes analysis
        SavedAnalysisKeys.NUMBER: len(st.session_state[StateKeys.SAVED_ANALYSES]) + 1,
    }


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


def gather_uniprot_data(features: List[str]) -> None:
    """
    Gathers UniProt data for a list of features and stores it in the session state.

    Features that are already in the session state are skipped.

    Args:
        features (List[str]): A list of features for which UniProt data needs to be gathered.
    Returns:
        None
    """
    features_to_fetch = [
        feature
        for feature in features
        if feature not in st.session_state[StateKeys.ANNOTATION_STORE]
    ]
    for feature in stqdm(
        features_to_fetch,
        desc="Retrieving uniprot data on selected features ...",
        mininterval=1,
    ):
        # TODO: Add some kind of rate limitation to avoid being locked out by uniprot
        st.session_state[StateKeys.ANNOTATION_STORE][feature] = (
            get_annotations_for_feature(feature)
        )


def get_regulated_features(analysis_object: ResultComponent) -> list:
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
        for feature, significance in zip(
            analysis_object.annotated_dataframe[Cols.INDEX],
            analysis_object.annotated_dataframe["significant"],
        )
        if significance != "non_sig"
    ]
    return regulated_features
