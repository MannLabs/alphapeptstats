import io
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from alphastats.gui.utils.analysis import (
    PCAPlotAnalysis,
    PlottingOptions,
    TSNEPlotAnalysis,
    UMAPPlotAnalysis,
    VolcanoPlotAnalysis,
)
from alphastats.gui.utils.ui_helper import StateKeys, convert_df
from alphastats.keys import Cols


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


def get_unique_values_from_column(column):
    unique_values = (
        st.session_state[StateKeys.DATASET].metadata[column].unique().tolist()
    )
    return unique_values


def st_general(method_dict):
    chosen_parameter_dict = {}

    if "settings" not in method_dict:
        # TODO: is this check really required here? If so, refactor to be part of plotting/statistic_options
        if st.session_state[StateKeys.DATASET].mat.isna().values.any():
            st.error(
                "Data contains missing values impute your data before plotting (Preprocessing - Imputation)."
            )
            return None

        st.info("No parameters to set.")

    else:
        for parameter, parameter_dict in method_dict["settings"].items():
            if "options" in parameter_dict:
                chosen_parameter = st.selectbox(
                    parameter_dict.get("label"), options=parameter_dict.get("options")
                )
            else:
                chosen_parameter = st.checkbox(parameter_dict.get("label"))

            chosen_parameter_dict[parameter] = chosen_parameter

    return chosen_parameter_dict


def do_analysis(
    method: str, options_dict: Dict[str, Any]
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    """Extract plotting options and display.

    Returns a tuple(figure, analysis_object, parameters) where figure is the plot,
    analysis_object is the underlying object, parameters is a dictionary of the parameters used.

    Currently, analysis_object is only not-None for Volcano Plot.
    # TODO unify the API of all analysis methods
    """
    options = {
        PlottingOptions.VOLCANO_PLOT: VolcanoPlotAnalysis,
        PlottingOptions.PCA_PLOT: PCAPlotAnalysis,
        PlottingOptions.UMAP_PLOT: UMAPPlotAnalysis,
        PlottingOptions.TSNE_PLOT: TSNEPlotAnalysis,
    }

    if (analysis_class := options.get(method)) is not None:
        analysis = analysis_class(st.session_state[StateKeys.DATASET])
        analysis.show_widget()
        if st.button("Run analysis .."):
            with st.spinner("Running analysis .."):
                return analysis.do_analysis()
        return None, None, {}

    method_dict = options_dict.get(method)

    # old, to be refactored logic:
    if method == "Differential Expression Analysis - T-test":
        parameters = helper_compare_two_groups()
        parameters.update({"method": "ttest"})

    elif method == "Differential Expression Analysis - Wald-test":
        parameters = helper_compare_two_groups()
        parameters.update({"method": "wald"})

    else:
        parameters = st_general(method_dict=method_dict)

    submitted = st.button("Run analysis ..")

    if submitted:
        with st.spinner("Calculating..."):
            return method_dict["function"](**parameters), None, parameters

    return None, None, {}


# TODO this can be deleted after all analysis adapted the new Pattern (cf. analysis.py:Analysis())
def helper_plot_dimensionality_reduction(method_dict):
    group = st.selectbox(
        method_dict["settings"]["group"].get("label"),
        options=method_dict["settings"]["group"].get("options"),
    )

    circle = False

    if group is not None:
        circle = st.checkbox("circle")

    chosen_parameter_dict = {
        "circle": circle,
        "group": group,
    }
    return chosen_parameter_dict


# TODO this can be deleted after all analysis adapted the new Pattern (cf. analysis.py:Analysis())
def helper_compare_two_groups():
    """
    Helper function to compare two groups for example
    Volcano Plot, Differential Expression Analysis and t-test
    selectbox based on selected column
    """
    dataset = st.session_state[StateKeys.DATASET]
    chosen_parameter_dict = {}
    default_option = "<select>"
    group = st.selectbox(
        "Grouping variable",
        options=[default_option] + dataset.metadata.columns.to_list(),
    )

    if group != default_option:
        unique_values = dataset.metadata[group].unique().tolist()

        group1 = st.selectbox("Group 1", options=unique_values)

        group2 = st.selectbox("Group 2", options=list(reversed(unique_values)))

        chosen_parameter_dict.update(
            {"column": group, "group1": group1, "group2": group2}
        )

        if group1 == group2:
            st.error(
                "Group 1 and Group 2 can not be the same please select different group."
            )

    else:
        group1 = st.multiselect(
            "Group 1 samples:",
            options=dataset.metadata[Cols.SAMPLE].to_list(),
        )

        group2 = st.multiselect(
            "Group 2 samples:",
            options=list(reversed(dataset.metadata[Cols.SAMPLE].to_list())),
        )

        intersection_list = list(set(group1).intersection(set(group2)))

        if len(intersection_list) > 0:
            st.warning(
                "Group 1 and Group 2 contain same samples: " + str(intersection_list)
            )

        chosen_parameter_dict.update({"group1": group1, "group2": group2})

    return chosen_parameter_dict
