from typing import List

import pandas as pd
import streamlit as st
from st_cytoscape import cytoscape

from alphastats.dataset.dataset import DataSet
from alphastats.dataset.keys import Cols

CYTOSCAPE_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "shape": "roundrectangle",
            "width": 200,
            "height": 60,
            "text-valign": "center",
            "text-halign": "center",
        },
    },
    {
        "selector": "node.active",
        "style": {
            "background-color": "lightgreen",
        },
    },
    {
        "selector": "node.inactive",
        "style": {
            "background-color": "lightgrey",
        },
    },
    {
        "selector": "node.selected",  # TODO: This currently does not work, figure out why
        "style": {"background-color": "red"},
    },
    {
        "selector": "edge",
        "style": {
            "width": 3,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
        },
    },
]


class PREPROCESSING_STEPS:
    """String constants for accessing preprocessing steps.

    The strings map to the keyword arguments of the alphastat.Dataset.preprocess method.
    """

    REMOVE_CONTAMINATIONS = "remove_contaminations"
    REMOVE_SAMPLES = "remove_samples"
    SUBSET = "subset"
    REPLACE_ZEROES = "replace_zeroes"
    DATA_COMPLETENESS = "data_completeness"
    LOG2_TRANSFORM = "log2_transform"
    NORMALIZATION = "normalization"
    IMPUTATION = "imputation"
    DROP_UNMEASURED_FEATURES = "drop_unmeasured_features"
    BATCH = "batch"


# TODO: Make help texts meaningful
# TODO: Show help texts on the widgets
UI_ELEMENTS = {
    PREPROCESSING_STEPS.REMOVE_CONTAMINATIONS: {
        "repr": "Remove contaminations",
        "help": "Remove contaminations annotated in the contaminations library and filter columns included in the dataset.",
    },
    PREPROCESSING_STEPS.REMOVE_SAMPLES: {
        "repr": "Remove samples",
        "help": "Remove samples from analysis, e.g. useful when failed or blank runs are included.",
    },
    PREPROCESSING_STEPS.SUBSET: {
        "repr": "Subset data",
        "help": "Subset data so it matches with metadata. Can for example be useful if several dimensions of an experiment were analysed together.",
    },
    PREPROCESSING_STEPS.REPLACE_ZEROES: {
        "repr": "0 --> NaN",
        "help": "Replace 0 in the data with NaN.",
    },
    PREPROCESSING_STEPS.DATA_COMPLETENESS: {
        "repr": "Filter data completeness",
        "help": "Filter data based on completeness across samples. E.g. if a protein has to be detected in at least 70% of the samples.",
    },
    PREPROCESSING_STEPS.LOG2_TRANSFORM: {
        "repr": "Log2 transform",
        "help": "Log2-transform dataset.",
    },
    PREPROCESSING_STEPS.NORMALIZATION: {
        "repr": "Normalization",
        "help": 'Normalize data using one of the available methods ("zscore", "quantile", "vst", "linear").',
    },
    PREPROCESSING_STEPS.IMPUTATION: {
        "repr": "Imputation",
        "help": 'Impute missing values using one of the available methods ("mean", "median", "knn", "randomforest").',
    },
    PREPROCESSING_STEPS.DROP_UNMEASURED_FEATURES: {
        "repr": "Drop empty proteins",
        "help": "Drop unmeasured features (protein groups), i.e. ones that are all NaNs or Infs.",
    },
    PREPROCESSING_STEPS.BATCH: {
        "repr": "Batch correction",
        "help": "Batch correction.",
    },
}

PREDEFINED_ORDER = [
    PREPROCESSING_STEPS.REMOVE_CONTAMINATIONS,
    PREPROCESSING_STEPS.REMOVE_SAMPLES,
    PREPROCESSING_STEPS.SUBSET,
    PREPROCESSING_STEPS.REPLACE_ZEROES,
    PREPROCESSING_STEPS.DATA_COMPLETENESS,
    PREPROCESSING_STEPS.LOG2_TRANSFORM,
    PREPROCESSING_STEPS.NORMALIZATION,
    PREPROCESSING_STEPS.IMPUTATION,
    PREPROCESSING_STEPS.DROP_UNMEASURED_FEATURES,
    PREPROCESSING_STEPS.BATCH,
]


def draw_workflow(workflow: List[str], order: List[str] = PREDEFINED_ORDER):
    """Draws a workflow using the given workflow and order of elements.

    The order defines which elements are shown and in which order. The workflow defines which elements are visually highlighted and connected by arrows.

    Args:
        workflow (list[str]): List of keys of the workflow elements.
        order (list[str], optional): Order of the elements. Defaults to PREDEFINED_ORDER.

    Returns:
        st.cytoscape: Cytoscape element representing the workflow. The value if this are selected nodes.
    """

    elements = [
        {
            "group": "nodes",
            "data": {
                "id": i,
                "label": UI_ELEMENTS[key]["repr"],
                "key": key,
            },
            "selectable": True,
            "classes": ["active"] if key in workflow else ["inactive"],
        }
        for i, key in enumerate(order)
    ]

    for key1, key2 in zip(workflow[:-1], workflow[1:]):
        i = order.index(key1)
        j = order.index(key2)
        elements.append(
            {
                "group": "edges",
                "data": {"id": f"{i}_{j}", "source": i, "target": j},
                "selectable": False,
            }
        )

    selected = cytoscape(
        elements,
        CYTOSCAPE_STYLESHEET,
        layout={"name": "grid", "columns": 1},
        selection_type="single",
        user_panning_enabled=False,
        user_zooming_enabled=False,
        height=f"{len(order)*80}px",
        key="predefined_workflow",
    )

    return selected


def configure_preprocessing(dataset):
    """Serve the preprocessing configuration widgets.

    Args:
        dataset (alphastat.Dataset): The dataset to be preprocessed.

    Returns:
        dict: Dictionary containing the preprocessing settings."""

    st.markdown(
        "Before analyzing your data, consider normalizing and imputing your data as well as the removal of contaminants. "
        + "A more detailed description about the preprocessing methods can be found in the AlphaPeptStats "
        + "[documentation](https://alphapeptstats.readthedocs.io/en/main/data_preprocessing.html)."
    )

    remove_contaminations = st.checkbox(
        f"Remove contaminations annotated in {dataset.filter_columns}",
        value=True,
    )

    subset = st.checkbox(
        "Subset data so it matches with metadata. Remove miscellanous samples in rawinput.",
        value=False,
    )

    # TODO: value of this widget does not persist across dataset reset (likely because the metadata is reset)
    remove_samples = st.multiselect(
        "Remove samples from analysis",
        options=dataset.metadata[Cols.SAMPLE].to_list(),
    )
    remove_samples = remove_samples if len(remove_samples) != 0 else None

    replace_zeroes = st.checkbox(
        "Replace 0 in the data with NaN.",
        value=True,
    )

    data_completeness = st.number_input(
        "Data completeness across samples cut-off \n(0.7 -> protein has to be detected in at least 70% of the samples)",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )

    log2_transform = st.checkbox(
        "Log2-transform dataset. Note: If this is skipped it weill be performed on the fly for select analyses (e.g. Volcano plot).",
        value=True,
    )

    normalization = st.selectbox(
        "Normalization", options=[None, "zscore", "quantile", "vst", "linear"]
    )

    imputation = st.selectbox(
        "Imputation", options=[None, "mean", "median", "knn", "randomforest"]
    )

    drop_unmeasured_features = st.checkbox(
        "Drop unmeasured features (protein groups), i.e. ones that are all NaNs or Infs.",
        value=True,
    )

    batch = st.selectbox(
        "Batch",
        options=[False] + dataset.metadata.columns.to_list(),
    )

    return {
        PREPROCESSING_STEPS.REMOVE_CONTAMINATIONS: remove_contaminations,
        PREPROCESSING_STEPS.REMOVE_SAMPLES: remove_samples,
        PREPROCESSING_STEPS.SUBSET: subset,
        PREPROCESSING_STEPS.REPLACE_ZEROES: replace_zeroes,
        PREPROCESSING_STEPS.DATA_COMPLETENESS: data_completeness,
        PREPROCESSING_STEPS.LOG2_TRANSFORM: log2_transform,
        PREPROCESSING_STEPS.NORMALIZATION: normalization,
        PREPROCESSING_STEPS.IMPUTATION: imputation,
        PREPROCESSING_STEPS.DROP_UNMEASURED_FEATURES: drop_unmeasured_features,
        PREPROCESSING_STEPS.BATCH: batch,
    }


def update_workflow(
    settings: dict,
):
    """Update the workflow based on the settings.

    All values are evaluated for their truthiness. If a value is not truthy, the corresponding element is removed from the workflow.

    Args:
        settings (dict): Dictionary containing the preprocessing settings. As generated by configure_preprocessing.

    Returns:
        list[str]: List of keys of the workflow elements that are to be run during preprocessing.
    """
    new_workflow = [key for key, setting in settings.items() if setting]
    return new_workflow


# TODO: cache this
# TODO: Add a progress bar
# TODO: Add a button to cancel the preprocessing
def run_preprocessing(settings, dataset):
    """Run the preprocessing based on the settings.

    Args:
        settings (dict): Dictionary containing the preprocessing settings. As generated by configure_preprocessing.
        dataset (alphastat.Dataset): The dataset to be preprocessed, which will be modified in place.

    Returns:
        None
    """
    dataset.preprocess(**settings)
    st.toast("Preprocessing finished successfully!", icon="✅")

    if settings[PREPROCESSING_STEPS.BATCH]:
        dataset.batch_correction(batch=settings[PREPROCESSING_STEPS.BATCH])
        st.toast("Batch correction finished successfully!", icon="✅")


def display_preprocessing_info(preprocessing_info):
    """Display the preprocessing information as a DataFrame.

    Args:
        preprocessing_info (dict): Dictionary containing the preprocessing information. As set by alphastat.Dataset.preprocess.

    Returns:
        None
    """
    st.dataframe(
        pd.DataFrame.from_dict(preprocessing_info, orient="index").astype(str),
        use_container_width=True,
    )


def reset_preprocessing(dataset: DataSet) -> None:
    """Reset the preprocessing of the dataset.

    Args:
        dataset (Dataset): The dataset to be reset. The dataset will be reset in place.

    Returns:
        None
    """

    dataset.reset_preprocessing()
    st.toast("Preprocessing has been reset.", icon="✅")
