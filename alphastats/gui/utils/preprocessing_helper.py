import streamlit as st
import pandas as pd
from st_cytoscape import cytoscape

import datetime

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

# TODO: Make help texts meaningful
# TODO: Show help texts on the widgets
class PREPROCESSING_STEPS:
    REMOVE_CONTAMINATONS = {
        "key": "remove_contaminations",
        "repr": "Remove contaminations",
        "help": "Remove contaminations annotated in the contaminations library and filter columns included in the dataset.",
    }
    REMOVE_SAMPLES = {
        "key": "remove_samples",
        "repr": "Remove samples",
        "help": "Remove samples from analysis, e.g. useful when failed or blank runs are included.",
    }
    SUBSET = {
        "key": "subset",
        "repr": "Subset data",
        "help": "Subset data so it matches with metadata. Can for example be useful if several dimensions of an experiment were analysed together.",
    }
    DATA_COMPLETENESS = {
        "key": "data_completeness",
        "repr": "Filter data completeness",
        "help": "Filter data based on completeness across samples. E.g. if a protein has to be detected in at least 70% of the samples.",
    }
    LOG2_TRANSFORM = {
        "key": "log2_transform",
        "repr": "Log2 transform",
        "help": "Log2-transform dataset."
    }
    NORMALIZATION = {
        "key": "normalization",
        "repr": "Normalization",
        "help": 'Normalize data using one of the available methods ("zscore", "quantile", "vst", "linear").',
    }
    IMPUTATION = {
        "key": "imputation",
        "repr": "Imputation",
        "help": 'Impute missing values using one of the available methods ("mean", "median", "knn", "randomforest").',
    }
    BATCH = {
        "key": "batch",
        "repr": "Batch correction",
        "help": "Batch correction.",
    }

PREDEFINED_ORDER = [
    PREPROCESSING_STEPS.REMOVE_CONTAMINATONS["key"],
    PREPROCESSING_STEPS.REMOVE_SAMPLES["key"],
    PREPROCESSING_STEPS.SUBSET["key"],
    PREPROCESSING_STEPS.DATA_COMPLETENESS["key"],
    PREPROCESSING_STEPS.LOG2_TRANSFORM["key"],
    PREPROCESSING_STEPS.NORMALIZATION["key"],
    PREPROCESSING_STEPS.IMPUTATION["key"],
    PREPROCESSING_STEPS.BATCH["key"],
]


def draw_workflow(workflow: list[str], order: list[str] = PREDEFINED_ORDER):

    def find_step_by_key_value(key, target_value):
        for _, attribute_value in PREPROCESSING_STEPS.__dict__.items():
            if isinstance(attribute_value, dict) and key in attribute_value and attribute_value[key] == target_value:
                return attribute_value
        return None

    elements = [
        {
            "group": "nodes",
            "data": {
                "id": i,
                "label": find_step_by_key_value('key', key)["repr"],
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
    st.markdown(
        "Before analyzing your data, consider normalizing and imputing your data as well as the removal of contaminants. "
        + "A more detailed description about the preprocessing methods can be found in the AlphaPeptStats "
        + "[documentation](https://alphapeptstats.readthedocs.io/en/main/data_preprocessing.html)."
    )

    remove_contaminations = st.selectbox(
        f"Remove contaminations annotated in {dataset.filter_columns}",
        options=[True, False],
    )

    subset = st.selectbox(
        "Subset data so it matches with metadata. Remove miscellanous samples in rawinput.",
        options=[True, False],
    )

    # TODO: value of this widget does not persist across dataset reset (likely because the metadata is reset)
    remove_samples = st.multiselect(
        "Remove samples from analysis",
        options=dataset.metadata[dataset.sample].to_list(),
    )
    remove_samples = remove_samples if len(remove_samples) != 0 else None

    data_completeness = st.number_input(
        f"Data completeness across samples cut-off \n(0.7 -> protein has to be detected in at least 70% of the samples)",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.1,
    )

    log2_transform = st.selectbox(
        "Log2-transform dataset",
        options=[True, False],
    )

    normalization = st.selectbox(
        "Normalization", options=[None, "zscore", "quantile", "vst", "linear"]
    )

    imputation = st.selectbox(
        "Imputation", options=[None, "mean", "median", "knn", "randomforest"]
    )

    batch = st.selectbox(
        "Batch",
        options=[False] + dataset.metadata.columns.to_list(),
    )

    return {
        PREPROCESSING_STEPS.REMOVE_CONTAMINATONS["key"]: remove_contaminations,
        PREPROCESSING_STEPS.REMOVE_SAMPLES["key"]: remove_samples,
        PREPROCESSING_STEPS.SUBSET["key"]: subset,
        PREPROCESSING_STEPS.DATA_COMPLETENESS["key"]: data_completeness,
        PREPROCESSING_STEPS.LOG2_TRANSFORM["key"]: log2_transform,
        PREPROCESSING_STEPS.NORMALIZATION["key"]: normalization,
        PREPROCESSING_STEPS.IMPUTATION["key"]: imputation,
        PREPROCESSING_STEPS.BATCH["key"]: batch,
    }


def update_workflow(
    settings: dict,
):
    new_workflow = [
        key
        for key, setting in settings.items()
        if bool(setting)  # use of falsiness
    ]
    return new_workflow


# TODO: cache this
def run_preprocessing(
    settings,
    dataset
):
    dataset.preprocess(**settings)
    st.info(
        "Data has been processed. "
        + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )

    if settings[PREPROCESSING_STEPS.BATCH["key"]]:
        dataset.batch_correction(batch=settings[PREPROCESSING_STEPS.BATCH["key"]])
        st.info(
            "Data has been batch corrected. "
            + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )


def display_preprocessing_info(preprocessing_info):
    st.dataframe(
        pd.DataFrame.from_dict(preprocessing_info, orient="index").astype(str),
        use_container_width=True,
    )


# TODO: cache this
def reset_preprocessing(dataset):
    # TODO: check if the method names make sense
    dataset.create_matrix()
    st.info(
        "Data has been reset. " + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )
