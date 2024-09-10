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
        "selector": "node.selected", # TODO: This currently does not work, figure out why
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
WORKFLOW_STEPS = {
    'remove_contaminations': {
        'repr': 'Remove contaminations',
        'help': 'Remove contaminations annotated in the contaminations library and filter columns included in the dataset.'
    },
    'remove_samples': {
        'repr': 'Remove samples',
        'help': 'Remove samples from analysis, e.g. useful when failed or blank runs are included.'
    },
    'subset': {
        'repr': 'Subset data',
        'help': 'Subset data so it matches with metadata. Can for example be useful if several dimensions of an experiment were analysed together.'
    },
    'data_completeness': {
        'repr': 'Filter data completeness',
        'help': 'Filter data based on completeness across samples. E.g. if a protein has to be detected in at least 70% of the samples.'
    },
    'log2_transform': {
        'repr': 'Log2 transform',
        'help': 'Log2-transform dataset.'
    },
    'normalization': {
        'repr': 'Normalization',
        'help': 'Normalize data using one of the available methods ("zscore", "quantile", "vst", "linear").'
    },
    'imputation': {
        'repr': 'Imputation',
        'help': 'Impute missing values using one of the available methods ("mean", "median", "knn", "randomforest").'
    },
    'batch': {
        'repr': 'Batch correction',
        'help': 'Batch correction.'
    },
}

PREDEFINED_ORDER = ["remove_contaminations", "remove_samples", "subset", "data_completeness", "log2_transform", "normalization", "imputation", "batch"]

def draw_workflow(workflow: list[str], order: list[str] = PREDEFINED_ORDER):

    elements = [
        {
            "group": "nodes",
            "data": {
                "id": i,
                "label": WORKFLOW_STEPS[key]["repr"],
                "key": key,
            },
            "selectable": True,
            "classes": ["active"]
            if key in workflow
            else ["inactive"],
        }
        for i, key in enumerate(order)
    ]

    for key1, key2 in zip(
        workflow[:-1], workflow[1:]
    ):
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
        'remove_contaminations': remove_contaminations,
        'remove_samples': remove_samples,
        'subset': subset,
        'data_completeness': data_completeness,
        'log2_transform': log2_transform,
        'normalization': normalization,
        'imputation': imputation,
        'batch': batch,
    }


def update_workflow(
    settings: dict,
):
    new_workflow = [
        key
        for key, setting in settings.items()
        if setting not in [None, False, [], 0.0]
    ]
    return new_workflow


def run_preprocessing(
    settings,
):
    settings['remove_samples'] = settings['remove_samples'] if len(settings['remove_samples']) != 0 else None
    st.session_state.dataset.preprocess(**settings)

    preprocessing = st.session_state.dataset.preprocessing_info
    st.info(
        "Data has been processed. "
        + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )
    st.dataframe(
        pd.DataFrame.from_dict(preprocessing, orient="index").astype(str),
        use_container_width=True,
    )
    if settings['batch']:
        st.session_state.dataset.batch_correction(batch=settings['batch'])


def reset_preprocessing():
    # TODO: check if the method names make sense
    st.session_state.dataset.create_matrix()
    preprocessing = st.session_state.dataset.preprocessing_info
    st.info(
        "Data has been reset. "
        + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )
    st.dataframe(
        pd.DataFrame.from_dict(preprocessing, orient="index").astype(str),
        use_container_width=True,
    )


def plot_intensity_distribution():
    st.selectbox(
        "Sample", options=st.session_state.dataset.metadata["sample"].to_list()
    )
