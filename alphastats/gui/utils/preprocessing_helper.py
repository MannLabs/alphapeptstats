import streamlit as st
import pandas as pd
from st_cytoscape import cytoscape

import datetime

def draw_predefined_workflow(workflow = ["remove contaminations", "subset data", "log2 transform"]):
    available_steps = ["remove contaminations", "remove samples", "subset data", "filter data completeness", "log2 transform", "normalization", "imputation"]

    elements = [
        {
            'group': 'nodes',
            'data': {
                'id': i,
                'label': label,
            },
            "selectable": True,
            "classes": ['active'] if label in st.session_state.workflow else ['inactive']
        } for i, label in enumerate(available_steps)
    ]

    for label1, label2 in zip(st.session_state.workflow[:-1], st.session_state.workflow[1:]):
        i = available_steps.index(label1)
        j = available_steps.index(label2)
        elements.append({'group':'edges', 'data': {'id': f'{i}_{j}', 'source': i, 'target': j}, 'selectable': False})

    stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            'shape':'roundrectangle',
            "width": 200,
            "height": 60,
            "text-valign": "center",
            "text-halign": "center"
        },
    },
    {
        "selector": "node.active",
        "style": {
            "background-color": 'lightgreen',
        },
    },
    {
        "selector": "node.inactive",
        "style": {
            "background-color": 'lightgrey',
        },
    },
    {
        "selector": "node.selected",
        "style": {
            "background-color": 'red'
        },
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

    selected = cytoscape(
        elements,
        stylesheet,
        layout={'name': 'grid', 'columns': 1},
        selection_type='single',
        user_panning_enabled=False,
        user_zooming_enabled=False,
        height=f'{len(available_steps)*80}px',
        key="predefined_workflow")

    return selected


def preprocessing():
    st.markdown(
        "Before analyzing your data, consider normalizing and imputing your data as well as the removal of contaminants. "
        + "A more detailed description about the preprocessing methods can be found in the AlphaPeptStats "
        + "[documentation](https://alphapeptstats.readthedocs.io/en/main/data_preprocessing.html)."
    )

    with st.form("preprocessing"):
        dataset = st.session_state["dataset"]

        remove_contaminations = st.selectbox(
            f"Remove contaminations annotated in {dataset.filter_columns}",
            options=[True, False],
        )

        subset = st.selectbox(
            "Subset data so it matches with metadata. Remove miscellanous samples in rawinput.",
            options=[True, False],
        )

        remove_samples = st.multiselect(
            "Remove samples from analysis",
            options=st.session_state.dataset.metadata[
                st.session_state.dataset.sample
            ].to_list(),
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

        submitted = st.form_submit_button("Submit")

    if submitted:
        if len(remove_samples) == 0:
            remove_samples = None

        st.session_state.workflow = [el for el, form in zip(["remove contaminations", "remove samples", "subset data", "filter data completeness", "log2 transform", "normalization", "imputation"], [remove_contaminations, remove_samples, subset, data_completeness, log2_transform, normalization, imputation]) if form not in [None, False, [], 0.0]]

        st.session_state.dataset.preprocess(
            remove_contaminations=remove_contaminations,
            log2_transform=log2_transform,
            remove_samples=remove_samples,
            data_completeness=data_completeness,
            subset=subset,
            normalization=normalization,
            imputation=imputation,
        )
        preprocessing = st.session_state.dataset.preprocessing_info
        st.info(
            "Data has been processed. "
            + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
        st.dataframe(
            pd.DataFrame.from_dict(preprocessing, orient="index").astype(str),
            use_container_width=True,
        )

    st.markdown("#### Batch correction: correct for technical bias")

    with st.form("Batch correction: correct for technical bias"):
        batch = st.selectbox(
            "Batch", options=st.session_state.dataset.metadata.columns.to_list()
        )
        submit_batch_correction = st.form_submit_button("Submit")

    if submit_batch_correction:
        st.session_state.dataset.batch_correction(batch=batch)
        st.info(
            "Data has been processed. "
            + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )

    reset_steps = st.button("Reset all Preprocessing steps")

    if reset_steps:
        reset_preprocessing()


def reset_preprocessing():
    st.session_state.dataset.create_matrix()
    preprocessing = st.session_state.dataset.preprocessing_info
    st.info(
        "Data has been reset. " + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )
    st.dataframe(
        pd.DataFrame.from_dict(preprocessing, orient="index").astype(str),
        use_container_width=True,
    )


def main_preprocessing():
    if "dataset" in st.session_state:
        preprocessing()

    else:
        st.info("Import Data first")


def plot_intensity_distribution():
    st.selectbox(
        "Sample", options=st.session_state.dataset.metadata["sample"].to_list()
    )
