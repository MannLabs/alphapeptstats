import streamlit as st
import pandas as pd


from alphastats.gui.utils.preprocessing_helper import (
    draw_predefined_workflow,
    configure_preprocessing,
    update_workflow,
    run_preprocessing,
    reset_preprocessing,
)
from alphastats.gui.utils.ui_helper import sidebar_info


sidebar_info()

if "workflow" not in st.session_state:
    st.session_state.workflow = [
        "remove contaminations",
        "subset data",
        "log2 transform",
    ]

st.markdown("### Preprocessing")
st.markdown(
    "Select either the predefined workflow where you can only enable/disable steps or create a custom workflow, that allows switching steps around."
)


tab1, tab2 = st.tabs(["Predefined workflow", "Custom workflow"])

with tab1:
    c1, c2 = st.columns([1, 1])

    with c2:
        (
            remove_contaminations,
            remove_samples,
            subset,
            data_completeness,
            log2_transform,
            normalization,
            imputation,
            batch,
        ) = configure_preprocessing()

        update_workflow(
            remove_contaminations,
            remove_samples,
            subset,
            data_completeness,
            log2_transform,
            normalization,
            imputation,
            batch,
        )

    with c1:
        st.write("### Flowchart of currenlty selected workflow:")

        selected_nodes = draw_predefined_workflow(st.session_state.workflow)

        if "dataset" in st.session_state:
            c11, c12 = st.columns([1, 1])

            with c11:
                if st.button("Run preprocessing"):
                    run_preprocessing(
                        remove_contaminations,
                        remove_samples,
                        subset,
                        data_completeness,
                        log2_transform,
                        normalization,
                        imputation,
                        batch,
                    )

            with c12:
                reset_preprocessing()

        else:
            st.info("Import Data first")

    # TODO: Add comparison plot of indensity distribution before and after preprocessing

with tab2:
    "Custom workflows coming soon"
