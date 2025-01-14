import os

import pandas as pd
import streamlit as st
from openai import AuthenticationError

from alphastats.dataset.keys import Cols
from alphastats.dataset.plotting import plotly_object
from alphastats.gui.utils.analysis_helper import (
    display_figure,
    gather_uniprot_data,
)
from alphastats.gui.utils.llm_helper import (
    create_protein_editor,
    display_uniprot,
    llm_connection_test,
    set_api_key,
)
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    init_session_state,
    sidebar_info,
)
from alphastats.llm.llm_integration import LLMIntegration, Models
from alphastats.llm.prompts import get_initial_prompt, get_system_message
from alphastats.plots.plot_utils import PlotlyObject

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


st.set_page_config(layout="wide")
init_session_state()
sidebar_info()


st.markdown("## LLM")

if StateKeys.DATASET not in st.session_state:
    st.info("Import data first.")
    st.stop()


@st.fragment
def llm_config():
    """Show the configuration options for the LLM analysis."""
    c1, _ = st.columns((1, 2))
    with c1:
        current_model = st.session_state.get(StateKeys.MODEL_NAME, None)

        models = [Models.GPT4O, Models.OLLAMA_31_70B, Models.OLLAMA_31_8B]
        model_name = st.selectbox(
            "Select LLM",
            models,
            index=models.index(st.session_state.get(StateKeys.MODEL_NAME))
            if current_model is not None
            else 0,
        )
        st.session_state[StateKeys.MODEL_NAME] = model_name

        base_url = None
        if st.session_state[StateKeys.MODEL_NAME] in [Models.GPT4O]:
            api_key = st.text_input(
                "Enter OpenAI API Key and press Enter", type="password"
            )
            set_api_key(api_key)
        elif st.session_state[StateKeys.MODEL_NAME] in [
            Models.OLLAMA_31_70B,
            Models.OLLAMA_31_8B,
        ]:
            base_url = OLLAMA_BASE_URL
            st.info(f"Expecting Ollama API at {base_url}.")

        test_connection = st.button("Test connection")
        if test_connection:
            with st.spinner(f"Testing connection to {model_name}.."):
                error = llm_connection_test(
                    model_name=st.session_state[StateKeys.MODEL_NAME],
                    api_key=st.session_state[StateKeys.OPENAI_API_KEY],
                    base_url=base_url,
                )
                if error is None:
                    st.success(f"Connection to {model_name} successful!")
                else:
                    st.error(f"Connection to {model_name} failed: {str(error)}")

        if current_model != st.session_state[StateKeys.MODEL_NAME]:
            st.rerun(scope="app")


st.markdown("#### Configure LLM")
llm_config()


st.markdown("#### Analysis Input")

if StateKeys.LLM_INPUT not in st.session_state:
    st.info("Create a Volcano plot first using the 'Analysis' page.")
    st.stop()

volcano_plot, plot_parameters = st.session_state[StateKeys.LLM_INPUT]

st.markdown(f"Parameters used for analysis: `{plot_parameters}`")
c1, c2 = st.columns((2, 1))

with c2:
    display_figure(volcano_plot.plot)

with c1:
    regulated_genes_df = volcano_plot.res[volcano_plot.res["label"] != ""]
    regulated_genes_dict = dict(
        zip(regulated_genes_df[Cols.INDEX], regulated_genes_df["color"].tolist())
    )

    if not regulated_genes_dict:
        st.text("No genes of interest found.")
        st.stop()

    # Separate upregulated and downregulated genes
    upregulated_genes = [
        key for key in regulated_genes_dict if regulated_genes_dict[key] == "up"
    ]
    downregulated_genes = [
        key for key in regulated_genes_dict if regulated_genes_dict[key] == "down"
    ]

    # Create dataframes with checkboxes for selection
    upregulated_genes_df = pd.DataFrame(
        {"Protein": upregulated_genes, "Selected": [True] * len(upregulated_genes)}
    )

    downregulated_genes_df = pd.DataFrame(
        {"Protein": downregulated_genes, "Selected": [True] * len(downregulated_genes)}
    )

    st.markdown("##### Genes of interest")
    c11, c12 = st.columns((1, 1), gap="medium")

    with c11:
        selected_upregulated_genes = create_protein_editor(
            upregulated_genes_df,
            "Upregulated Proteins",
            st.session_state[StateKeys.DATASET]._feature_to_repr_map,
        )

    with c12:
        selected_downregulated_genes = create_protein_editor(
            downregulated_genes_df,
            "Downregulated Proteins",
            st.session_state[StateKeys.DATASET]._feature_to_repr_map,
        )

    # Combine the selected genes into a new regulated_genes_dict
    selected_regulated_genes = selected_upregulated_genes + selected_downregulated_genes
    regulated_genes_dict = {
        gene: "up" if gene in selected_upregulated_genes else "down"
        for gene in selected_regulated_genes
    }

    # If no genes are selected, stop the script
    if not regulated_genes_dict:
        st.text("No genes selected for analysis.")
        st.stop()

    if c1.button("Gather UniProt data for selected proteins"):
        gather_uniprot_data(selected_regulated_genes)

if any(
    feature not in st.session_state[StateKeys.ANNOTATION_STORE]
    for feature in selected_regulated_genes
):
    st.info(
        "No UniProt data stored for some proteins. Please run UniProt data fetching first to ensure correct annotation from Protein IDs instead of gene names."
    )


model_name = st.session_state[StateKeys.MODEL_NAME]
llm_integration_set_for_model = (
    st.session_state.get(StateKeys.LLM_INTEGRATION, {}).get(model_name, None)
    is not None
)

st.markdown("##### Select which information from Uniprot to supply to the LLM")
display_uniprot(
    regulated_genes_dict,
    st.session_state[StateKeys.DATASET]._feature_to_repr_map,
    disabled=llm_integration_set_for_model,
)

st.markdown("##### Prompts generated based on analysis input")
with st.expander("System message", expanded=False):
    system_message = st.text_area(
        "",
        value=get_system_message(st.session_state[StateKeys.DATASET]),
        height=150,
        disabled=llm_integration_set_for_model,
    )

# TODO: Regenerate initial prompt on reset
with st.expander("Initial prompt", expanded=True):
    feature_to_repr_map = st.session_state[StateKeys.DATASET]._feature_to_repr_map
    initial_prompt = st.text_area(
        "",
        value=get_initial_prompt(
            plot_parameters,
            list(map(feature_to_repr_map.get, selected_upregulated_genes)),
            list(map(feature_to_repr_map.get, selected_downregulated_genes)),
        ),
        height=200,
        disabled=llm_integration_set_for_model,
    )


st.markdown(f"##### LLM Analysis with {model_name}")

c1, c2, _ = st.columns((0.2, 0.2, 0.6))
llm_submitted = c1.button(
    "Run LLM analysis ...", disabled=llm_integration_set_for_model
)

llm_reset = c2.button(
    "❌ Reset LLM analysis ...", disabled=not llm_integration_set_for_model
)
if llm_reset:
    del st.session_state[StateKeys.LLM_INTEGRATION]
    st.rerun()


if st.session_state[StateKeys.LLM_INTEGRATION].get(model_name) is None:
    if not llm_submitted:
        st.stop()

    try:
        llm_integration = LLMIntegration(
            model_name=model_name,
            system_message=system_message,
            api_key=st.session_state[StateKeys.OPENAI_API_KEY],
            base_url=OLLAMA_BASE_URL,
            dataset=st.session_state[StateKeys.DATASET],
            genes_of_interest=list(regulated_genes_dict.keys()),
        )

        st.session_state[StateKeys.LLM_INTEGRATION][model_name] = llm_integration

        st.toast(
            f"{st.session_state[StateKeys.MODEL_NAME]} integration initialized successfully!",
            icon="✅",
        )

        with st.spinner("Processing initial prompt..."):
            llm_integration.chat_completion(initial_prompt)

        st.rerun(scope="app")
    except AuthenticationError:
        st.warning(
            "Incorrect API key provided. Please enter a valid API key, it should look like this: sk-XXXXX"
        )
        st.stop()


@st.fragment
def llm_chat(llm_integration: LLMIntegration, show_all: bool = False):
    """The chat interface for the LLM analysis."""

    # TODO dump to file -> static file name, plus button to do so
    # Ideas: save chat as txt, without encoding objects, just put a replacement string.
    # Offer bulk download of zip with all figures (via plotly download as svg.).
    # Alternatively write it all in one pdf report using e.g. pdfrw and reportlab (I have code for that combo).

    # no. tokens spent
    for message in llm_integration.get_print_view(show_all=show_all):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for artifact in message["artifacts"]:
                if isinstance(artifact, pd.DataFrame):
                    st.dataframe(artifact)
                elif isinstance(
                    artifact, (PlotlyObject, plotly_object)
                ):  # TODO can there be non-plotly types here
                    st.plotly_chart(artifact)
                elif not isinstance(artifact, str):
                    st.warning("Don't know how to display artifact:")
                    st.write(artifact)

    if prompt := st.chat_input("Say something"):
        with st.spinner("Processing prompt..."):
            llm_integration.chat_completion(prompt)
        st.rerun(scope="fragment")

    st.download_button(
        "Download chat log",
        llm_integration.get_chat_log_txt(),
        f"chat_log_{model_name}.txt",
        "text/plain",
    )


show_all = st.checkbox(
    "Show system messages",
    key="show_system_messages",
    help="Show all messages in the chat interface.",
)

llm_chat(st.session_state[StateKeys.LLM_INTEGRATION][model_name], show_all)
