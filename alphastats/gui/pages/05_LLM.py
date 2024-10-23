import os

import pandas as pd
import streamlit as st
from openai import AuthenticationError

from alphastats.gui.utils.analysis_helper import (
    display_figure,
)
from alphastats.gui.utils.llm_helper import (
    get_display_proteins_html,
    llm_connection_test,
    set_api_key,
)
from alphastats.gui.utils.ui_helper import StateKeys, init_session_state, sidebar_info
from alphastats.llm.llm_integration import LLMIntegration, Models
from alphastats.llm.prompts import get_initial_prompt, get_system_message

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


init_session_state()
sidebar_info()


if StateKeys.DATASET not in st.session_state:
    st.info("Import Data first")
    st.stop()


st.markdown("### LLM")


@st.fragment
def llm_config():
    """Show the configuration options for the LLM analysis."""
    c1, _ = st.columns((1, 2))
    with c1:
        model_before = st.session_state.get(StateKeys.API_TYPE, None)

        st.session_state[StateKeys.API_TYPE] = st.selectbox(
            "Select LLM",
            [Models.GPT4O, Models.OLLAMA_31_70B, Models.OLLAMA_31_8B],
        )

        base_url = None
        if st.session_state[StateKeys.API_TYPE] in [Models.GPT4O]:
            api_key = st.text_input(
                "Enter OpenAI API Key and press Enter", type="password"
            )
            set_api_key(api_key)
        elif st.session_state[StateKeys.API_TYPE] in [
            Models.OLLAMA_31_70B,
            Models.OLLAMA_31_8B,
        ]:
            base_url = OLLAMA_BASE_URL
            st.info(f"Expecting Ollama API at {base_url}.")

        test_connection = st.button("Test connection")
        if test_connection:
            with st.spinner(f"Testing connection to {api_type}.."):
                error = llm_connection_test(
                    api_type=st.session_state[StateKeys.API_TYPE],
                    api_key=st.session_state[StateKeys.OPENAI_API_KEY],
                    base_url=base_url,
                )
                if error is None:
                    st.success(f"Connection to {api_type} successful!")
                else:
                    st.error(f"❌ Connection to {api_type} failed: {str(error)}")

        if model_before != st.session_state[StateKeys.API_TYPE]:
            st.rerun(scope="app")


st.markdown("#### Configure LLM")
llm_config()


st.markdown("#### Analysis Input")

if StateKeys.LLM_INPUT not in st.session_state:
    st.info("Create a Volcano plot first using the 'Analysis' page.")
    st.stop()

volcano_plot, parameter_dict = st.session_state[StateKeys.LLM_INPUT]

st.write(f"Parameters used for analysis: {parameter_dict}")
c1, c2 = st.columns((1, 2))

with c1:
    # TODO move this to volcano anyway ?
    genes_of_interest_colored_df = volcano_plot.get_colored_labels_df()

    gene_names_colname = st.session_state[StateKeys.LOADER].gene_names
    prot_ids_colname = st.session_state[StateKeys.LOADER].index_column

    gene_to_prot_id_map = dict(  # TODO move this logic to dataset
        zip(
            genes_of_interest_colored_df[gene_names_colname].tolist(),
            genes_of_interest_colored_df[prot_ids_colname].tolist(),
        )
    )

    with c2:
        display_figure(volcano_plot.plot)

    genes_of_interest_colored = volcano_plot.get_colored_labels()
    if not genes_of_interest_colored:
        st.text("No proteins of interest found.")
        st.stop()

    upregulated_genes = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "up"
    ]
    downregulated_genes = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "down"
    ]

    st.markdown("##### Genes of interest")
    c1, c2 = st.columns((1, 2), gap="medium")
    with c1:
        st.write("Upregulated genes")
        st.markdown(
            get_display_proteins_html(upregulated_genes, True), unsafe_allow_html=True
        )

    with c2:
        st.write("Downregulated genes")
        st.markdown(
            get_display_proteins_html(downregulated_genes, False),
            unsafe_allow_html=True,
        )


st.markdown("##### Prompts generated based on analysis input")

api_type = st.session_state[StateKeys.API_TYPE]
llm_integration_set_for_model = (
    st.session_state.get(StateKeys.LLM_INTEGRATION, {}).get(api_type, None) is not None
)
with st.expander("System message", expanded=False):
    system_message = st.text_area(
        "",
        value=get_system_message(st.session_state[StateKeys.DATASET]),
        height=150,
        disabled=llm_integration_set_for_model,
    )

with st.expander("Initial prompt", expanded=True):
    initial_prompt = st.text_area(
        "",
        value=get_initial_prompt(
            parameter_dict, upregulated_genes, downregulated_genes
        ),
        height=200,
        disabled=llm_integration_set_for_model,
    )


st.markdown(f"##### LLM Analysis with {api_type}")

llm_submitted = st.button(
    "Run LLM analysis ...", disabled=llm_integration_set_for_model
)

if st.session_state[StateKeys.LLM_INTEGRATION].get(api_type) is None:
    if not llm_submitted:
        st.stop()

    try:
        llm_integration = LLMIntegration(
            api_type=api_type,
            system_message=system_message,
            api_key=st.session_state[StateKeys.OPENAI_API_KEY],
            base_url=OLLAMA_BASE_URL,
            dataset=st.session_state[StateKeys.DATASET],
            gene_to_prot_id_map=gene_to_prot_id_map,
        )

        st.session_state[StateKeys.LLM_INTEGRATION][api_type] = llm_integration

        st.success(
            f"{st.session_state[StateKeys.API_TYPE]} integration initialized successfully!"
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
    # how to deal with binaries? base64 encode?
    # "import chat" functionality?

    # no. tokens spent
    for message in llm_integration.get_print_view(show_all=show_all):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for artifact in message["artifacts"]:
                if isinstance(artifact, pd.DataFrame):
                    st.dataframe(artifact)
                elif "plotly" in str(type(artifact)):
                    st.plotly_chart(artifact)
                elif not isinstance(artifact, str):
                    st.warning("Don't know how to display artifact:")
                    st.write(artifact)

    if prompt := st.chat_input("Say something"):
        with st.spinner("Processing prompt..."):
            llm_integration.chat_completion(prompt)
        st.rerun(scope="fragment")


show_all = st.checkbox(
    "Show system messages",
    key="show_system_messages",
    help="Show all messages in the chat interface.",
)

llm_chat(st.session_state[StateKeys.LLM_INTEGRATION][api_type], show_all)
