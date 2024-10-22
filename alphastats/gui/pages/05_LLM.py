import os

import pandas as pd
import streamlit as st
from openai import AuthenticationError

from alphastats.gui.utils.analysis_helper import (
    display_figure,
)
from alphastats.gui.utils.gpt_helper import (
    display_proteins,
    get_subgroups_for_each_group,
)
from alphastats.gui.utils.ollama_utils import LLMIntegration, Models
from alphastats.gui.utils.openai_utils import set_api_key
from alphastats.gui.utils.ui_helper import StateKeys, init_session_state, sidebar_info

init_session_state()
sidebar_info()


if StateKeys.DATASET not in st.session_state:
    st.info("Import Data first")
    st.stop()


st.markdown("### LLM Analysis")


@st.fragment
def llm_config():
    """Show the configuration options for the LLM analysis."""
    c1, _ = st.columns((1, 2))
    with c1:
        st.session_state[StateKeys.API_TYPE] = st.selectbox(
            "Select LLM",
            [Models.GPT, Models.OLLAMA],
        )

        if st.session_state[StateKeys.API_TYPE] == Models.GPT:
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            set_api_key(api_key)
        else:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            st.info(f"Expecting Ollama API at {base_url}.")


llm_config()

st.markdown("#### Analysis")


if StateKeys.LLM_INPUT not in st.session_state:
    st.info("Create a Volcano plot first using the 'Analysis' page.")
    st.stop()

volcano_plot, parameter_dict = st.session_state[StateKeys.LLM_INPUT]

c1, c2 = st.columns((1, 2))

with c1:
    # TODO move this to volcano anyway ?
    genes_of_interest_colored_df = volcano_plot.get_colored_labels_df()

    gene_names_colname = st.session_state[StateKeys.LOADER].gene_names
    prot_ids_colname = st.session_state[StateKeys.LOADER].index_column

    # st.session_state[StateKeys.PROT_ID_TO_GENE] = dict(
    #     zip(
    #         genes_of_interest_colored_df[prot_ids_colname].tolist(),
    #         genes_of_interest_colored_df[gene_names_colname].tolist(),
    #     )
    # ) # TODO unused?

    gene_to_prot_id_map = dict(
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

    # st.session_state["gene_functions"] = get_info(genes_of_interest_colored, organism)
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

    st.subheader("Genes of interest")
    c1, c2 = st.columns((1, 2), gap="medium")
    with c1:
        st.write("Upregulated genes")
        display_proteins(upregulated_genes, [])
    with c2:
        st.write("Downregulated genes")
        display_proteins([], downregulated_genes)


st.subheader("Prompts generated based on gene functions")

subgroups = get_subgroups_for_each_group(st.session_state[StateKeys.DATASET].metadata)
system_message = (
    f"You are an expert biologist and have extensive experience in molecular biology, medicine and biochemistry.{os.linesep}"
    "A user will present you with data regarding proteins upregulated in certain cells "
    "sourced from UniProt and abstracts from scientific publications. They seek your "
    "expertise in understanding the connections between these proteins and their potential role "
    f"in disease genesis. {os.linesep}Provide a detailed and insightful, yet concise response based on the given information. Use formatting to make your response more human readable."
    f"The data you have has following groups and respective subgroups: {str(subgroups)}."
    "Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that. If the data coming to"
    " you from a function has references to the literature (for example, PubMed), always quote the references in your response."
)
user_prompt = (
    f"We've recently identified several proteins that appear to be differently regulated in cells "
    f"when comparing {parameter_dict['group1']} and {parameter_dict['group2']} in the {parameter_dict['column']} group. "
    f"From our proteomics experiments, we know that the following ones are upregulated: {', '.join(upregulated_genes)}.{os.linesep}{os.linesep}"
    f"Here is the list of proteins that are downregulated: {', '.join(downregulated_genes)}.{os.linesep}{os.linesep}"
    f"Help us understand the potential connections between these proteins and how they might be contributing "
    f"to the differences. After that provide a high level summary"
)

with st.expander("System message", expanded=False):
    system_message = st.text_area("", value=system_message, height=150)

with st.expander("User prompt", expanded=True):
    user_prompt = st.text_area("", value=user_prompt, height=200)

llm_submitted = st.button("Run LLM analysis")


# creating new assistant only once TODO: add a button to create new assistant
if StateKeys.LLM_INTEGRATION not in st.session_state:
    if not llm_submitted:
        st.stop()

    try:
        llm = LLMIntegration(
            api_type=st.session_state[StateKeys.API_TYPE],
            api_key=st.session_state[StateKeys.OPENAI_API_KEY],
            base_url=os.getenv("OLLAMA_BASE_URL", None),
            dataset=st.session_state[StateKeys.DATASET],
            gene_to_prot_id_map=gene_to_prot_id_map,
        )

        # Set instructions and update tools

        llm.messages = [{"role": "system", "content": system_message}]

        st.session_state[StateKeys.LLM_INTEGRATION] = llm
        st.success(
            f"{st.session_state[StateKeys.API_TYPE].upper()} integration initialized successfully!"
        )

        llm.chat_completion(user_prompt)

    except AuthenticationError:
        st.warning(
            "Incorrect API key provided. Please enter a valid API key, it should look like this: sk-XXXXX"
        )
        st.stop()


@st.fragment
def llm_chat():
    """The chat interface for the LLM analysis."""
    llm = st.session_state[StateKeys.LLM_INTEGRATION]

    for message in llm.get_print_view(show_all=False):
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
        llm.chat_completion(prompt)
        st.rerun(scope="fragment")


llm_chat()
