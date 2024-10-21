import os

import pandas as pd
import streamlit as st
from openai import AuthenticationError

from alphastats.gui.utils.analysis_helper import (
    display_figure,
)
from alphastats.gui.utils.gpt_helper import (
    display_proteins,
    get_assistant_functions,
    get_general_assistant_functions,
    get_subgroups_for_each_group,
)
from alphastats.gui.utils.ollama_utils import LLMIntegration
from alphastats.gui.utils.openai_utils import set_api_key
from alphastats.gui.utils.ui_helper import StateKeys, init_session_state, sidebar_info

init_session_state()
sidebar_info()


if StateKeys.DATASET not in st.session_state:
    st.info("Import Data first")
    st.stop()

if "LLM" not in st.session_state:
    st.info("Create a Volcano plot first using the 'Analysis' page.")
    st.stop()

volcano_plot, chosen_parameter_dict = st.session_state["LLM"]


st.markdown("### LLM Analysis")


@st.fragment
def llm_config():
    """Show the configuration options for the LLM analysis."""
    c1, _ = st.columns((1, 2))
    with c1:
        st.session_state[StateKeys.API_TYPE] = st.selectbox(
            "Select LLM",
            ["gpt4o", "llama3.1 70b"],
            # index=0 if st.session_state[StateKeys.API_TYPE] == "gpt4o" else 1,
        )

        if st.session_state[StateKeys.API_TYPE] == "gpt4o":
            api_key = st.text_input("Enter OpenAI API Key", type="password")
            set_api_key(api_key)


llm_config()

st.markdown("#### Analysis")

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
    st.session_state[StateKeys.GENE_TO_PROT_ID] = dict(
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
    st.session_state[StateKeys.UPREGULATED] = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "up"
    ]
    st.session_state[StateKeys.DOWNREGULATED] = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "down"
    ]

    st.subheader("Genes of interest")
    c1, c2 = st.columns((1, 2), gap="medium")
    with c1:
        st.write("Upregulated genes")
        display_proteins(st.session_state[StateKeys.UPREGULATED], [])
    with c2:
        st.write("Downregulated genes")
        display_proteins([], st.session_state[StateKeys.DOWNREGULATED])


st.session_state[StateKeys.INSTRUCTIONS] = (
    f"You are an expert biologist and have extensive experience in molecular biology, medicine and biochemistry.{os.linesep}"
    "A user will present you with data regarding proteins upregulated in certain cells "
    "sourced from UniProt and abstracts from scientific publications. They seek your "
    "expertise in understanding the connections between these proteins and their potential role "
    f"in disease genesis. {os.linesep}Provide a detailed and insightful, yet concise response based on the given information. Use formatting to make your response more human readable."
    f"The data you have has following groups and respective subgroups: {str(get_subgroups_for_each_group(st.session_state[StateKeys.DATASET].metadata))}."
    "Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that. If the data coming to"
    " you from a function has references to the literature (for example, PubMed), always quote the references in your response."
)
if "column" in chosen_parameter_dict and StateKeys.UPREGULATED in st.session_state:
    st.session_state[StateKeys.USER_PROMPT] = (
        f"We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {chosen_parameter_dict['group1']} and {chosen_parameter_dict['group2']} in the {chosen_parameter_dict['column']} group. "
        f"From our proteomics experiments, we know that the following ones are upregulated: {', '.join(st.session_state[StateKeys.UPREGULATED])}.{os.linesep}{os.linesep}"
        f"Here is the list of proteins that are downregulated: {', '.join(st.session_state[StateKeys.DOWNREGULATED])}.{os.linesep}{os.linesep}"
        f"Help us understand the potential connections between these proteins and how they might be contributing "
        f"to the differences. After that provide a high level summary"
    )

if StateKeys.USER_PROMPT in st.session_state:
    st.subheader("Automatically generated prompt based on gene functions:")
    with st.expander("System prompt", expanded=True):
        st.session_state[StateKeys.INSTRUCTIONS] = st.text_area(
            "", value=st.session_state[StateKeys.INSTRUCTIONS], height=150
        )

    with st.expander("User prompt", expanded=True):
        st.session_state[StateKeys.USER_PROMPT] = st.text_area(
            "", value=st.session_state[StateKeys.USER_PROMPT], height=200
        )

llm_submitted = st.button("Run LLM analysis")


# creating new assistant only once TODO: add a button to create new assistant
if StateKeys.LLM_INTEGRATION not in st.session_state:
    if not llm_submitted:
        st.stop()

    try:
        if st.session_state[StateKeys.API_TYPE] == "gpt4o":
            llm = LLMIntegration(
                api_type="gpt",
                api_key=st.session_state[StateKeys.OPENAI_API_KEY],
                dataset=st.session_state[StateKeys.DATASET],
                metadata=st.session_state[StateKeys.DATASET].metadata,
            )
        else:
            llm = LLMIntegration(
                api_type="ollama",
                base_url=os.getenv("OLLAMA_BASE_URL", None),
                dataset=st.session_state[StateKeys.DATASET],
                metadata=st.session_state[StateKeys.DATASET].metadata,
            )

        # Set instructions and update tools
        llm.tools = [
            *get_general_assistant_functions(),
            *get_assistant_functions(
                gene_to_prot_id_dict=st.session_state[StateKeys.GENE_TO_PROT_ID],
                metadata=st.session_state[StateKeys.DATASET].metadata,
                subgroups_for_each_group=get_subgroups_for_each_group(
                    st.session_state[StateKeys.DATASET].metadata
                ),
            ),
        ]

        st.session_state[StateKeys.ARTIFACTS] = {}
        llm.messages = [
            {"role": "system", "content": st.session_state[StateKeys.INSTRUCTIONS]}
        ]

        st.session_state[StateKeys.LLM_INTEGRATION] = llm
        st.success(
            f"{st.session_state[StateKeys.API_TYPE].upper()} integration initialized successfully!"
        )

        response = llm.chat_completion(st.session_state[StateKeys.USER_PROMPT])

    except AuthenticationError:
        st.warning(
            "Incorrect API key provided. Please enter a valid API key, it should look like this: sk-XXXXX"
        )
        st.stop()


@st.fragment
def llm_chat():
    """The chat interface for the LLM analysis."""
    llm = st.session_state[StateKeys.LLM_INTEGRATION]

    for num, role_content_dict in enumerate(st.session_state[StateKeys.MESSAGES]):
        if role_content_dict["role"] == "tool" or role_content_dict["role"] == "system":
            continue
        if "tool_calls" in role_content_dict:
            continue
        with st.chat_message(role_content_dict["role"]):
            st.markdown(role_content_dict["content"])
            if num in st.session_state[StateKeys.ARTIFACTS]:
                for artefact in st.session_state[StateKeys.ARTIFACTS][num]:
                    if isinstance(artefact, pd.DataFrame):
                        st.dataframe(artefact)
                    elif "plotly" in str(type(artefact)):
                        st.plotly_chart(artefact)

    if prompt := st.chat_input("Say something"):
        llm.chat_completion(prompt)
        st.rerun(scope="fragment")


llm_chat()
