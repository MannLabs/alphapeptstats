import os
import warnings
from typing import Dict

import pandas as pd
import streamlit as st
from openai import AuthenticationError

from alphastats.dataset.keys import Cols
from alphastats.dataset.plotting import plotly_object
from alphastats.gui.utils.analysis import ResultComponent
from alphastats.gui.utils.analysis_helper import (
    display_figure,
    gather_uniprot_data,
)
from alphastats.gui.utils.llm_helper import (
    LLM_ENABLED_ANALYSIS,
    OLLAMA_BASE_URL,
    display_uniprot,
    get_df_for_protein_selector,
    protein_selector,
)
from alphastats.gui.utils.state_keys import DefaultStates, LLMKeys, StateKeys
from alphastats.gui.utils.state_utils import (
    init_session_state,
)
from alphastats.gui.utils.ui_helper import (
    sidebar_info,
)
from alphastats.llm.llm_integration import LLMIntegration, MessageKeys, Roles
from alphastats.llm.prompts import get_initial_prompt, get_system_message
from alphastats.llm.uniprot_utils import (
    format_uniprot_annotation,
    get_uniprot_state_key,
)
from alphastats.plots.plot_utils import PlotlyObject

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()


st.markdown("## LLM")

if StateKeys.DATASET not in st.session_state:
    st.info("Import data first.")
    st.stop()


def _pretty_print_analysis(key: str) -> str:
    """Pretty print the analysis key."""
    analysis = st.session_state[StateKeys.SAVED_ANALYSES][key]
    return (
        f"[{key}] #{analysis['number']} {analysis['method']} {analysis['parameters']}"
    )


saved_analyses_keys = [
    k
    for k, analysis in st.session_state[StateKeys.SAVED_ANALYSES].items()
    if analysis["method"] in LLM_ENABLED_ANALYSIS
]

if not saved_analyses_keys:
    st.info(
        f"Create a supported analysis first on the 'Analysis' page. Currently supported: {LLM_ENABLED_ANALYSIS}"
    )
    st.stop()

selected_analysis_key = st.selectbox(
    "Select result to discuss",
    saved_analyses_keys,
    format_func=_pretty_print_analysis,
    index=None if len(saved_analyses_keys) > 1 else 0,
)
selected_analysis = st.session_state[StateKeys.SAVED_ANALYSES].get(
    selected_analysis_key, None
)

st.markdown("#### Analysis Input")

if selected_analysis is None:
    st.info("Select analysis first in the dropdown")
    st.stop()

volcano_plot: ResultComponent = selected_analysis["result"]
plot_parameters: Dict = selected_analysis["parameters"]

st.markdown(f"Parameters used for analysis: `{plot_parameters}`")

c1, c2, c3 = st.columns((1, 1, 1))

with c3:
    st.markdown("##### Volcano plot")
    display_figure(volcano_plot.plot)

regulated_genes_df = volcano_plot.annotated_dataframe[
    volcano_plot.annotated_dataframe["significant"] != "non_sig"
]
regulated_genes_dict = dict(
    zip(regulated_genes_df[Cols.INDEX], regulated_genes_df["significant"].tolist())
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

if st.session_state[StateKeys.LLM_CHATS].get(selected_analysis_key) is None:
    st.session_state[StateKeys.LLM_CHATS][selected_analysis_key] = {}
selected_llm_chat = st.session_state[StateKeys.LLM_CHATS][selected_analysis_key]

# TODO gather all selected_llm_chat-inits in a method
if LLMKeys.RECENT_CHAT_WARNINGS not in selected_llm_chat:
    selected_llm_chat[LLMKeys.RECENT_CHAT_WARNINGS] = []
if selected_llm_chat.get(LLMKeys.SELECTED_UNIPROT_FIELDS) is None:
    selected_llm_chat[LLMKeys.SELECTED_UNIPROT_FIELDS] = (
        DefaultStates.SELECTED_UNIPROT_FIELDS.copy()
    )

# Create dataframes with checkboxes for selection
if selected_llm_chat.get(LLMKeys.SELECTED_GENES_UP) is None:
    selected_llm_chat[LLMKeys.SELECTED_GENES_UP] = upregulated_genes
upregulated_genes_df = get_df_for_protein_selector(
    upregulated_genes, selected_llm_chat[LLMKeys.SELECTED_GENES_UP]
)

if selected_llm_chat.get(LLMKeys.SELECTED_GENES_DOWN) is None:
    selected_llm_chat[LLMKeys.SELECTED_GENES_DOWN] = downregulated_genes
downregulated_genes_df = get_df_for_protein_selector(
    downregulated_genes, selected_llm_chat[LLMKeys.SELECTED_GENES_DOWN]
)


with c1:
    st.markdown("##### Genes of interest")
    protein_selector(
        upregulated_genes_df,
        "Upregulated Proteins",
        selected_analysis_key,
        state_key=LLMKeys.SELECTED_GENES_UP,
    )

with c2:
    st.markdown("##### ")
    protein_selector(
        downregulated_genes_df,
        "Downregulated Proteins",
        selected_analysis_key,
        state_key=LLMKeys.SELECTED_GENES_DOWN,
    )

# Combine the selected genes into a new regulated_genes_dict
selected_genes = (
    selected_llm_chat[LLMKeys.SELECTED_GENES_UP]
    + selected_llm_chat[LLMKeys.SELECTED_GENES_DOWN]
)
regulated_genes_dict = {
    gene: "up" if gene in selected_llm_chat[LLMKeys.SELECTED_GENES_UP] else "down"
    for gene in selected_genes
}

# If no genes are selected, stop the script
if not regulated_genes_dict:
    st.text("No genes selected for analysis.")
    st.stop()

if st.button("Gather UniProt data for selected proteins"):
    gather_uniprot_data(selected_genes)

if any(
    feature not in st.session_state[StateKeys.ANNOTATION_STORE]
    for feature in selected_genes
):
    st.info(
        "No or incomplete UniProt data stored for the selected proteins. Please run UniProt data fetching first to ensure correct annotation from Protein IDs instead of gene names."
    )


model_name = st.session_state[StateKeys.MODEL_NAME]
llm_integration_set_for_model = selected_llm_chat.get(model_name, None) is not None

st.markdown("##### Select which information from Uniprot to supply to the LLM")

display_uniprot(
    regulated_genes_dict,
    st.session_state[StateKeys.DATASET]._feature_to_repr_map,
    model_name=model_name,
    selected_analysis_key=selected_analysis_key,
    disabled=llm_integration_set_for_model,
)

st.markdown("##### Prompts generated based on analysis input")
with st.expander("System message", expanded=False):
    system_message = st.text_area(
        " ",
        value=get_system_message(st.session_state[StateKeys.DATASET]),
        height=150,
        disabled=llm_integration_set_for_model,
    )

# TODO: Regenerate initial prompt on reset
with st.expander("Initial prompt", expanded=True):
    feature_to_repr_map = st.session_state[StateKeys.DATASET]._feature_to_repr_map
    if st.session_state[get_uniprot_state_key(selected_analysis_key)]:
        texts = [
            format_uniprot_annotation(
                st.session_state[StateKeys.ANNOTATION_STORE][feature],
                fields=selected_llm_chat[LLMKeys.SELECTED_UNIPROT_FIELDS],
            )
            for feature in regulated_genes_dict
        ]
        uniprot_info = f"{os.linesep}{os.linesep}".join(texts)
    else:
        uniprot_info = ""

    initial_prompt = st.text_area(
        " ",
        value=get_initial_prompt(
            plot_parameters,
            list(
                map(
                    feature_to_repr_map.get,
                    selected_llm_chat[LLMKeys.SELECTED_GENES_UP],
                )
            ),
            list(
                map(
                    feature_to_repr_map.get,
                    selected_llm_chat[LLMKeys.SELECTED_GENES_DOWN],
                )
            ),
            uniprot_info,
        ),
        height=200,
        disabled=llm_integration_set_for_model,
    )

    # a bit hacky but makes tool calling of `get_uniprot_info_for_search_string` much simpler
    st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = selected_llm_chat[
        LLMKeys.SELECTED_UNIPROT_FIELDS
    ].copy()

st.markdown(f"##### LLM Analysis with {model_name}")

c1, c2, _ = st.columns((0.2, 0.2, 0.6))
llm_submitted = c1.button(
    "Run LLM analysis ...", disabled=llm_integration_set_for_model
)

llm_reset = c2.button(
    "❌ Reset LLM analysis ...", disabled=not llm_integration_set_for_model
)
if llm_reset:
    del selected_llm_chat[model_name]
    st.rerun()


if not llm_integration_set_for_model:
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
            max_tokens=st.session_state[StateKeys.MAX_TOKENS],
        )

        selected_llm_chat[model_name] = llm_integration

        st.toast(
            f"{st.session_state[StateKeys.MODEL_NAME]} integration initialized successfully!",
            icon="✅",
        )

        with st.spinner("Processing initial prompt..."):
            llm_integration.chat_completion(initial_prompt, pin_message=True)

        st.rerun(scope="app")
    except AuthenticationError:
        st.warning(
            "Incorrect API key provided. Please enter a valid API key, it should look like this: sk-XXXXX"
        )
        st.stop()


@st.fragment
def llm_chat(
    llm_integration: LLMIntegration,
    show_all: bool = False,
    show_individual_tokens: bool = False,
):
    """The chat interface for the LLM analysis."""

    # TODO dump to file -> static file name, plus button to do so
    # Ideas: save chat as txt, without encoding objects, just put a replacement string.
    # Offer bulk download of zip with all figures (via plotly download as svg.).
    # Alternatively write it all in one pdf report using e.g. pdfrw and reportlab (I have code for that combo).

    # no. tokens spent
    messages, total_tokens, pinned_tokens = llm_integration.get_print_view(
        show_all=show_all
    )
    for message in messages:
        with st.chat_message(message[MessageKeys.ROLE]):
            st.markdown(message[MessageKeys.CONTENT])
            if (
                message[MessageKeys.PINNED]
                or not message[MessageKeys.IN_CONTEXT]
                or show_individual_tokens
            ):
                token_message = ""
                if message[MessageKeys.PINNED]:
                    token_message += ":pushpin: "
                if not message[MessageKeys.IN_CONTEXT]:
                    token_message += ":x: "
                if show_individual_tokens:
                    tokens = llm_integration.estimate_tokens(
                        [message], model=model_name
                    )
                    token_message += f"*tokens: {str(tokens)}*"
                st.markdown(token_message)
            for artifact in message[MessageKeys.ARTIFACTS]:
                if isinstance(artifact, pd.DataFrame):
                    st.dataframe(artifact)
                elif isinstance(
                    artifact, (PlotlyObject, plotly_object)
                ):  # TODO can there be non-plotly types here
                    st.plotly_chart(artifact)
                elif not isinstance(artifact, str):
                    st.warning("Don't know how to display artifact:")
                    st.write(artifact)

    st.markdown(
        f"*total tokens used: {str(total_tokens)}, tokens used for pinned messages: {str(pinned_tokens)}*"
    )

    if selected_llm_chat.get(LLMKeys.RECENT_CHAT_WARNINGS):
        st.warning("Warnings during last chat completion:")
        for warning in selected_llm_chat[LLMKeys.RECENT_CHAT_WARNINGS]:
            st.warning(str(warning.message).replace("\n", "\n\n"))

    if prompt := st.chat_input("Say something"):
        with st.chat_message(Roles.USER):
            st.markdown(prompt)
            if show_individual_tokens:
                st.markdown(
                    f"*tokens: {str(llm_integration.estimate_tokens([{MessageKeys.CONTENT:prompt}], model=model_name))}*"
                )
        with st.spinner("Processing prompt..."), warnings.catch_warnings(
            record=True
        ) as caught_warnings:
            llm_integration.chat_completion(prompt)
            selected_llm_chat[LLMKeys.RECENT_CHAT_WARNINGS] = caught_warnings

        st.rerun(scope="fragment")

    st.download_button(
        "Download chat log",
        llm_integration.get_chat_log_txt(),
        f"chat_log_{model_name}.txt",
        "text/plain",
    )

    st.markdown(
        "*icons: :pushpin: pinned message, :x: message no longer in context due to token limitations*"
    )


c1, c2 = st.columns((1, 2))
with c1:
    show_all = st.checkbox(
        "Show system messages",
        key="show_system_messages",
        help="Show all messages in the chat interface.",
    )
with c2:
    show_inidvidual_tokens = st.checkbox(
        "Show individual token estimates",
        key="show_individual_tokens",
        help="Show individual token estimates for each message.",
    )

llm_chat(
    selected_llm_chat[model_name],
    show_all,
    show_inidvidual_tokens,
)
