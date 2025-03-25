import os
from typing import Dict

import streamlit as st
from openai import AuthenticationError

from alphastats.dataset.keys import Cols
from alphastats.gui.utils.analysis import ResultComponent
from alphastats.gui.utils.analysis_helper import (
    display_figure,
    gather_uniprot_data,
)
from alphastats.gui.utils.llm_helper import (
    LLM_ENABLED_ANALYSIS,
    OLLAMA_BASE_URL,
    display_uniprot,
    format_analysis_key,
    get_df_for_protein_selector,
    init_llm_chat_state,
    protein_selector,
    show_llm_chat,
)
from alphastats.gui.utils.state_keys import LLMKeys, SavedAnalysisKeys, StateKeys
from alphastats.gui.utils.state_utils import (
    init_session_state,
)
from alphastats.gui.utils.ui_helper import (
    sidebar_info,
)
from alphastats.llm.llm_integration import LLMIntegration
from alphastats.llm.prompts import get_initial_prompt, get_system_message
from alphastats.llm.uniprot_utils import (
    format_uniprot_annotation,
    get_uniprot_state_key,
)

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()


st.markdown("## LLM Interpretation")

if StateKeys.DATASET not in st.session_state:
    st.info("Import data first.")
    st.stop()


##################################### Select Analysis #####################################

st.markdown("#### Select Analysis for LLM interpretation")

if not (
    available_analyses_keys := [
        key
        for key, analysis in st.session_state[StateKeys.SAVED_ANALYSES].items()
        if analysis[SavedAnalysisKeys.METHOD] in LLM_ENABLED_ANALYSIS
    ]
):
    st.info(
        f"Create a supported analysis first on the 'Analysis' page. Currently supported: {LLM_ENABLED_ANALYSIS}"
    )
    st.page_link("pages/05_Analysis.py", label="=> Goto Analysis page...")
    st.stop()

selected_analysis_key = st.selectbox(
    "Select analysis to interpret with LLM",
    available_analyses_keys,
    format_func=format_analysis_key,
    index=None if len(available_analyses_keys) > 1 else 0,
)

if (
    selected_analysis := st.session_state[StateKeys.SAVED_ANALYSES].get(
        selected_analysis_key, None
    )
) is None:
    st.stop()

if st.session_state[StateKeys.LLM_CHATS].get(selected_analysis_key) is None:
    st.session_state[StateKeys.LLM_CHATS][selected_analysis_key] = {}
selected_llm_chat = st.session_state[StateKeys.LLM_CHATS][selected_analysis_key]

model_name = st.session_state[StateKeys.MODEL_NAME]
llm_integration_set_for_model = selected_llm_chat.get(model_name, None) is not None

volcano_plot: ResultComponent = selected_analysis[SavedAnalysisKeys.RESULT]
plot_parameters: Dict = selected_analysis[SavedAnalysisKeys.PARAMETERS]

st.markdown(f"Parameters used for analysis: `{plot_parameters}`")


##################################### Analysis Input #####################################

st.markdown("#### Analysis Input to LLM")
c1, c2, c3 = st.columns((1, 1, 1))

##################################### Volcano plot #####################################

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

init_llm_chat_state(selected_llm_chat, upregulated_genes, downregulated_genes)

upregulated_genes_df = get_df_for_protein_selector(
    upregulated_genes, selected_llm_chat[LLMKeys.SELECTED_GENES_UP]
)
downregulated_genes_df = get_df_for_protein_selector(
    downregulated_genes, selected_llm_chat[LLMKeys.SELECTED_GENES_DOWN]
)


##################################### Genes of interest #####################################

with c1:
    st.markdown(
        "##### Select Genes of interest",
        help="Select which genes shall be used in the LLM interpretation.",
    )

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

selected_genes = (
    selected_llm_chat[LLMKeys.SELECTED_GENES_UP]
    + selected_llm_chat[LLMKeys.SELECTED_GENES_DOWN]
)
regulated_genes_dict = {
    gene: "up" if gene in selected_llm_chat[LLMKeys.SELECTED_GENES_UP] else "down"
    for gene in selected_genes
}

if not regulated_genes_dict:
    st.text("No genes selected for analysis.")
    st.stop()


##################################### Uniprot information #####################################

st.markdown(
    "##### Select Uniprot information",
    help="Select which information from Uniprot to supply to the LLM",
)

if st.button("Fetch UniProt data for selected proteins"):
    gather_uniprot_data(selected_genes)

display_uniprot(
    regulated_genes_dict,
    st.session_state[StateKeys.DATASET]._feature_to_repr_map,
    model_name=model_name,
    selected_analysis_key=selected_analysis_key,
    disabled=llm_integration_set_for_model,
)


##################################### System and initial prompt #####################################

st.markdown("##### System and initial prompt")
st.write(
    "The prompts are generated based on the above selection on genes and Uniprot information."
)
if st.button(
    "Update prompts with selected genes and UniProt information",
    disabled=llm_integration_set_for_model,
    help="Regenerate system message and initial prompt based on current selections",
):
    st.rerun(scope="app")

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
    if st.session_state.get(get_uniprot_state_key(selected_analysis_key), None):
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


##################################### LLM interpretation #####################################

st.markdown(f"#### LLM Interpretation with {model_name}")

c1, c2, _ = st.columns((0.2, 0.2, 0.6))
llm_submitted = c1.button(
    "Run LLM interpretation ...", disabled=llm_integration_set_for_model
)

llm_reset = c2.button(
    "❌ Reset LLM interpretation ...", disabled=not llm_integration_set_for_model
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

show_llm_chat(
    selected_llm_chat[model_name],
    selected_analysis_key,
    show_all,
    show_inidvidual_tokens,
)
