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
    configure_initial_prompt,
    display_uniprot,
    enrichment_analysis,
    format_analysis_key,
    get_selected_regulated_features,
    init_llm_chat_state,
    on_select_new_analysis_fill_state,
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
from alphastats.llm.llm_integration import LLMClientWrapper, LLMIntegration, ModelFlags
from alphastats.llm.prompts import get_system_message

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()


st.markdown("## LLM Interpretation")

if StateKeys.DATASET not in st.session_state:
    st.info("Import data first.")
    st.stop()
feature_to_repr_map = st.session_state[StateKeys.DATASET]._feature_to_repr_map


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
    st.page_link("pages_/05_Analysis.py", label="➔ Go to Analysis page...")
    st.stop()

selected_analysis_key = st.selectbox(
    "Select analysis to interpret with LLM",
    available_analyses_keys,
    format_func=format_analysis_key,
    index=None if len(available_analyses_keys) > 1 else 0,
    on_change=on_select_new_analysis_fill_state,
    key=StateKeys.SELECTED_ANALYSIS,
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

regulated_features_df = volcano_plot.annotated_dataframe[
    volcano_plot.annotated_dataframe["significant"] != "non_sig"
]
regulated_features_dict = dict(
    zip(
        regulated_features_df[Cols.INDEX], regulated_features_df["significant"].tolist()
    )
)

if not regulated_features_dict:
    st.text("No genes of interest found.")
    st.stop()


# Separate upregulated and downregulated features
upregulated_features = [
    key for key in regulated_features_dict if regulated_features_dict[key] == "up"
]
downregulated_features = [
    key for key in regulated_features_dict if regulated_features_dict[key] == "down"
]


##################################### Initialize LLM chat state and sync session state #####################################


init_llm_chat_state(
    selected_llm_chat,
    upregulated_features,
    downregulated_features,
    plot_parameters,
    feature_to_repr_map,
)


##################################### Proteins of interest #####################################

with c1:
    st.markdown(
        "##### Select proteins of interest",
        help="Select which features shall be used in the LLM interpretation.",
    )

    protein_selector(
        upregulated_features,
        "Upregulated Proteins",
        selected_analysis_key,
        state_key=LLMKeys.SELECTED_FEATURES_UP,
    )

with c2:
    st.markdown("##### ")
    protein_selector(
        downregulated_features,
        "Downregulated Proteins",
        selected_analysis_key,
        state_key=LLMKeys.SELECTED_FEATURES_DOWN,
    )

selected_features, regulated_features_dict = get_selected_regulated_features(
    selected_llm_chat
)

if not regulated_features_dict:
    st.text("No genes selected for analysis.")
    st.stop()


##################################### Uniprot information #####################################

st.markdown(
    "##### Select Uniprot information",
    help="Select which information from Uniprot to supply to the LLM",
)

if st.button("Fetch UniProt data for selected proteins"):
    gather_uniprot_data(selected_features)

is_llm_integration_initialized = (
    selected_llm_chat.get(LLMKeys.LLM_INTEGRATION) is not None
)


display_uniprot(
    regulated_features_dict,
    st.session_state[StateKeys.DATASET]._feature_to_repr_map,
    model_name=selected_llm_chat[LLMKeys.MODEL_NAME],
    selected_analysis_key=selected_analysis_key,
    disabled=is_llm_integration_initialized,
)

##################################### Enrichment analysis ####################################
enrichment_analysis(selected_llm_chat, disabled=is_llm_integration_initialized)


##################################### System and initial prompt #####################################

st.markdown("##### System and initial prompt")
st.write(
    "The prompts are generated based on the above selection of proteins and Uniprot information."
)

with st.expander("System message", expanded=False):
    system_message = st.text_area(
        " ",
        value=get_system_message(st.session_state[StateKeys.DATASET]),
        height=150,
        disabled=is_llm_integration_initialized,
    )

# TODO: Regenerate initial prompt on reset
with st.expander("Initial prompt", expanded=True):
    initial_prompt = configure_initial_prompt(
        selected_llm_chat,
        plot_parameters,
        feature_to_repr_map,
        disabled=is_llm_integration_initialized,
    )

    # a bit hacky but makes tool calling of `get_uniprot_info_for_search_string` much simpler
    st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = selected_llm_chat[
        LLMKeys.SELECTED_UNIPROT_FIELDS
    ].copy()


##################################### LLM interpretation #####################################

st.markdown(f"#### LLM Interpretation with {selected_llm_chat[LLMKeys.MODEL_NAME]}")

st.info(
    f"Model: {selected_llm_chat[LLMKeys.MODEL_NAME]} Max tokens: {selected_llm_chat[LLMKeys.MAX_TOKENS]}"
)

if (
    model := selected_llm_chat[LLMKeys.MODEL_NAME]
) in ModelFlags.REQUIRES_API_KEY and not st.session_state.get(StateKeys.OPENAI_API_KEY):
    st.page_link(
        "pages_/01_Home.py",
        label=f"❗ Please configure an API key to use the {model} model on the ➔ Home page",
    )
    st.stop()

c1, c2, _ = st.columns((0.2, 0.2, 0.6))
llm_submitted = c1.button(
    "Run LLM interpretation ...", disabled=is_llm_integration_initialized
)

llm_reset = c2.button(
    "❌ Reset LLM interpretation ...", disabled=not is_llm_integration_initialized
)
if llm_reset:
    del selected_llm_chat[LLMKeys.LLM_INTEGRATION]
    del selected_llm_chat[LLMKeys.MODEL_NAME]
    del selected_llm_chat[LLMKeys.MAX_TOKENS]
    del selected_llm_chat[LLMKeys.IS_INITIALIZED]
    st.rerun()


if not is_llm_integration_initialized:
    if not llm_submitted:
        st.stop()

    try:
        client_wrapper = LLMClientWrapper(
            model_name=selected_llm_chat[LLMKeys.MODEL_NAME],
            api_key=st.session_state[StateKeys.OPENAI_API_KEY],
            base_url=st.session_state[StateKeys.BASE_URL],
        )

        llm_integration = LLMIntegration(
            client_wrapper=client_wrapper,
            system_message=system_message,
            dataset=st.session_state[StateKeys.DATASET],
            genes_of_interest=list(regulated_features_dict.keys()),
            max_tokens=selected_llm_chat[StateKeys.MAX_TOKENS],
        )

        selected_llm_chat[LLMKeys.LLM_INTEGRATION] = llm_integration
        selected_llm_chat[LLMKeys.IS_INITIALIZED] = True

        st.toast(
            f"{selected_llm_chat[LLMKeys.MODEL_NAME]} integration initialized successfully!",
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
    show_individual_tokens = st.checkbox(
        "Show individual token estimates",
        key="show_individual_tokens",
        help="Show individual token estimates for each message.",
    )

show_llm_chat(
    selected_llm_chat[LLMKeys.LLM_INTEGRATION],
    selected_analysis_key,
    show_all,
    show_individual_tokens,
)
