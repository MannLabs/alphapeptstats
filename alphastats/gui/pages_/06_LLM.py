from typing import Dict

import streamlit as st
from openai import AuthenticationError

from alphastats.dataset.keys import Cols, Regulation
from alphastats.gui.utils.analysis import ResultComponent
from alphastats.gui.utils.analysis_helper import (
    display_figure,
    gather_uniprot_data,
)
from alphastats.gui.utils.llm_config_helper import (
    format_model_config_for_display,
    get_model_config_by_id,
    get_test_status_icon,
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
    show_llm_chat,
)
from alphastats.gui.utils.state_keys import (
    LLMKeys,
    ModelKeys,
    SavedAnalysisKeys,
    StateKeys,
)
from alphastats.gui.utils.state_utils import (
    init_session_state,
)
from alphastats.gui.utils.ui_helper import (
    sidebar_info,
)
from alphastats.llm.llm_integration import LLMClientWrapper, LLMIntegration
from alphastats.llm.llm_utils import get_subgroups_for_each_group
from alphastats.llm.prompts import get_system_message

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()


st.markdown("## LLM Interpretation")

has_dataset = (dataset := st.session_state.get(StateKeys.DATASET, None)) is not None

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

result_component: ResultComponent = selected_analysis[SavedAnalysisKeys.RESULT]
plot_parameters: Dict = selected_analysis[SavedAnalysisKeys.PARAMETERS]
# group1 is the right side of the plot and has a positive fold-change if upregulated
# group2 is the left side of the plot and has a negative fold-change if upregulated

subgroups = get_subgroups_for_each_group(dataset.metadata) if has_dataset else {}

regulated_features_df = result_component.annotated_dataframe[
    result_component.annotated_dataframe[Cols.SIGNIFICANT] != Regulation.NON_SIG
]
regulated_features_dict = dict(
    zip(
        regulated_features_df[Cols.INDEX],
        regulated_features_df[Cols.SIGNIFICANT].tolist(),
    )
)

feature_to_repr_map = (
    dataset.id_holder.feature_to_repr_map
    if has_dataset
    else selected_analysis[SavedAnalysisKeys.ID_HOLDER].feature_to_repr_map
)

st.markdown(f"Parameters used for analysis: `{plot_parameters}`")


if st.session_state[StateKeys.LLM_CHATS].get(selected_analysis_key) is None:
    st.session_state[StateKeys.LLM_CHATS][selected_analysis_key] = {}

selected_llm_chat = st.session_state[StateKeys.LLM_CHATS][selected_analysis_key]

##################################### Select LLM Configuration #####################################


selected_llm_integration = selected_llm_chat.get(LLMKeys.LLM_INTEGRATION)
is_llm_integration_initialized = selected_llm_integration is not None

st.markdown("#### Select LLM Configuration")
if is_llm_integration_initialized and selected_llm_integration.has_client_wrapper:
    st.info(
        "LLM integration is already initialized for this analysis. "
        "To change the configuration, please reset the LLM interpretation first."
    )
else:
    available_configurations = st.session_state.get(StateKeys.LLM_CONFIGURATIONS, [])

    if not available_configurations:
        st.warning(
            "No LLM configurations found. Please configure at least one model first."
        )
        st.page_link(
            "pages_/09_LLM_Configuration.py",
            label="➔ Go to LLM Configuration page...",
        )
        st.stop()

    # Create selectbox with configurations
    config_options = {
        config[ModelKeys.ID]: config for config in available_configurations
    }
    config_ids = list(config_options.keys())

    # Get current selection or default to first config
    current_config_id = selected_llm_chat.get(LLMKeys.LLM_CONFIGURATION_ID)
    if current_config_id and current_config_id in config_ids:
        default_index = config_ids.index(current_config_id)
    else:
        default_index = 0 if config_ids else None

    selected_config_id = st.selectbox(
        "Select configuration to use for this analysis. Note: model will be locked once conversation is started.",
        options=config_ids,
        format_func=lambda config_id: format_model_config_for_display(
            config_options[config_id]
        ),
        index=default_index,
        key=f"config_selector_{selected_analysis_key}",
        help="Configuration is locked once LLM interpretation is initialized. Reset to change configuration.",
    )

    # Store selection in chat state
    if selected_config_id:
        selected_llm_chat[LLMKeys.LLM_CONFIGURATION_ID] = selected_config_id

    if not selected_llm_integration.has_client_wrapper():
        st.warning(
            "LLM integration is only partially initialized, as the current session was loaded from the store. "
            "Select a configuration and click 'Complete initialization' to associate the selected model with the current chat. "
            f"Note: the model associated with this session before was **{selected_llm_chat.get(LLMKeys.LLM_INTEGRATION).model_name}**"
        )
        if st.button("Complete initialization"):
            model_config = get_model_config_by_id(selected_config_id)
            client_wrapper = LLMClientWrapper(
                model_name=model_config[ModelKeys.MODEL_NAME],
                api_key=model_config.get(ModelKeys.API_KEY) or None,
                base_url=model_config.get(ModelKeys.BASE_URL) or None,
            )
            selected_llm_chat.get(
                LLMKeys.LLM_INTEGRATION
            ).client_wrapper = client_wrapper
            st.rerun()

##################################### Analysis Input #####################################

st.markdown("#### Analysis Input to LLM")
c1, _c2, _c3 = st.columns((1, 1, 1))

##################################### Volcano plot #####################################

if result_component.plot:
    with c1:
        st.markdown("##### Volcano plot")
        display_figure(result_component.plot)


if not regulated_features_dict:
    st.text("No genes of interest found.")
    st.stop()

# Separate upregulated and downregulated features
upregulated_features = [
    key
    for key in regulated_features_dict
    if regulated_features_dict[key] == Regulation.UP
]
downregulated_features = [
    key
    for key in regulated_features_dict
    if regulated_features_dict[key] == Regulation.DOWN
]


##################################### Initialize LLM chat state and sync session state #####################################


init_llm_chat_state(
    selected_llm_chat,
    upregulated_features,
    downregulated_features,
    plot_parameters,
    feature_to_repr_map,
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


display_uniprot(
    regulated_features_dict,
    feature_to_repr_map,
    model_name=get_model_config_by_id(
        selected_llm_chat.get(LLMKeys.LLM_CONFIGURATION_ID)
    )[ModelKeys.MODEL_NAME],
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
        value=get_system_message(subgroups),
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

# Retrieve configuration for display
display_config_id = selected_llm_chat.get(LLMKeys.LLM_CONFIGURATION_ID)
if display_config_id:
    display_config = get_model_config_by_id(display_config_id)
    if not is_llm_integration_initialized:
        st.info("You may change the model in the dropdown on the top of the page.")
    if display_config:
        st.markdown(
            f"#### LLM Interpretation with {display_config[ModelKeys.MODEL_NAME]}"
        )

        test_status = display_config.get(ModelKeys.TEST_STATUS, "not_tested")
        icon = get_test_status_icon(test_status)
        opt = (
            f"\n**Base URL:** {display_config[ModelKeys.BASE_URL]}"
            if display_config.get(ModelKeys.BASE_URL)
            else ""
        )
        st.info(
            f"**Model:** {display_config[ModelKeys.MODEL_NAME]}\n"
            + f"**Max Tokens:** {display_config[ModelKeys.MAX_TOKENS]:,}\n"
            + f"**Test Status:** {icon} {test_status}"
            + opt
        )

    else:
        st.warning(
            "Configuration no longer exists. Please select a new configuration and reset."
        )
else:
    st.markdown("#### LLM Interpretation")
    st.warning("No configuration selected")


c1, c2, c3, _ = st.columns((0.2, 0.2, 0.2, 0.6))
llm_submitted = c1.button(
    "Run LLM interpretation ...", disabled=is_llm_integration_initialized
)

llm_reset = c2.button(
    "❌ Reset LLM interpretation ...", disabled=not is_llm_integration_initialized
)

if llm_reset:
    del selected_llm_chat[LLMKeys.LLM_CONFIGURATION_ID]
    del selected_llm_chat[LLMKeys.LLM_INTEGRATION]
    st.rerun()

config_id = selected_llm_chat.get(LLMKeys.LLM_CONFIGURATION_ID)
model_config = get_model_config_by_id(config_id)

if not is_llm_integration_initialized:
    if not llm_submitted:
        st.stop()

    try:
        # Use configuration values for initialization
        client_wrapper = LLMClientWrapper(
            model_name=model_config[ModelKeys.MODEL_NAME],
            api_key=model_config.get(ModelKeys.API_KEY) or None,
            base_url=model_config.get(ModelKeys.BASE_URL) or None,
        )

        llm_integration = LLMIntegration(
            client_wrapper=client_wrapper,
            system_message=system_message,
            dataset=dataset,
            max_tokens=model_config[ModelKeys.MAX_TOKENS],
        )

        st.toast(
            f"{model_config[ModelKeys.MODEL_NAME]} integration initialized successfully!",
            icon="✅",
        )

        with st.spinner("Processing initial prompt..."):
            # Do not pass tools on first chat completion, since not all models handle them correctly and we want to make sure the (CoT) initial prompt is processed correctly.
            llm_integration.chat_completion(
                initial_prompt, pin_message=True, pass_tools=False
            )

        selected_llm_chat[LLMKeys.LLM_INTEGRATION] = llm_integration

        st.rerun(scope="app")
    except AuthenticationError:
        st.error(
            f"❌ Authentication failed for {model_config[ModelKeys.MODEL_NAME]}. "
            "The API key in the configuration is incorrect or invalid."
        )
        st.info(
            "Please update the API key in the configuration. It should look like: sk-XXXXX"
        )
        st.page_link(
            "pages_/09_LLM_Configuration.py",
            label="➔ Go to LLM Configuration page to update...",
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

llm_integration = selected_llm_chat[LLMKeys.LLM_INTEGRATION]
show_llm_chat(
    llm_integration,
    selected_analysis_key,
    show_all,
    show_individual_tokens,
)
