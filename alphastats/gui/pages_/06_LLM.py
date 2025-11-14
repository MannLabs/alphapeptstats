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
    format_config_for_display,
    get_config_by_id,
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
from alphastats.gui.utils.state_keys import LLMKeys, SavedAnalysisKeys, StateKeys
from alphastats.gui.utils.state_utils import (
    init_session_state,
)
from alphastats.gui.utils.ui_helper import (
    sidebar_info,
)
from alphastats.llm.llm_integration import LLMClientWrapper, LLMIntegration, Model
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

st.markdown("#### Select LLM Configuration")

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

is_llm_integration_initialized = (
    selected_llm_chat.get(LLMKeys.LLM_INTEGRATION) is not None
)

# Create selectbox with configurations
config_options = {config["id"]: config for config in available_configurations}
config_ids = list(config_options.keys())

# Get current selection or default to first config
current_config_id = selected_llm_chat.get(LLMKeys.LLM_CONFIGURATION_ID)
if current_config_id and current_config_id in config_ids:
    default_index = config_ids.index(current_config_id)
else:
    default_index = 0 if config_ids else None

selected_config_id = st.selectbox(
    "Select configuration to use for this analysis",
    options=config_ids,
    format_func=lambda config_id: format_config_for_display(config_options[config_id]),
    index=default_index,
    disabled=is_llm_integration_initialized,
    key=f"config_selector_{selected_analysis_key}",
    help="Configuration is locked once LLM interpretation is initialized. Reset to change configuration.",
)

# Store selection in chat state
if selected_config_id:
    selected_llm_chat[LLMKeys.LLM_CONFIGURATION_ID] = selected_config_id
    selected_config = config_options[selected_config_id]

    st.markdown(f"**Model:** {selected_config['model_name']}")
    st.markdown(f"**Max Tokens:** {selected_config['max_tokens']:,}")
    test_status = selected_config.get("test_status", "not_tested")
    icon = get_test_status_icon(test_status)
    st.markdown(f"**Test Status:** {icon} {test_status}")

    if selected_config.get("base_url"):
        st.markdown(f"**Base URL:** {selected_config['base_url']}")

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


# Get model name from configuration for display_uniprot
uniprot_config = get_config_by_id(selected_llm_chat.get(LLMKeys.LLM_CONFIGURATION_ID))
uniprot_model_name = uniprot_config["model_name"] if uniprot_config else "unknown"

display_uniprot(
    regulated_features_dict,
    feature_to_repr_map,
    model_name=uniprot_model_name,
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
    display_config = get_config_by_id(display_config_id)
    if display_config:
        st.markdown(f"#### LLM Interpretation with {display_config['model_name']}")

        st.info(f"**Model:** {display_config['model_name']}")
        st.info(f"**Max Tokens:** {display_config['max_tokens']:,}")
        test_status = display_config.get("test_status", "not_tested")
        icon = get_test_status_icon(test_status)
        st.info(f"**Test Status:** {icon}")
    else:
        st.warning(
            "Configuration no longer exists. Please select a new configuration and reset."
        )
else:
    st.markdown("#### LLM Interpretation")
    st.warning("No configuration selected")

# Validate configuration is selected
config_id = selected_llm_chat.get(LLMKeys.LLM_CONFIGURATION_ID)
if not config_id:
    st.error("Please select a configuration first")
    st.stop()

# Retrieve configuration
selected_config = get_config_by_id(config_id)
if selected_config is None:
    st.error(
        "Selected configuration no longer exists. Please select a different configuration."
    )
    st.stop()

# Validate API key if required
model_name = selected_config["model_name"]
if Model(model_name).requires_api_key() and not selected_config.get("api_key"):
    st.error(
        f"API key is required for {model_name}. Please update the configuration on the LLM Configuration page."
    )
    st.page_link(
        "pages_/09_LLM_Configuration.py",
        label="➔ Go to LLM Configuration page...",
    )
    st.stop()

c1, c2, c3, _ = st.columns((0.2, 0.2, 0.2, 0.6))
llm_submitted = c1.button(
    "Run LLM interpretation ...", disabled=is_llm_integration_initialized
)

llm_reset = c2.button(
    "❌ Reset LLM interpretation ...", disabled=not is_llm_integration_initialized
)

if llm_reset:
    del selected_llm_chat[LLMKeys.LLM_CONFIGURATION_ID]
    st.rerun()


if not is_llm_integration_initialized:
    if not llm_submitted:
        st.stop()

    try:
        # Use configuration values for initialization
        client_wrapper = LLMClientWrapper(
            model_name=selected_config["model_name"],
            api_key=selected_config.get("api_key") or None,
            base_url=selected_config.get("base_url") or None,
        )

        llm_integration = LLMIntegration(
            client_wrapper=client_wrapper,
            system_message=system_message,
            dataset=dataset,
            max_tokens=selected_config["max_tokens"],
        )

        st.toast(
            f"{selected_config['model_name']} integration initialized successfully!",
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
            f"❌ Authentication failed for {selected_config['model_name']}. "
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
