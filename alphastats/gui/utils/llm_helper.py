from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from alphastats.dataset.keys import ConstantsClass
from alphastats.dataset.plotting import plotly_object
from alphastats.gui.utils.analysis import NewAnalysisOptions
from alphastats.gui.utils.state_keys import (
    MODEL_SYNCED_LLM_KEYS,
    WIDGET_SYNCED_LLM_KEYS,
    DefaultStates,
    KeySyncNames,
    LLMKeys,
    SavedAnalysisKeys,
    StateKeys,
)
from alphastats.gui.utils.ui_helper import has_llm_support
from alphastats.llm.enrichment_analysis import (
    HUMAN_ORGANISM_ID,
    get_enrichment_data,
    gprofiler_organisms,
)
from alphastats.llm.llm_integration import (
    LLMClientWrapper,
    LLMIntegration,
    MessageKeys,
    ModelFlags,
    Models,
    Roles,
)
from alphastats.llm.prompts import (
    LLMInstructionKeys,
    _get_experimental_design_prompt,
    _get_initial_instruction,
    _get_protein_data_prompt,
    get_initial_prompt,
)
from alphastats.llm.uniprot_utils import (
    ExtractedUniprotFields,
    format_uniprot_annotation,
)
from alphastats.plots.plot_utils import PlotlyObject

LLM_ENABLED_ANALYSIS = (
    [NewAnalysisOptions.DIFFERENTIAL_EXPRESSION_TWO_GROUPS] if has_llm_support() else []
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# TODO: Deduplicate this code without introducing a circular import


class EnrichmentAnalysisKeys(metaclass=ConstantsClass):
    """Keys for accessing the session state for enrichment analysis."""

    PARAMETERS = "parameters"
    DIFEXPRESSED = "difexpressed"
    ORGANISM_ID = "organism_id"
    TOOL = "tool"
    INCLUDE_BACKGROUND = "include_background"
    RESULT = "result"


@st.fragment
def llm_config() -> None:
    """Show the configuration options for the LLM interpretation."""

    current_model = st.session_state.get(StateKeys.MODEL_NAME, None)
    current_base_url = st.session_state.get(StateKeys.BASE_URL, None)

    c1, _ = st.columns((1, 2))
    with c1:
        models = Models.get_values()
        new_model = st.selectbox(
            "Select LLM",
            models,
            index=models.index(current_model) if current_model is not None else 0,
        )
        if new_model != current_model:
            st.session_state[StateKeys.MODEL_NAME] = new_model
            on_change_save_state()

        requires_api_key = new_model in ModelFlags.REQUIRES_API_KEY

        new_base_url = current_base_url
        api_key = st.text_input(
            f"Enter API Key and press Enter {'' if requires_api_key else '(optional)'}",
            type="password",
        )
        set_api_key(api_key)

        if new_model in ModelFlags.REQUIRES_BASE_URL:
            new_base_url = st.text_input(
                "base url",
                value=current_base_url,
            )
            if new_base_url != current_base_url:
                st.session_state[StateKeys.BASE_URL] = new_base_url
                on_change_save_state()
            st.info(f"Expecting Ollama API at {new_base_url}.")

        test_connection = st.button("Test connection")
        if test_connection:
            with st.spinner(f"Testing connection to {new_model}.."):
                error = llm_connection_test(
                    model_name=new_model,
                    api_key=st.session_state[StateKeys.OPENAI_API_KEY],
                    base_url=new_base_url,
                )
                if error is None:
                    st.success(f"Connection to {new_model} successful!")
                else:
                    st.error(f"Connection to {new_model} failed: {str(error)}")

        tokens = st.number_input(
            "Maximal number of tokens",
            step=1000,
            value=st.session_state[StateKeys.MAX_TOKENS],
        )
        if tokens != st.session_state[StateKeys.MAX_TOKENS]:
            st.session_state[StateKeys.MAX_TOKENS] = tokens
            on_change_save_state()

        if current_model != new_model or new_base_url != current_base_url:
            st.rerun(scope="app")


def format_analysis_key(key: str) -> str:
    """Pretty print an analysis referenced by `key`."""
    analysis = st.session_state[StateKeys.SAVED_ANALYSES][key]
    return f"[{key}] #{analysis[SavedAnalysisKeys.NUMBER]} {analysis[SavedAnalysisKeys.METHOD]} {analysis[SavedAnalysisKeys.PARAMETERS]}"


def init_llm_chat_state(
    selected_llm_chat: dict[str, Any],
    upregulated_features: list[str],
    downregulated_features: list[str],
    plot_parameters: dict[str, Any],
    feature_to_repr_map: dict[str, str],
) -> None:
    """Initialize the state for a given llm_chat."""
    if LLMKeys.RECENT_CHAT_WARNINGS not in selected_llm_chat:
        selected_llm_chat[LLMKeys.RECENT_CHAT_WARNINGS] = []

    if selected_llm_chat.get(LLMKeys.SELECTED_UNIPROT_FIELDS) is None:
        selected_llm_chat[LLMKeys.SELECTED_UNIPROT_FIELDS] = (
            DefaultStates.SELECTED_UNIPROT_FIELDS.copy()
        )

    if selected_llm_chat.get(LLMKeys.SELECTED_FEATURES_UP) is None:
        selected_llm_chat[LLMKeys.SELECTED_FEATURES_UP] = upregulated_features
    if selected_llm_chat.get(LLMKeys.SELECTED_FEATURES_DOWN) is None:
        selected_llm_chat[LLMKeys.SELECTED_FEATURES_DOWN] = downregulated_features
    if selected_llm_chat.get(LLMKeys.ENRICHMENT_ANALYSIS) is None:
        selected_llm_chat[LLMKeys.ENRICHMENT_ANALYSIS] = {
            EnrichmentAnalysisKeys.PARAMETERS: {},
            EnrichmentAnalysisKeys.RESULT: None,
        }
    if selected_llm_chat.get(LLMKeys.ENRICHMENT_COLUMNS) is None:
        selected_llm_chat[LLMKeys.ENRICHMENT_COLUMNS] = []

    if selected_llm_chat.get(LLMKeys.IS_INITIALIZED) is None:
        selected_llm_chat[LLMKeys.IS_INITIALIZED] = False

    if selected_llm_chat.get(LLMKeys.PROMPT_EXPERIMENTAL_DESIGN) is None:
        experimental_design_prompt, protein_data_prompt, initial_instructions = (
            initialize_initial_prompt_modules(
                selected_llm_chat, plot_parameters, feature_to_repr_map
            )
        )
        selected_llm_chat[LLMKeys.PROMPT_EXPERIMENTAL_DESIGN] = (
            experimental_design_prompt
        )
        selected_llm_chat[LLMKeys.PROMPT_PROTEIN_DATA] = protein_data_prompt
        selected_llm_chat[LLMKeys.PROMPT_INSTRUCTIONS] = initial_instructions

    # TODO model name is determined when loading LLM page -> need better model selection.
    if not selected_llm_chat[LLMKeys.IS_INITIALIZED]:
        selected_llm_chat[LLMKeys.MODEL_NAME] = st.session_state[StateKeys.MODEL_NAME]
        selected_llm_chat[LLMKeys.BASE_URL] = st.session_state[StateKeys.BASE_URL]
        selected_llm_chat[LLMKeys.MAX_TOKENS] = st.session_state[StateKeys.MAX_TOKENS]

    on_select_new_analysis_fill_state()


def initialize_initial_prompt_modules(
    llm_chat: dict[str, Any],
    plot_parameters: dict[str, Any],
    feature_to_repr_map: dict[str, str],
) -> None:
    _, regulated_features_dict = get_selected_regulated_features(llm_chat)

    experimental_design_prompt = _get_experimental_design_prompt(plot_parameters)

    if llm_chat.get(StateKeys.INCLUDE_UNIPROT_INTO_INITIAL_PROMPT):
        texts = [
            format_uniprot_annotation(
                st.session_state[StateKeys.ANNOTATION_STORE][feature],
                fields=llm_chat[LLMKeys.SELECTED_UNIPROT_FIELDS],
            )
            for feature in regulated_features_dict
        ]
        uniprot_info = f"{os.linesep}{os.linesep}".join(texts)
    else:
        uniprot_info = ""
    enrichment_data = (
        None
        if llm_chat[LLMKeys.ENRICHMENT_ANALYSIS].get(EnrichmentAnalysisKeys.RESULT)
        is None
        else llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][EnrichmentAnalysisKeys.RESULT][
            llm_chat[LLMKeys.ENRICHMENT_COLUMNS]
        ]
    )
    protein_data_prompt = _get_protein_data_prompt(
        llm_chat[LLMKeys.SELECTED_FEATURES_UP],
        llm_chat[LLMKeys.SELECTED_FEATURES_DOWN],
        uniprot_info,
        feature_to_repr_map=feature_to_repr_map,
        parameter_dict=plot_parameters,
        enrichment_data=enrichment_data,
    )

    initial_instruction = _get_initial_instruction(LLMInstructionKeys.SIMPLE)

    return experimental_design_prompt, protein_data_prompt, initial_instruction


@st.fragment
def protein_selector(
    regulated_features: list[str],
    title: str,
    selected_analysis_key: str,
    state_key: str,
) -> None:
    """Creates a data editor for protein selection and returns the selected proteins.

    Args:
        regulated_features: List of regulated features to display in the table
        title: Title to display above the editor
        selected_analysis_key: Key to access the selected analysis in the session state
        state_key: Key to access the selected proteins in the selected analysis

    Returns:
        None
    """
    st.write(title)
    selected_analysis_session_state = st.session_state[StateKeys.LLM_CHATS][
        selected_analysis_key
    ]
    df = get_df_for_protein_selector(
        regulated_features, selected_analysis_session_state[state_key]
    )
    if len(df) == 0:
        st.markdown("No significant proteins.")
        return
    c1, c2 = st.columns([1, 1])

    if c1.button("Select all", help=f"Select all {title} for analysis"):
        selected_analysis_session_state[state_key] = df["Protein"].tolist()
        st.rerun(scope="fragment")
    if c2.button("Select none", help=f"Select no {title} for analysis"):
        selected_analysis_session_state[state_key] = []
        st.rerun(scope="fragment")
    deselect = st.text_input(
        "Deselect list",
        placeholder="Enter comma-separated feature ids to deselect",
        key=f"{state_key}_deselect",
        help="This is so papers can be reproduced exactly. Make cutoffs that include all originally reported significant proteins, then deselect the ones, that were not reported significant in the original paper.",
    )
    if deselect:
        deselect = [x.strip() for x in deselect.split(",")]
        new_selection = [
            x for x in selected_analysis_session_state[state_key] if x not in deselect
        ]
        if new_selection != selected_analysis_session_state[state_key]:
            selected_analysis_session_state[state_key] = new_selection
            st.rerun(scope="fragment")

    edited_df = st.data_editor(
        df,
        column_config={
            "Selected": st.column_config.CheckboxColumn(
                "Include?",
                help="Check to include this feature in analysis",
                default=True,
            ),
            "Gene": st.column_config.TextColumn(
                "Gene",
                help="The gene name to be included in the analysis",
            ),
        },
        disabled=["Gene", "Protein"],
        hide_index=True,
        # explicitly setting key: otherwise it's calculated from the data which causes problems if two analysis are exactly mirrored
        key=f"{state_key}_data_editor",
    )
    # Extract the selected features
    new_list = edited_df.loc[edited_df["Selected"], "Protein"].tolist()
    if new_list != selected_analysis_session_state[state_key]:
        selected_analysis_session_state[state_key] = new_list
        st.rerun(scope="fragment")


def get_df_for_protein_selector(
    proteins: list[str], selected: list[str]
) -> pd.DataFrame:
    """Create a DataFrame for the protein selector.

    Args:
        proteins (List[str]): A list of proteins.

    Returns:
        pd.DataFrame: A DataFrame with 'Gene', 'Selected', 'Protein' columns.
    """
    return pd.DataFrame(
        {
            "Gene": [
                st.session_state[StateKeys.DATASET]._feature_to_repr_map[protein]
                for protein in proteins
            ],
            "Selected": [protein in selected for protein in proteins],
            "Protein": proteins,
        }
    )


def get_display_proteins_html(
    protein_ids: list[str],
    is_upregulated: True,
    annotation_store: dict[str, dict],
    feature_to_repr_map: dict[str, str],
) -> str:
    """
    Get HTML code for displaying a list of proteins, color according to expression.

    Args:
        protein_ids (list[str]): a list of proteins.
        is_upregulated (bool): whether the proteins are up- or down-regulated.
    """

    warnings.warn(
        "This function will be deprecated in version 2.0 because it is no longer used.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    uniprot_url = "https://www.uniprot.org/uniprotkb/"

    color = "green" if is_upregulated else "red"
    protein_ids_html = "".join(
        f'<a href = {uniprot_url + annotation_store[protein].get("primaryAccession",protein)}><li style="color: {color};">{feature_to_repr_map[protein]}</li></a>'
        for protein in protein_ids
    )

    return f"<ul>{protein_ids_html}</ul>"


def set_api_key(api_key: str = None) -> None:
    """Put the API key in the session state.

    If provided, use the `api_key`.
    If not, take the key from the secrets.toml file.
    Show a message if the file is not found.

    Args:
        api_key (str, optional): The API key. Defaults to None.
    """
    if not api_key:
        api_key = st.session_state.get(StateKeys.OPENAI_API_KEY, None)

    if api_key:
        st.info(f"API key set: {api_key[:3]}{(len(api_key)-6)*'*'}{api_key[-3:]}")
    else:
        try:
            if Path("./.streamlit/secrets.toml").exists():
                api_key = st.secrets["api_key"]
                st.toast("API key loaded from secrets.toml.", icon="✅")
            else:
                st.info(
                    "Please enter an OpenAI key or provide it in a secrets.toml file in the "
                    "alphastats/gui/.streamlit directory like "
                    "`api_key = <key>`"
                )
        except KeyError:
            st.error("API key not found in secrets.toml .")
        except Exception as e:
            st.error(f"Error loading API key: {e}.")

    st.session_state[StateKeys.OPENAI_API_KEY] = api_key


def llm_connection_test(
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str | None:
    """Test the connection to the LLM API, return None in case of success, error message otherwise."""
    try:
        llm = LLMIntegration(
            LLMClientWrapper(model_name, base_url=base_url, api_key=api_key),
            load_tools=False,
        )
        llm.chat_completion(
            "This is a test. Simply respond 'yes' if you got this message."
        )
        return None

    except Exception as e:
        return str(e)


# Unused now, but could be useful in the future
# TODO: Remove this by end of year if still unused.
def get_display_available_uniprot_info(regulated_features: list[str]) -> dict:
    """
    Retrieves and formats UniProt information for a list of regulated features.

    Note: The information is retrieved from the `annotation_store` in the `session_state`, which is filled when the LLM interpretation is set up from the anlaysis page.

    Args:
        regulated_features (list): A list of features for which UniProt information is to be retrieved.
    Returns:
        dict: A dictionary where each key is a feature representation and the value is another dictionary
              containing the 'protein ids' and 'generated text' with formatted UniProt information or an error message, starting with ERROR, so it can be filtered before passing on to the LLM.
    """
    text_repr = {}
    for feature in regulated_features:
        try:
            text = format_uniprot_annotation(
                st.session_state[StateKeys.ANNOTATION_STORE][feature]
            )
        except Exception as e:
            text = f"ERROR: {e}"
        text_repr[st.session_state[StateKeys.DATASET]._feature_to_repr_map[feature]] = {
            "protein ids": feature,
            "generated text": text,
        }
    return text_repr


# TODO: Write test for this display
@st.fragment
def display_uniprot(
    regulated_features_dict: dict,
    feature_to_repr_map: dict,
    model_name: str,
    selected_analysis_key: str,
    *,
    disabled=False,
):
    """Display the interface for selecting fields from UniProt information, including a preview of the selected fields."""
    all_fields = ExtractedUniprotFields.get_values()
    if any(
        feature not in st.session_state[StateKeys.ANNOTATION_STORE]
        for feature in regulated_features_dict
    ):
        st.info(
            "No or incomplete UniProt data stored for the selected proteins. Please run UniProt data fetching first to ensure correct annotation from Protein IDs instead of gene names."
        )
        return

    st.markdown(
        "We recommend providing at least limited information from Uniprot for all proteins as part of the initial "
        "prompt to avoid misinterpretation of gene names or ids by the LLM. You can edit the selection of fields to "
        "include while chatting for on-the-fly demand for more information."
    )
    c1, c2, c3, c4, c5, c6 = st.columns((1, 1, 1, 1, 1, 1))
    selected_analysis_session_state = st.session_state[StateKeys.LLM_CHATS][
        selected_analysis_key
    ]
    if c1.button("Select all", disabled=disabled):
        selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS] = all_fields
        st.rerun(scope="fragment")
    if c2.button("Select none", disabled=disabled):
        selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS] = []
        st.rerun(scope="fragment")
    if c3.button("Recommended selection", disabled=disabled):
        selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS] = (
            DefaultStates.SELECTED_UNIPROT_FIELDS.copy()
        )
        st.rerun(scope="fragment")
    with c4:
        texts = [
            format_uniprot_annotation(
                st.session_state[StateKeys.ANNOTATION_STORE].get(feature, {}),
                fields=selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS],
            )
            for feature in regulated_features_dict
        ]
        tokens = LLMIntegration.estimate_tokens(
            [{MessageKeys.CONTENT: text} for text in texts], model=model_name
        )
        st.markdown(f"Total tokens: {tokens:.0f}")
    with c5:
        # this is required to persist the state of the "Integrate into initial prompt" checkbox for different analyses

        st.checkbox(
            "Integrate into initial prompt",
            help="If this is ticked and the initial prompt is updated, the Uniprot information will be included in the prompt and the instructions regarding uniprot will change to onl;y look up more information if explicitly asked to do so. Make sure that the total tokens are below the message limit of your LLM.",
            key=StateKeys.INCLUDE_UNIPROT_INTO_INITIAL_PROMPT,
            disabled=disabled,
            on_change=on_change_save_state,
        )

    c1, c2 = st.columns((1, 3))
    with c1, st.expander("Show options", expanded=not disabled):
        selected_fields = []
        for field in all_fields:
            if st.checkbox(
                field,
                value=field
                in selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS],
                disabled=disabled,
            ):
                selected_fields.append(field)
        if set(selected_fields) != set(
            selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS]
        ):
            selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS] = (
                selected_fields
            )
            st.rerun(scope="fragment")
    with c2, st.expander("Show preview", expanded=True):
        # TODO: Fix desync on rerun (widget state not updated on rerun, value becomes ind0)
        preview_feature = st.selectbox(
            "Feature id",
            options=[
                feature
                for feature in regulated_features_dict
                if feature in st.session_state[StateKeys.ANNOTATION_STORE]
            ],
            format_func=lambda x: feature_to_repr_map[x],
        )
        if preview_feature is not None:
            uniprot_url = "https://www.uniprot.org/uniprotkb/"
            st.markdown(
                f"[Open in Uniprot ...]({uniprot_url + st.session_state[StateKeys.ANNOTATION_STORE][preview_feature]['primaryAccession']})"
            )
            st.markdown(f"Text generated from feature id {preview_feature}:")
            st.markdown(
                format_uniprot_annotation(
                    st.session_state[StateKeys.ANNOTATION_STORE][preview_feature],
                    fields=selected_analysis_session_state[
                        LLMKeys.SELECTED_UNIPROT_FIELDS
                    ],
                )
            )


def on_select_new_analysis_fill_state() -> None:
    """Upon selecting a new analysis set the values for mirrored session state keys before rerunning the app."""
    selected_chat = st.session_state[StateKeys.LLM_CHATS].get(
        st.session_state[StateKeys.SELECTED_ANALYSIS], {}
    )

    for synced_key in WIDGET_SYNCED_LLM_KEYS:
        st.session_state[getattr(synced_key, KeySyncNames.STATE)] = selected_chat.get(
            getattr(synced_key, KeySyncNames.LLM),
            getattr(synced_key, KeySyncNames.GET_DEFAULT),
        )

    if selected_chat.get(LLMKeys.IS_INITIALIZED):
        for synced_key in MODEL_SYNCED_LLM_KEYS:
            st.session_state[getattr(synced_key, KeySyncNames.STATE)] = (
                selected_chat.get(
                    getattr(synced_key, KeySyncNames.LLM),
                    getattr(synced_key, KeySyncNames.GET_DEFAULT),
                )
            )

    st.toast("State filled from saved analysis.", icon="🔍")


def on_change_save_state() -> None:
    """Save the state of LLM related widgets to the selected analysis before rerunning the page.

    This can be expanded to other widgets as needed.
    """
    selected_chat: dict = st.session_state[StateKeys.LLM_CHATS].get(
        st.session_state[StateKeys.SELECTED_ANALYSIS], {}
    )

    if not selected_chat:
        return

    for synced_key in WIDGET_SYNCED_LLM_KEYS:
        selected_chat[getattr(synced_key, KeySyncNames.LLM)] = st.session_state.get(
            getattr(synced_key, KeySyncNames.STATE),
            getattr(synced_key, KeySyncNames.GET_DEFAULT),
        )

    if not selected_chat.get(LLMKeys.IS_INITIALIZED):
        for synced_key in MODEL_SYNCED_LLM_KEYS:
            selected_chat[getattr(synced_key, KeySyncNames.LLM)] = st.session_state.get(
                getattr(synced_key, KeySyncNames.STATE),
                getattr(synced_key, KeySyncNames.GET_DEFAULT),
            )


def get_selected_regulated_features(llm_chat: dict) -> tuple[list, dict]:
    selected_features = (
        llm_chat[LLMKeys.SELECTED_FEATURES_UP]
        + llm_chat[LLMKeys.SELECTED_FEATURES_DOWN]
    )
    regulated_features_dict = {
        feature: "up" if feature in llm_chat[LLMKeys.SELECTED_FEATURES_UP] else "down"
        for feature in selected_features
    }
    return selected_features, regulated_features_dict


@st.fragment
def enrichment_analysis(llm_chat: dict, *, disabled: bool = False) -> None:
    new_settings = llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][
        EnrichmentAnalysisKeys.PARAMETERS
    ].copy()
    NO_ENRICHMENT = "<no enrichment>"
    with st.form(key="enrichment_analysis_form"):
        st.markdown(
            "##### Enrichment analysis",
            help="Select the organism and tool for enrichment analysis. Including the background for the anlaysis is recommended.",
        )
        c1, c2, c3 = st.columns((1, 1, 1))
        with c1:
            organism_id = st.selectbox(
                "Organism ID",
                options=list(gprofiler_organisms.keys()),
                index=list(gprofiler_organisms.keys()).index(
                    llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][
                        EnrichmentAnalysisKeys.PARAMETERS
                    ].get(EnrichmentAnalysisKeys.ORGANISM_ID, HUMAN_ORGANISM_ID)
                ),
                format_func=lambda x: f"{gprofiler_organisms[x]} ({x})",
                disabled=disabled,
            )
        with c2:
            options = ["string", "gprofiler", NO_ENRICHMENT]
            display_names = {"string": "STRING", "gprofiler": "GProfiler"}
            enrichment_tool = st.selectbox(
                "External enrichment tool",
                options=options,
                index=options.index(
                    llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][
                        EnrichmentAnalysisKeys.PARAMETERS
                    ].get(EnrichmentAnalysisKeys.TOOL, "gprofiler")
                ),
                format_func=lambda x: display_names.get(x, x),
                disabled=disabled,
                help="Select the tool for enrichment analysis. If you select don't run, make sure to adjust the CoT prompt to remove the enrichment interpretation step, or call enrichment on the fly.",
            )
        with c3:
            include_background = st.checkbox(
                "Use experiment background",
                value=llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][
                    EnrichmentAnalysisKeys.PARAMETERS
                ].get(EnrichmentAnalysisKeys.INCLUDE_BACKGROUND, True),
                disabled=disabled,
                help="Use experiment background proteins in the enrichment analysis. This is recommended for most analyses, as using the whole ontology background can significantly distort results if the experiment is not representative of the whole genome.",
            )
        submit_button = st.form_submit_button(
            "Run enrichment analysis",
            disabled=disabled,
        )
        if submit_button:
            new_settings.update(
                {
                    EnrichmentAnalysisKeys.ORGANISM_ID: organism_id,
                    EnrichmentAnalysisKeys.TOOL: enrichment_tool,
                    EnrichmentAnalysisKeys.INCLUDE_BACKGROUND: include_background,
                }
            )
    old_settings = llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][
        EnrichmentAnalysisKeys.PARAMETERS
    ]
    enrichment_data = llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][
        EnrichmentAnalysisKeys.RESULT
    ]
    if new_settings != old_settings:
        if enrichment_tool == NO_ENRICHMENT:
            enrichment_data = None
        else:
            with st.spinner("Running enrichment analysis..."):
                feature_to_repr_map = st.session_state[
                    StateKeys.DATASET
                ]._feature_to_repr_map
                enrichment_data = get_enrichment_data(
                    difexpressed=list(
                        map(
                            feature_to_repr_map.get,
                            llm_chat[LLMKeys.SELECTED_FEATURES_UP],
                        )
                    )
                    + list(
                        map(
                            feature_to_repr_map.get,
                            llm_chat[LLMKeys.SELECTED_FEATURES_DOWN],
                        )
                    ),
                    organism_id=organism_id,
                    tool=enrichment_tool,
                    include_background=include_background,
                )
                if st.session_state.get(StateKeys.ENRICHMENT_COLUMNS) == [] or any(
                    column not in enrichment_data.columns
                    for column in st.session_state.get(StateKeys.ENRICHMENT_COLUMNS)
                ):
                    st.session_state[StateKeys.ENRICHMENT_COLUMNS] = list(
                        enrichment_data.columns
                    )
                    llm_chat[LLMKeys.ENRICHMENT_COLUMNS] = st.session_state.get(
                        StateKeys.ENRICHMENT_COLUMNS
                    )
        llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][EnrichmentAnalysisKeys.RESULT] = (
            enrichment_data
        )
        llm_chat[LLMKeys.ENRICHMENT_ANALYSIS][EnrichmentAnalysisKeys.PARAMETERS].update(
            new_settings
        )

    if isinstance(enrichment_data, pd.DataFrame):
        st.dataframe(enrichment_data)
        st.multiselect(
            "Select columns to include in the prompt",
            options=enrichment_data.columns,
            key=StateKeys.ENRICHMENT_COLUMNS,
            on_change=on_change_save_state,
            help="Select the columns to include in the initial prompt.",
            disabled=disabled,
        )


@st.fragment
def configure_initial_prompt(
    llm_chat: dict,
    plot_parameters: dict,
    feature_to_repr_map: dict,
    *,
    disabled: bool,
) -> None:
    c1, c2 = st.columns((5, 1))
    with c2:
        st.markdown("#####")
        if st.button(
            "Regenerate experimental design prompt from analysis parameters",
            disabled=disabled,
        ):
            generated_experimental_design_prompt, generated_protein_data_prompt, _ = (
                initialize_initial_prompt_modules(
                    llm_chat, plot_parameters, feature_to_repr_map
                )
            )
            st.session_state[StateKeys.PROMPT_EXPERIMENTAL_DESIGN] = (
                generated_experimental_design_prompt
            )
            on_change_save_state()
    with c1:
        experimental_design_prompt = st.text_area(
            "Please explain your experimental design",
            height=100,
            disabled=disabled,
            key=StateKeys.PROMPT_EXPERIMENTAL_DESIGN,
            on_change=on_change_save_state,
        )
    c1, c2 = st.columns((5, 1))
    with c2:
        st.markdown("#####")
        if st.button(
            "Update prompts with selected features, UniProt information and enrichment analysis",
            disabled=disabled,
            help="Regenerate system message and initial prompt based on current selections",
        ):
            generated_experimental_design_prompt, generated_protein_data_prompt, _ = (
                initialize_initial_prompt_modules(
                    llm_chat, plot_parameters, feature_to_repr_map
                )
            )
            st.session_state[StateKeys.PROMPT_PROTEIN_DATA] = (
                generated_protein_data_prompt
            )
            on_change_save_state()
    with c1:
        protein_data_prompt = st.text_area(
            "Please edit your selection above and update the prompt",
            height=200,
            disabled=disabled,
            key=StateKeys.PROMPT_PROTEIN_DATA,
            on_change=on_change_save_state,
        )
    c1, c2 = st.columns((5, 1))
    with c2:
        st.markdown("#####")
        preset = st.selectbox(
            "Select initial instruction",
            options=LLMInstructionKeys.get_values(),
            index=LLMInstructionKeys.get_values().index(LLMInstructionKeys.CUSTOM),
            disabled=disabled,
        )
        if preset != LLMInstructionKeys.CUSTOM:
            st.session_state[StateKeys.PROMPT_INSTRUCTIONS] = _get_initial_instruction(
                preset
            )
            on_change_save_state()
    with c1:
        initial_instruction = st.text_area(
            "Please provide an initial instruction",
            height=100,
            disabled=disabled,
            key=StateKeys.PROMPT_INSTRUCTIONS,
            on_change=on_change_save_state,
        )
    return get_initial_prompt(
        experimental_design_prompt, protein_data_prompt, initial_instruction
    )


@st.fragment
def show_llm_chat(
    llm_integration: LLMIntegration,
    selected_analysis_key: str,
    show_all: bool = False,
    show_individual_tokens: bool = False,
    show_prompt: bool = True,
) -> None:
    """The chat interface for the LLM interpretation."""

    # TODO dump to file -> static file name, plus button to do so
    # Ideas: save chat as txt, without encoding objects, just put a replacement string.
    # Offer bulk download of zip with all figures (via plotly download as svg.).
    # Alternatively write it all in one pdf report using e.g. pdfrw and reportlab (I have code for that combo).

    # TODO show errors by default
    # e.g. {"result": "Error executing get_uniprot_info_for_search_string: 'st.session_state has no key "selected_uniprot_fields".

    selected_analysis_session_state = st.session_state[StateKeys.LLM_CHATS][
        selected_analysis_key
    ]

    model_name = llm_integration.client_wrapper.model_name

    # no. tokens spent
    messages, total_tokens, pinned_tokens = llm_integration.get_print_view(
        show_all=show_all
    )
    for message in messages:
        with st.chat_message(message[MessageKeys.ROLE]):
            st.markdown(
                f"[{message[MessageKeys.TIMESTAMP]}] {message[MessageKeys.CONTENT]}"
            )
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
                    tokens = LLMIntegration.estimate_tokens([message], model=model_name)
                    token_message += f"*tokens: {str(tokens)}*"
                st.markdown(token_message)
            for artifact in message[MessageKeys.ARTIFACTS]:
                if isinstance(artifact, pd.DataFrame):
                    st.dataframe(artifact, key=str(id(artifact)))
                elif isinstance(
                    artifact, (PlotlyObject, plotly_object)
                ):  # TODO can there be non-plotly types here
                    st.plotly_chart(artifact, key=str(id(artifact)))
                elif not isinstance(artifact, str):
                    st.warning("Don't know how to display artifact:")
                    st.write(artifact)

    st.markdown(
        f"*total tokens used: {str(total_tokens)}, tokens used for pinned messages: {str(pinned_tokens)}*"
    )

    if selected_analysis_session_state.get(LLMKeys.RECENT_CHAT_WARNINGS):
        st.warning("Warnings during last chat completion:")
        for warning in selected_analysis_session_state[LLMKeys.RECENT_CHAT_WARNINGS]:
            st.warning(str(warning.message).replace("\n", "\n\n"))

    if show_prompt and (prompt := st.chat_input("Say something")):
        with st.chat_message(Roles.USER):
            st.markdown(prompt)
            if show_individual_tokens:
                st.markdown(
                    f"*tokens: {str(LLMIntegration.estimate_tokens([{MessageKeys.CONTENT:prompt}], model=model_name))}*"
                )
        with (
            st.spinner("Processing prompt..."),
            warnings.catch_warnings(record=True) as caught_warnings,
        ):
            llm_integration.chat_completion(prompt)
            selected_analysis_session_state[LLMKeys.RECENT_CHAT_WARNINGS] = (
                caught_warnings
            )

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
