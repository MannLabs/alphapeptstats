import os
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from dataset.plotting import plotly_object
from plots.plot_utils import PlotlyObject

from alphastats.gui.utils.analysis import NewAnalysisOptions
from alphastats.gui.utils.state_keys import DefaultStates, LLMKeys, StateKeys
from alphastats.llm.llm_integration import LLMIntegration, MessageKeys, Models, Roles
from alphastats.llm.uniprot_utils import (
    ExtractedUniprotFields,
    format_uniprot_annotation,
    get_uniprot_state_key,
)

LLM_ENABLED_ANALYSIS = [NewAnalysisOptions.DIFFERENTIAL_EXPRESSION_TWO_GROUPS]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@st.fragment
def llm_config():
    """Show the configuration options for the LLM interpretation."""
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

        st.number_input(
            "Maximal number of tokens",
            value=st.session_state[StateKeys.MAX_TOKENS],
            min_value=2000,
            max_value=128000,  # TODO: set this automatically based on the selected model
            key=StateKeys.MAX_TOKENS,
        )

        if current_model != st.session_state[StateKeys.MODEL_NAME]:
            st.rerun(scope="app")


def pretty_print_analysis(key: str) -> str:
    """Pretty print an analysis referenced by `key`."""
    analysis = st.session_state[StateKeys.SAVED_ANALYSES][key]
    return (
        f"[{key}] #{analysis['number']} {analysis['method']} {analysis['parameters']}"
    )


def init_llm_chat_state(
    selected_llm_chat: dict, upregulated_genes: list, downregulated_genes: list
) -> None:
    """Initialize the state for a given llm_chat."""
    if LLMKeys.RECENT_CHAT_WARNINGS not in selected_llm_chat:
        selected_llm_chat[LLMKeys.RECENT_CHAT_WARNINGS] = []

    if selected_llm_chat.get(LLMKeys.SELECTED_UNIPROT_FIELDS) is None:
        selected_llm_chat[LLMKeys.SELECTED_UNIPROT_FIELDS] = (
            DefaultStates.SELECTED_UNIPROT_FIELDS.copy()
        )

    if selected_llm_chat.get(LLMKeys.SELECTED_GENES_UP) is None:
        selected_llm_chat[LLMKeys.SELECTED_GENES_UP] = upregulated_genes
    if selected_llm_chat.get(LLMKeys.SELECTED_GENES_DOWN) is None:
        selected_llm_chat[LLMKeys.SELECTED_GENES_DOWN] = downregulated_genes


@st.fragment
def protein_selector(
    df: pd.DataFrame, title: str, selected_analysis_key: str, state_key: str
) -> None:
    """Creates a data editor for protein selection and returns the selected proteins.

    Args:
        df: DataFrame containing protein data with 'Gene', 'Selected', 'Protein' columns
        title: Title to display above the editor
        selected_analysis_key: Key to access the selected analysis in the session state
        state_key: Key to access the selected proteins in the selected analysis

    Returns:
        selected_proteins (List[str]): A list of selected proteins.
    """
    st.write(title)
    if len(df) == 0:
        st.markdown("No significant proteins.")
        return []
    c1, c2 = st.columns([1, 1])
    selected_analysis_session_state = st.session_state[StateKeys.LLM_CHATS][
        selected_analysis_key
    ]

    if c1.button("Select all", help=f"Select all {title} for analysis"):
        selected_analysis_session_state[state_key] = df["Protein"].tolist()
        st.rerun()
    if c2.button("Select none", help=f"Select no {title} for analysis"):
        selected_analysis_session_state[state_key] = []
        st.rerun()

    edited_df = st.data_editor(
        df,
        column_config={
            "Selected": st.column_config.CheckboxColumn(
                "Include?",
                help="Check to include this gene in analysis",
                default=True,
            ),
            "Gene": st.column_config.TextColumn(
                "Gene",
                help="The gene name to be included in the analysis",
                width="medium",
            ),
        },
        disabled=["Gene"],
        hide_index=True,
        # explicitly setting key: otherwise it's calculated from the data which causes problems if two analysis are exactly mirrored
        key=f"{state_key}_data_editor",
    )
    # Extract the selected genes
    selected_analysis_session_state[state_key] = edited_df.loc[
        edited_df["Selected"], "Protein"
    ].tolist()


def get_df_for_protein_selector(
    proteins: List[str], selected: List[str]
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
    protein_ids: List[str], is_upregulated: True, annotation_store, feature_to_repr_map
) -> str:
    """
    Get HTML code for displaying a list of proteins, color according to expression.

    Args:
        protein_ids (list[str]): a list of proteins.
        is_upregulated (bool): whether the proteins are up- or down-regulated.
    """

    uniprot_url = "https://www.uniprot.org/uniprotkb/"

    color = "green" if is_upregulated else "red"
    protein_ids_html = "".join(
        f'<a href = {uniprot_url + annotation_store[protein].get("primaryAccession",protein)}><li style="color: {color};">{feature_to_repr_map[protein]}</li></a>'
        for protein in protein_ids
    )

    return f"<ul>{protein_ids_html}</ul>"


def set_api_key(api_key: str = None) -> None:
    """Put the OpenAI API key in the session state.

    If provided, use the `api_key`.
    If not, take the key from the secrets.toml file.
    Show a message if the file is not found.

    Args:
        api_key (str, optional): The OpenAI API key. Defaults to None.
    """
    if not api_key:
        api_key = st.session_state.get(StateKeys.OPENAI_API_KEY, None)

    if api_key:
        st.info(
            f"OpenAI API key set: {api_key[:3]}{(len(api_key)-6)*'*'}{api_key[-3:]}"
        )
    else:
        try:
            if Path("./.streamlit/secrets.toml").exists():
                api_key = st.secrets["openai_api_key"]
                st.toast("OpenAI API key loaded from secrets.toml.", icon="✅")
            else:
                st.info(
                    "Please enter an OpenAI key or provide it in a secrets.toml file in the "
                    "alphastats/gui/.streamlit directory like "
                    "`openai_api_key = <key>`"
                )
        except KeyError:
            st.error("OpenAI API key not found in secrets.toml .")
        except Exception as e:
            st.error(f"Error loading OpenAI API key: {e}.")

    st.session_state[StateKeys.OPENAI_API_KEY] = api_key


def llm_connection_test(
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Test the connection to the LLM API, return None in case of success, error message otherwise."""
    try:
        llm = LLMIntegration(
            model_name, base_url=base_url, api_key=api_key, load_tools=False
        )
        llm.chat_completion(
            "This is a test. Simply respond 'yes' if you got this message."
        )
        return None

    except Exception as e:
        return str(e)


# Unused now, but could be useful in the future
# TODO: Remove this by end of year if still unused.
def get_display_available_uniprot_info(regulated_features: list) -> dict:
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
    regulated_genes_dict,
    feature_to_repr_map,
    model_name: str,
    selected_analysis_key: str,
    *,
    disabled=False,
):
    """Display the interface for selecting fields from UniProt information, including a preview of the selected fields."""
    all_fields = ExtractedUniprotFields.get_values()
    st.markdown(
        "We recommend to provide at least limited information from Uniprot for all proteins as part of the initial prompt to avoid misinterpretation of gene names or ids by the LLM. You can edit the selection of fields to include while chatting for on the fly demand for more information."
    )
    c1, c2, c3, c4, c5, c6 = st.columns((1, 1, 1, 1, 1, 1))
    selected_analysis_session_state = st.session_state[StateKeys.LLM_CHATS][
        selected_analysis_key
    ]
    if c1.button("Select all"):
        selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS] = all_fields
        st.rerun(scope="fragment")
    if c2.button("Select none"):
        selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS] = []
        st.rerun(scope="fragment")
    if c3.button("Recommended selection"):
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
            for feature in regulated_genes_dict
        ]
        tokens = LLMIntegration.estimate_tokens(
            [{MessageKeys.CONTENT: text} for text in texts], model=model_name
        )
        st.markdown(f"Total tokens: {tokens:.0f}")
    with c5:
        # this is required to persist the state of the "Integrate into initial prompt" checkbox for different analyses
        if (
            st.session_state.get(
                session_state_key := get_uniprot_state_key(selected_analysis_key)
            )
            is None
        ):
            st.session_state[session_state_key] = False

        st.checkbox(
            "Integrate into initial prompt",
            help="If this is ticked and the initial prompt is updated, the Uniprot information will be included in the prompt and the instructions regarding uniprot will change to onl;y look up more information if explicitly asked to do so. Make sure that the total tokens are below the message limit of your LLM.",
            key=get_uniprot_state_key(selected_analysis_key),
            value=st.session_state[
                get_uniprot_state_key(selected_analysis_key)
            ],  # st.session_state.get(get_uniprot_state_key(selected_analysis_key), False),
            disabled=disabled,
        )
    if c6.button("Update prompt", disabled=disabled):
        st.rerun(scope="app")
    c1, c2 = st.columns((1, 3))
    with c1, st.expander("Show options", expanded=True):
        selected_fields = []
        for field in all_fields:
            if st.checkbox(
                field,
                value=field
                in selected_analysis_session_state[LLMKeys.SELECTED_UNIPROT_FIELDS],
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
                for feature in regulated_genes_dict
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


@st.fragment
def show_llm_chat(
    llm_integration: LLMIntegration,
    selected_analysis_key: str,
    show_all: bool = False,
    show_individual_tokens: bool = False,
    show_prompt: bool = True,
):
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
    model_name = llm_integration._model
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

    if selected_analysis_session_state.get(LLMKeys.RECENT_CHAT_WARNINGS):
        st.warning("Warnings during last chat completion:")
        for warning in selected_analysis_session_state[LLMKeys.RECENT_CHAT_WARNINGS]:
            st.warning(warning)  # str(warning.message).replace("\n", "\n\n"))

    if show_prompt and (prompt := st.chat_input("Say something")):
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
            selected_analysis_session_state[LLMKeys.RECENT_CHAT_WARNINGS] = (
                caught_warnings
            )

        selected_analysis_session_state[LLMKeys.RECENT_CHAT_WARNINGS].append(
            "some warning"
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
