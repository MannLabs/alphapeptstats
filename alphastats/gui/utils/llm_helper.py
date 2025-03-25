import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from alphastats.gui.utils.analysis import NewAnalysisOptions
from alphastats.gui.utils.state_keys import DefaultStates, StateKeys
from alphastats.llm.llm_integration import LLMIntegration, MessageKeys, Models
from alphastats.llm.uniprot_utils import (
    ExtractedUniprotFields,
    format_uniprot_annotation,
)

LLM_ENABLED_ANALYSIS = [NewAnalysisOptions.DIFFERENTIAL_EXPRESSION_TWO_GROUPS]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@st.fragment
def llm_config():
    """Show the configuration options for the LLM analysis."""
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


@st.fragment
def protein_selector(df: pd.DataFrame, title: str, state_key: str) -> List[str]:
    """Creates a data editor for protein selection and returns the selected proteins.

    Args:
        df: DataFrame containing protein data with 'Gene', 'Selected', 'Protein' columns
        title: Title to display above the editor

    Returns:
        selected_proteins (List[str]): A list of selected proteins.
    """
    st.write(title)
    if len(df) == 0:
        st.markdown("No significant proteins.")
        return []
    c1, c2 = st.columns([1, 1])
    if c1.button("Select all", help=f"Select all {title} for analysis"):
        st.session_state[state_key] = df["Protein"].tolist()
        st.rerun()
    if c2.button("Select none", help=f"Select no {title} for analysis"):
        st.session_state[state_key] = []
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
    )
    # Extract the selected genes
    return edited_df.loc[edited_df["Selected"], "Protein"].tolist()


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

    Note: The information is retrieved from the `annotation_store` in the `session_state`, which is filled when the LLM analysis is set up from the anlaysis page.

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
    *,
    disabled=False,
):
    """Display the interface for selecting fields from UniProt information, including a preview of the selected fields."""
    all_fields = ExtractedUniprotFields.get_values()
    st.markdown(
        "We reccomend to provide at least limited information from Uniprot for all proteins as part of the initial prompt to avoid misinterpretaiton of gene names or ids by the LLM. You can edit the selection of fields to include while chatting for on the fly demand for more information."
    )
    c1, c2, c3, c4, c5, c6 = st.columns((1, 1, 1, 1, 1, 1))
    if c1.button("Select all"):
        st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = all_fields
        st.rerun(scope="fragment")
    if c2.button("Select none"):
        st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = []
        st.rerun(scope="fragment")
    if c3.button("Recommended selection"):
        st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = (
            DefaultStates.SELECTED_UNIPROT_FIELDS.copy()
        )
        st.rerun(scope="fragment")
    with c4:
        texts = [
            format_uniprot_annotation(
                st.session_state[StateKeys.ANNOTATION_STORE].get(feature, {}),
                fields=st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS],
            )
            for feature in regulated_genes_dict
        ]
        tokens = LLMIntegration.estimate_tokens(
            [{MessageKeys.CONTENT: text} for text in texts], model=model_name
        )
        st.markdown(f"Total tokens: {tokens:.0f}")
    with c5:
        st.checkbox(
            "Integrate into initial prompt",
            help="If this is ticked and the initial prompt is updated, the Uniprot information will be included in the prompt and the instructions regarding uniprot will change to onl;y look up more information if explicitly asked to do so. Make sure that the total tokens are below the message limit of your LLM.",
            key=StateKeys.INTEGRATE_UNIPROT,
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
                value=field in st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS],
            ):
                selected_fields.append(field)
        if set(selected_fields) != set(
            st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS]
        ):
            st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = selected_fields
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
                    fields=st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS],
                )
            )
