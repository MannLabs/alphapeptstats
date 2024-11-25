from pathlib import Path
from typing import List, Optional

import streamlit as st

from alphastats.gui.utils.ui_helper import StateKeys
from alphastats.llm.llm_integration import LLMIntegration
from alphastats.llm.uniprot_utils import format_uniprot_annotation


# TODO: pass the annotation store and the feature repr map as arguments
def get_display_proteins_html(protein_ids: List[str], is_upregulated: True) -> str:
    """
    Get HTML code for displaying a list of proteins, color according to expression.

    Args:
        protein_ids (list[str]): a list of proteins.
        is_upregulated (bool): whether the proteins are up- or down-regulated.
    """

    uniprot_url = "https://www.uniprot.org/uniprotkb/"

    color = "green" if is_upregulated else "red"
    protein_ids_html = "".join(
        f'<a href = {uniprot_url + st.session_state[StateKeys.ANNOTATION_STORE][protein].get("primaryAccession",protein)}><li style="color: {color};">{st.session_state[StateKeys.DATASET]._feature_to_repr_map[protein]}</li></a>'
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


def get_display_available_uniprot_info(regulated_features: list) -> dict:
    """
    Retrieves and formats UniProt information for a list of regulated features.

    Note: The information is retrieved from the `annotation_store` in the `session_state`, which is filled when the LLM analysis is set up from the anlaysis page.

    Args:
        regulated_features (list): A list of features for which UniProt information is to be retrieved.
    Returns:
        dict: A dictionary where each key is a feature representation and the value is another dictionary
              containing the 'protein ids' and 'generated text' with formatted UniProt information or an error message.
    """
    text_repr = {}
    for feature in regulated_features:
        try:
            text = format_uniprot_annotation(
                st.session_state[StateKeys.ANNOTATION_STORE][feature]
            )
        except Exception as e:
            text = e
            # TODO: make downstream filtering of faulty information possible.
        text_repr[st.session_state[StateKeys.DATASET]._feature_to_repr_map[feature]] = {
            "protein ids": feature,
            "generated text": text,
        }
    return text_repr
