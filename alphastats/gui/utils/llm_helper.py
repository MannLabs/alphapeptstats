from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from alphastats.gui.utils.ui_helper import StateKeys


def get_subgroups_for_each_group(
    metadata: pd.DataFrame,
) -> Dict:
    """
    Get the unique values for each column in the metadata file.

    Args:
        metadata (pd.DataFrame, optional): The metadata dataframe (which sample has which disease/treatment/condition/etc).

    Returns:
        dict: A dictionary with the column names as keys and a list of unique values as values.
    """
    groups = [str(group) for group in metadata.columns.to_list()]
    group_to_subgroup_values = {
        group: [str(subgroup) for subgroup in metadata[group].unique().tolist()]
        for group in groups
    }
    return group_to_subgroup_values


def display_proteins(overexpressed: List[str], underexpressed: List[str]) -> None:
    """
    Display a list of overexpressed and underexpressed proteins in a Streamlit app.

    Args:
        overexpressed (list[str]): A list of overexpressed proteins.
        underexpressed (list[str]): A list of underexpressed proteins.
    """

    # Start with the overexpressed proteins
    link = "https://www.uniprot.org/uniprotkb?query="
    overexpressed_html = "".join(
        f'<a href = {link + protein}><li style="color: green;">{protein}</li></a>'
        for protein in overexpressed
    )
    # Continue with the underexpressed proteins
    underexpressed_html = "".join(
        f'<a href = {link + protein}><li style="color: red;">{protein}</li></a>'
        for protein in underexpressed
    )

    # Combine both lists into one HTML string
    full_html = f"<ul>{overexpressed_html}{underexpressed_html}</ul>"

    # Display in Streamlit
    st.markdown(full_html, unsafe_allow_html=True)


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
                st.info("OpenAI API key loaded from secrets.toml.")
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
