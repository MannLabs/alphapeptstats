from typing import List

import pandas as pd
import requests
import streamlit as st
from gprofiler import GProfiler

from alphastats.dataset.dataset import DataSet
from alphastats.gui.utils.state_utils import StateKeys


def _get_functional_annotation_string(
    identifiers: list, background_identifiers: list = None, species_id: str = "9606"
) -> pd.DataFrame:
    """
    Get functional annotation from STRING for a gene identifier.

    Args:
        identifiers (list): A list of String gene identifiers.
        background_identifiers (list, optional): A list of background gene identifiers.
        species_id (str, optional): The Uniprot organism ID to search in.

    Returns:
        pd.DataFrame: The functional annotation data.
    """
    params = {
        "identifiers": "%0d".join(identifiers),  # your protein list
        "species": species_id,  # NCBI/STRING taxon identifier
        "caller_identity": "alphapeptstats",  # your app name
    }
    if background_identifiers:
        params["background_string_identifiers"] = "%0d".join(background_identifiers)
    url = "https://string-db.org/api/json/enrichment"
    response = requests.post(url, data=params)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError(f"Request failed with status code {response.status_code}")


def _map_representation_to_string(
    representations: List[str], species: str = "9606"
) -> List[str]:
    """
    Map feature representations to STRING identifiers.

    Modified from https://string-db.org/cgi/help.pl?subpage=api%23mapping-identifiers.

    Args:
        representations (list[str]): A list of feature representations.
        species (str, optional): The species to map to.

    Returns:
        list[str]: The mapped STRING identifiers.
    """

    identifiers = [
        input_repr.split(";")[0].split(":")[-1] for input_repr in representations
    ]

    string_api_url = "https://version-12-0.string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"

    params = {
        "identifiers": "\r".join(identifiers),  # your protein list
        "species": species,  # NCBI/STRING taxon identifier
        "limit": 1,  # only one (best) identifier per input protein
        "echo_query": 1,  # see your input identifiers in the output
        "caller_identity": "alphapeptstats",  # your app name
    }

    request_url = "/".join([string_api_url, output_format, method])

    response = requests.post(request_url, data=params)
    if response.status_code == 200:
        results = response.text.strip()
        string_identifiers = []
        for line in results.split("\n"):
            string_identifiers.append(line.split("\t")[2])
        return string_identifiers
    else:
        raise ValueError(
            f"Request to map string identifiers failed with status code {response.status_code}"
        )


def _get_functional_annotation_gprofiler(identifiers: List[str]) -> pd.DataFrame:
    """
    Get functional annotation from g:Profiler for a list of gene identifiers.

    Args:
        identifiers (list[str]): A list of gene identifiers.

    Returns:
        pd.DataFrame: The functional annotation data.
    """
    gp = GProfiler(
        user_agent="AlphaPeptStats",
        return_dataframe=True,
    )
    df = gp.profile(query=identifiers)
    return df


def get_enrichment_data(
    difexpressed: List[str],
    organism_id: str = "9606",
    tool: str = "gprofiler",
    include_background: bool = False,
) -> pd.DataFrame:
    """
    Get enrichment data for a list of differentially expressed genes.

    Args:
        difexpressed (list[str]): A list of differentially expressed genes.
        organism_id (str, optional): The Uniprot organism ID to search in.
        tool (str, optional): The tool to use for enrichment analysis.
        include_background (bool, optional): Whether to include background genes.

    Returns:
        pd.DataFrame: The enrichment data.
    """
    assert tool in [
        "gprofiler",
        "string",
    ], "Tool must be either 'gprofiler' or 'string'"
    if tool == "gprofiler":
        enrichment_data = _get_functional_annotation_gprofiler(difexpressed)
    if tool == "string":
        if include_background:
            dataset: DataSet = st.session_state.get(StateKeys.DATASET)
            background_identifiers = _map_representation_to_string(
                dataset._feature_to_repr_map.values(), organism_id
            )
        else:
            background_identifiers = None
        identifiers = _map_representation_to_string(difexpressed, organism_id)
        enrichment_data = _get_functional_annotation_string(
            identifiers=identifiers,
            background_identifiers=background_identifiers,
            species_id=organism_id,
        )
    else:
        enrichment_data = _get_functional_annotation_string(
            "%0d".join(difexpressed), organism_id
        )

    return enrichment_data
