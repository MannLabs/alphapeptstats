"""Enrichment analysis functions used by the LLM."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests
import streamlit as st
from gprofiler import GProfiler

from alphastats.dataset.dataset import DataSet
from alphastats.gui.utils.state_utils import StateKeys


def _get_functional_annotation_string(
    identifiers: list[str],
    background_identifiers: Optional | list[str] = None,
    species_id: str = "9606",
) -> pd.DataFrame:
    """Get functional annotation from STRING for a list of gene identifiers.

    Parameters
    ----------
    identifiers : list
        A list of STRING gene identifiers.
    background_identifiers : list, optional
        A list of background gene identifiers for enrichment analysis. Default is None.
    species_id : str, optional
        The NCBI/STRING taxon identifier for the species. Default is "9606" (human).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the functional annotation data.

    Raises
    ------
    ValueError
        If the request to the STRING API fails.

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
        enrichment_data = pd.DataFrame(data)
        enrichment_data = enrichment_data.drop(
            [
                "ncbiTaxonId",
                "inputGenes",
            ],
            axis=1,
        )
        return enrichment_data
    raise ValueError(f"Request failed with status code {response.status_code}")


def _map_short_representation_to_string(
    short_representations: list[str],
    species: str = "9606",
) -> list[str]:
    """Map feature representations to STRING identifiers.

    Parameters
    ----------
    short_representations : list of str
        A list of feature representations to map.
    species : str, optional
        The NCBI/STRING taxon identifier for the species. Default is "9606" (human).

    Returns
    -------
    list of str
        A list of mapped STRING identifiers.

    Raises
    ------
    ValueError
        If the request to the STRING API fails.

    """
    string_api_url = "https://version-12-0.string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"

    params = {
        "identifiers": "\r".join(short_representations),  # your protein list
        "species": species,  # NCBI/STRING taxon identifier
        "limit": 1,  # only one (best) identifier per input protein
        "echo_query": 1,  # see your input identifiers in the output
        "caller_identity": "alphapeptstats",  # your app name
    }

    request_url = f"{string_api_url}/{output_format}/{method}"

    response = requests.post(request_url, data=params)
    if response.status_code == 200:
        results = response.text.strip()
        return [line.split("\t")[2] for line in results.split("\n")]
    raise ValueError(
        f"Request to map string identifiers failed with status code {response.status_code}",
    )


def _shorten_representations(representations: list[str], sep: str = ";") -> list[str]:
    """Shorten feature representations by extracting the first part of each representation.

    Parameters
    ----------
    representations : list of str
        A list of feature representations to shorten.
    sep : str
        The separator used to split the representations. Default is ";".

    Returns
    -------
    list of str
        A list of shortened feature representations.

    """
    return [input_repr.split(sep)[0].split(":")[-1] for input_repr in representations]


def _get_functional_annotation_gprofiler(
    query: list[str],
    background: Optional | list[str] = None,
    organism: str = "hsapiens",
) -> pd.DataFrame:
    """Get functional annotation from g:Profiler for a list of gene identifiers.

    Parameters
    ----------
    query : list of str
        A list of gene identifiers to query.
    background : list, optional
        A list of background gene identifiers for enrichment analysis. Default is None.
    organism : str, optional
        The organism to search in. Default is "hsapiens" (human).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the functional annotation data.

    Raises
    ------
    Warning
        If the organism is not in the predefined list of organisms supported by g:Profiler.

    """
    if organism not in gprofiler_organisms.values():
        raise Warning(
            f"Organism {organism} not necessarily supported by g:Profiler. Supported organisms are {gprofiler_organisms.values()}",
        )
    gp = GProfiler(
        user_agent="AlphaPeptStats",
        return_dataframe=True,
    )
    return gp.profile(query=query, organism=organism, background=background)


gprofiler_organisms = {
    "9606": "hsapiens",
    "10090": "mmusculus",
    "10116": "rnorvegicus",
    "7227": "dmelanogaster",
    "6239": "celegans",
    "4932": "scerevisiae",
    "3702": "athaliana",
}


def get_enrichment_data(
    difexpressed: list[str],
    organism_id: str = "9606",
    tool: str = "string",
    *,
    include_background: bool = True,
) -> pd.DataFrame:
    """Get enrichment data for a list of differentially expressed genes.

    The tool shortens the gene representations and maps them to STRING identifiers if necessary.
    The background is fetched from the dataset if include_background is True.

    Parameters
    ----------
    difexpressed : list of str
        A list of differentially expressed genes.
    organism_id : str, optional
        The NCBI/STRING taxon identifier for the organism. Default is "9606" (human).
    tool : str, optional
        The tool to use for enrichment analysis. Must be either "gprofiler" or "string". Default is "gprofiler".
    include_background : bool, optional
        Whether to include background genes in the analysis. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the enrichment data.

    Raises
    ------
    ValueError
        If the tool is not "gprofiler" or "string".
    ValueError
        If the organism ID is not supported by the selected tool.

    """
    if tool not in [
        "gprofiler",
        "string",
    ]:
        raise ValueError(
            f"Tool {tool} not supported. Must be either 'gprofiler' or 'string'."
        )

    # Get single id for each feature
    if include_background:
        dataset: DataSet = st.session_state.get(StateKeys.DATASET)
        background_identifiers = _shorten_representations(
            dataset._feature_to_repr_map.values(),
        )
    else:
        background_identifiers = None
    diff_identifiers = _shorten_representations(difexpressed)

    # Call tool
    if tool == "gprofiler":
        if organism_id in gprofiler_organisms:
            organism_id = gprofiler_organisms[organism_id]
        else:
            raise ValueError(
                f"Organism ID {organism_id} not supported by g:Profiler. Supported IDs are {gprofiler_organisms.keys()}",
            )
        enrichment_data = _get_functional_annotation_gprofiler(
            query=diff_identifiers,
            background=background_identifiers,
            organism=organism_id,
        )
    elif tool == "string":
        if background_identifiers:
            background_identifiers = _map_short_representation_to_string(
                background_identifiers,
                organism_id,
            )
        diff_identifiers = _map_short_representation_to_string(
            diff_identifiers,
            organism_id,
        )
        enrichment_data = _get_functional_annotation_string(
            identifiers=diff_identifiers,
            background_identifiers=background_identifiers,
            species_id=organism_id,
        )

    return enrichment_data
