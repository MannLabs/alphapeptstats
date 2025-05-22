"""Enrichment analysis functions used by the LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd
import requests
from gprofiler import GProfiler

from alphastats.dataset.dataset import DataSet
from alphastats.gui.utils.state_keys import StateKeys

if TYPE_CHECKING:
    from alphastats.dataset.dataset import DataSet

HUMAN_ORGANISM_ID = "9606"

gprofiler_organisms = {
    HUMAN_ORGANISM_ID: "hsapiens",
    "10090": "mmusculus",
    "10116": "rnorvegicus",
    "7227": "dmelanogaster",
    "6239": "celegans",
    "4932": "scerevisiae",
    "3702": "athaliana",
}


def _wrap_exceptions_requests_post(
    api_descriptor: str, url: str, timeout: int, **kwargs
) -> requests.Response:
    """Wrap exceptions for requests.post.

    Parameters
    ----------
    api_descriptor : str
        Short string to describe the API called.
    url : str
        The URL to call.
    timeout : int
        The timeout for the request in seconds.
    **kwargs
        Keyword arguments to pass to requests.post.

    Returns
    -------
    requests.Response
        The response from the API call.

    Raises
    ------
    ValueError
        If the request to the API fails.

    """
    try:
        return requests.post(url=url, timeout=timeout, **kwargs)
    except requests.exceptions.Timeout as e:
        raise ValueError(
            f"Request to {api_descriptor} timed out after {timeout} seconds"
        ) from e
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request to {api_descriptor} failed: {e}") from e


def _get_functional_annotation_stringdb(
    identifiers: list[str],
    background_identifiers: Optional | list[str] = None,
    species_id: str = HUMAN_ORGANISM_ID,
    timeout: int = 600,
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
    timeout : int, optional
        The timeout for the request in seconds. Default is 5 minutes.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the functional annotation data.

    """
    stringdb_id_separator = "%0d"
    params = {
        "identifiers": stringdb_id_separator.join(identifiers),  # your protein list
        "species": species_id,  # NCBI/STRING taxon identifier
        "caller_identity": "alphapeptstats",  # your app name
    }
    if background_identifiers:
        params["background_string_identifiers"] = stringdb_id_separator.join(
            background_identifiers
        )
    url = "https://string-db.org/api/json/enrichment"

    response = _wrap_exceptions_requests_post(
        api_descriptor="STRING API", url=url, data=params, timeout=timeout
    )

    data = response.json()
    enrichment_data = pd.DataFrame(data)
    return enrichment_data.drop(
        [
            "ncbiTaxonId",
            "inputGenes",
        ],
        axis=1,
    )


def _map_short_representation_to_stringdb(
    short_representations: list[str],
    species: str = HUMAN_ORGANISM_ID,
    timeout: int = 600,
) -> list[str]:
    """Map feature representations to STRING identifiers.

    The API returns one line per mapped input identifier, where the second value contains the index in the input and the third value contains the STRING identifier.

    Parameters
    ----------
    short_representations : list of str
        A list of feature representations to map.
    species : str, optional
        The NCBI/STRING taxon identifier for the species. Default is "9606" (human).
    timeout : int, optional
        The timeout for the request in seconds. Default is 5 minutes.

    Returns
    -------
    list of str
        A list of mapped STRING identifiers.

    Raises
    ------
    ValueError
        If the request to the STRING API fails.

    """
    stringdb_api_url = "https://version-12-0.string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"

    params = {
        "identifiers": "\r".join(short_representations),  # your protein list
        "species": species,  # NCBI/STRING taxon identifier
        "limit": 1,  # only one (best) identifier per input protein
        "echo_query": 1,  # see your input identifiers in the output
        "caller_identity": "alphapeptstats",  # your app name
    }

    request_url = f"{stringdb_api_url}/{output_format}/{method}"

    response = _wrap_exceptions_requests_post(
        api_descriptor="STRING API", url=request_url, data=params, timeout=timeout
    )

    results = response.text.strip()
    if not results:
        raise ValueError("No identifiers could be mapped to STRING identifiers.")
    return [line.split("\t")[2] for line in results.split("\n")]


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

    Examples
    --------
    >>> _shorten_representations(["id:P12345;Q12345", "GENE1;GENE2", "GENE3"])
    ["P12345", "GENE1", "GENE3"]

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
    if not query:
        raise ValueError("No query genes provided for enrichment analysis.")
    gp = GProfiler(
        user_agent="AlphaPeptStats",
        return_dataframe=True,
    )
    return gp.profile(query=query, organism=organism, background=background)


def get_enrichment_data(
    difexpressed: list[str],
    organism_id: str = HUMAN_ORGANISM_ID,
    tool: str = "string",
    *,
    include_background: bool = True,
    background: Optional | list[str] = None,
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
    background : list of str, optional
        A list of background genes for the enrichment analysis in case this funciton is run outside the GUI. Default is None.

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

    background_identifiers = (
        None if not include_background else _get_background(background)
    )

    diff_identifiers = _shorten_representations(difexpressed)

    if tool == "gprofiler":
        if (mapped_organism_id := gprofiler_organisms.get(organism_id)) is None:
            raise ValueError(
                f"Organism ID {organism_id} not supported by g:Profiler. Supported IDs are {gprofiler_organisms.keys()}",
            )
        enrichment_data = _get_functional_annotation_gprofiler(
            query=diff_identifiers,
            background=background_identifiers,
            organism=mapped_organism_id,
        )
    elif tool == "string":
        if background_identifiers:
            background_identifiers = _map_short_representation_to_stringdb(
                background_identifiers,
                organism_id,
            )
        diff_identifiers = _map_short_representation_to_stringdb(
            diff_identifiers,
            organism_id,
        )
        enrichment_data = _get_functional_annotation_stringdb(
            identifiers=diff_identifiers,
            background_identifiers=background_identifiers,
            species_id=organism_id,
        )

    return enrichment_data


def _get_dataset() -> DataSet:
    """Get the dataset from the session state.

    Returns
    -------
    DataSet
        The dataset from the session state.

    Raises
    ------
    ValueError
        If no dataset is found in the session state.

    """
    import streamlit as st

    dataset: DataSet = st.session_state.get(StateKeys.DATASET, None)
    if dataset is None:
        raise ValueError("No dataset found in the session state.")
    return dataset


def _get_background(background: list[str]) -> list[str]:
    """Get the background identifiers for enrichment analysis.

    Parameters
    ----------
    background : list of str
        A list of background identifiers or str representation of backgorund features.

    Returns
    -------
    list of str
        A list of background identifiers.

    Raises
    ------
    ValueError
        If no dataset is found in the session state.
    ValueError
        If background is not provided as additional argument when enrichment is not run from the GUI.

    """
    try:
        dataset: DataSet = _get_dataset()
    except Exception as e:
        if not background:
            raise ValueError(
                "Background identifiers must be provided as additional argument if enrichment is not run from the GUI."
            ) from e
        background_identifiers = background
    else:
        background_identifiers = dataset._feature_to_repr_map.values()  # noqa: SLF001
    return _shorten_representations(background_identifiers)
