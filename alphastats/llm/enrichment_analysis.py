from typing import List

import pandas as pd
import requests
from gprofiler import GProfiler

# TODO needed?
# from Bio import Entrez
# Entrez.email = "lebedev_mikhail@outlook.com"  # Always provide your email address when using NCBI services.


def _get_functional_annotation_string(identifier, species_id="9606") -> pd.DataFrame:
    """
    Get functional annotation from STRING for a gene identifier.

    Args:
        identifier (str): A gene identifier.
        species_id (str, optional): The Uniprot organism ID to search in.

    Returns:
        pd.DataFrame: The functional annotation data.
    """
    url = f"https://string-db.org/api/json/enrichment?identifiers={identifier}&species={int(species_id)}&caller_identity=alphapeptstats"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        print(f"Request failed with status code {response.status_code}")
        return None


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
    difexpressed: List[str], organism_id: str = 9606, tool: str = "gprofiler"
) -> pd.DataFrame:
    """
    Get enrichment data for a list of differentially expressed genes.

    Args:
        difexpressed (list[str]): A list of differentially expressed genes.
        organism_id (str, optional): The Uniprot organism ID to search in.
        tool (str, optional): The tool to use for enrichment analysis.

    Returns:
        pd.DataFrame: The enrichment data.
    """
    assert tool in [
        "gprofiler",
        "string",
    ], "Tool must be either 'gprofiler' or 'string'"
    if tool == "gprofiler":
        enrichment_data = _get_functional_annotation_gprofiler(difexpressed)
    else:
        enrichment_data = _get_functional_annotation_string(
            "%0d".join(difexpressed), organism_id
        )

    return enrichment_data
