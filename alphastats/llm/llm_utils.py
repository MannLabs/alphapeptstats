from typing import Dict

import pandas as pd


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


def get_protein_id_for_gene_name(
    gene_name: str, gene_to_prot_id_map: Dict[str, str]
) -> str:
    """Get protein id from gene id. If gene id is not present, return gene id, as we might already have a gene id.
    'VCL;HEL114' -> 'P18206;A0A024QZN4;V9HWK2;B3KXA2;Q5JQ13;B4DKC9;B4DTM7;A0A096LPE1'

    Args:
        gene_name (str): Gene id

    Returns:
        str: Protein id or gene id if not present in the mapping.
    """
    if gene_name in gene_to_prot_id_map:
        return gene_to_prot_id_map[gene_name]

    for gene, protein_id in gene_to_prot_id_map.items():
        if gene_name in gene.split(";"):
            return protein_id

    return gene_name
