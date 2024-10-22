from typing import Dict


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
