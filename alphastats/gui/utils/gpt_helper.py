import copy
from typing import Union, List, Dict

import json

from Bio import Entrez

import pandas as pd
import streamlit as st

from alphastats.plots.DimensionalityReduction import DimensionalityReduction
from alphastats.gui.utils.ui_helper import StateKeys

Entrez.email = "lebedev_mikhail@outlook.com"  # Always provide your email address when using NCBI services.


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
    groups = [str(i) for i in metadata.columns.to_list()]
    group_to_subgroup_values = {
        group: get_unique_values_from_column(group, metadata=metadata)
        for group in groups
    }
    return group_to_subgroup_values


def get_unique_values_from_column(column: str, metadata: pd.DataFrame) -> List[str]:
    """
    Get the unique values from a column in the metadata file.

    Args:
        column (str): The name of the column in the metadata file.
        metadata (pd.DataFrame, optional): The metadata dataframe (which sample has which disease/treatment/condition/etc).

    Returns:
        List[str]: A list of unique values from the column.
    """
    unique_values = metadata[column].unique().tolist()
    return [str(i) for i in unique_values]


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


def get_general_assistant_functions() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_gene_function",
                "description": "Get the gene function and description by UniProt lookup of gene identifier/name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene_name": {
                            "type": "string",
                            "description": "Gene identifier/name for UniProt lookup",
                        },
                    },
                    "required": ["gene_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_enrichment_data",
                "description": "Get enrichment data for a list of differentially expressed genes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "difexpressed": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of differentially expressed gene names to search for",
                        },
                        "organism_id": {
                            "type": "string",
                            "description": "The Uniprot organism ID to search in, e.g. 9606 for human",
                        },
                        "tool": {
                            "type": "string",
                            "description": "The tool to use for enrichment analysis",
                            "enum": ["gprofiler", "string"],
                        },
                    },
                    "required": ["difexpressed", "organism_id"],
                },
            },
        },
    ]


def get_assistant_functions(
    gene_to_prot_id_dict: Dict,
    metadata: pd.DataFrame,
    subgroups_for_each_group: Dict,
) -> List[Dict]:
    """
    Get a list of assistant functions for function calling in the ChatGPT model.
    You can call this function with no arguments, arguments are given for clarity on what changes the behavior of the function.
    For more information on how to format functions for Assistants, see https://platform.openai.com/docs/assistants/tools/function-calling

    Args:
        gene_to_prot_id_dict (dict, optional): A dictionary with gene names as keys and protein IDs as values.
        metadata (pd.DataFrame, optional): The metadata dataframe (which sample has which disease/treatment/condition/etc).
        subgroups_for_each_group (dict, optional): A dictionary with the column names as keys and a list of unique values as values. Defaults to get_subgroups_for_each_group().
    Returns:
        list[dict]: A list of assistant functions.
    """
    # TODO figure out how this relates to the parameter `subgroups_for_each_group`
    subgroups_for_each_group_ = str(
        get_subgroups_for_each_group(st.session_state[StateKeys.DATASET].metadata)
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "plot_intensity",
                "description": "Create an intensity plot based on protein data and analytical methods.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "protein_id": {
                            "type": "string",
                            "enum": [i for i in gene_to_prot_id_dict.keys()],
                            "description": "Identifier for the protein of interest",
                        },
                        "group": {
                            "type": "string",
                            "enum": [str(i) for i in metadata.columns.to_list()],
                            "description": "Column name in the dataset for the group variable",
                        },
                        "subgroups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Specific subgroups within the group to analyze. For each group you need to look up the subgroups in the dict"
                            f" {subgroups_for_each_group_} or present user with them first if you are not sure what to choose",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["violin", "box", "scatter", "all"],
                            "description": "The method of plot to create",
                        },
                        "add_significance": {
                            "type": "boolean",
                            "description": "Whether to add significance markers to the plot",
                        },
                        "log_scale": {
                            "type": "boolean",
                            "description": "Whether to use a logarithmic scale for the plot",
                        },
                    },
                    "required": ["protein_id", "group"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "perform_dimensionality_reduction",
                "description": "Perform dimensionality reduction on a given dataset and generate a plot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group": {
                            "type": "string",
                            "description": "The name of the group column in the dataset",
                            "enum": [str(i) for i in metadata.columns.to_list()],
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pca", "umap", "tsne"],
                            "description": "The dimensionality reduction method to apply",
                        },
                        "circle": {
                            "type": "boolean",
                            "description": "Flag to draw circles around groups in the scatterplot",
                        },
                    },
                    "required": ["group", "method", "circle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plot_sampledistribution",
                "description": "Generates a histogram plot for each sample in the dataset matrix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "The name of the group column in the dataset to color the samples by",
                            "enum": [str(i) for i in metadata.columns.to_list()],
                        },
                        "method": {
                            "type": "string",
                            "enum": ["violin", "box"],
                            "description": "The method of plot to create",
                        },
                    },
                    "required": ["group", "method"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plot_volcano",
                "description": "Generates a volcano plot based on two subgroups of the same group",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "items": {"type": "string"},
                            "description": f"Column name in the dataset for the group variable. Must be from {str(subgroups_for_each_group.keys())} and group1 and group2 must be from THIS very group.",
                        },
                        "group1": {
                            "type": "string",
                            "items": {"type": "string"},
                            "description": f"Specific subgroup within the group to analyze. For each group you get from prompt you need to look up the subgroups in the dict {str(subgroups_for_each_group)} or present user with them first if you are not sure what to choose. You can use ONLY 1.",
                        },
                        "group2": {
                            "type": "string",
                            "items": {"type": "string"},
                            "description": f"Second subgroup from the same group in {str(subgroups_for_each_group)} or present user with them first if you are not sure what to choose. You can use ONLY 1.",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["wald", "ttest", "anova", "welch-ttest", "sam"],
                            "description": "The method of plot to create",
                        },
                        "labels": {
                            "type": "boolean",
                            "description": "Whether to add gene names to the plot",
                        },
                        "min_fc": {
                            "type": "string",
                            "enum": ["0", "1", "2"],
                            "description": "Minimal foldchange cutoff that is considered significant",
                        },
                        "alpha": {
                            "type": "string",
                            "enum": ["0.01", "0.02", "0.03", "0.04", "0.05"],
                            "description": "Alpha value for significance",
                        },
                        "draw_line": {
                            "type": "boolean",
                            "description": "Whether to draw lines for significance",
                        },
                        "perm": {
                            "type": "string",
                            "enum": ["1", "10", "100", "1000"],
                            "description": "Number of permutations for SAM",
                        },
                        "fdr": {
                            "type": "string",
                            "enum": ["0.005", "0.01", "0.05", "0.1"],
                            "description": "False Discovery Rate cutoff for SAM",
                        },
                    },
                    "required": ["column", "group1", "group2"],
                },
            },
        },
        # {"type": "code_interpreter"},
    ]


def perform_dimensionality_reduction(group, method, circle, **kwargs):
    dr = DimensionalityReduction(
        st.session_state[StateKeys.DATASET], group, method, circle, **kwargs
    )
    return dr.plot


def turn_args_to_float(json_string: Union[str, bytes, bytearray]) -> Dict:
    """
    Turn all values in a JSON string to floats if possible.

    Args:
        json_string (Union[str, bytes, bytearray]): The JSON string to convert.

    Returns:
        dict: The converted JSON string as a dictionary.
    """
    data = json.loads(json_string)
    for key, value in data.items():
        if isinstance(value, str):
            try:
                data[key] = float(value)
            except ValueError:
                continue
    return data


def get_gene_to_prot_id_mapping(gene_id: str) -> str:
    """Get protein id from gene id. If gene id is not present, return gene id, as we might already have a gene id.
    'VCL;HEL114' -> 'P18206;A0A024QZN4;V9HWK2;B3KXA2;Q5JQ13;B4DKC9;B4DTM7;A0A096LPE1'
    Args:
        gene_id (str): Gene id

    Returns:
        str: Protein id or gene id if not present in the mapping.
    """
    import streamlit as st

    session_state_copy = dict(copy.deepcopy(st.session_state))
    if StateKeys.GENE_TO_PROT_ID not in session_state_copy:
        session_state_copy[StateKeys.GENE_TO_PROT_ID] = {}
    if gene_id in session_state_copy[StateKeys.GENE_TO_PROT_ID]:
        return session_state_copy[StateKeys.GENE_TO_PROT_ID][gene_id]
    for gene, prot_id in session_state_copy[StateKeys.GENE_TO_PROT_ID].items():
        if gene_id in gene.split(";"):
            return prot_id
    return gene_id
