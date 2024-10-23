"""Module for defining assistant functions for the ChatGPT model."""

from typing import Dict, List

import pandas as pd
from DataSet import DataSet

from alphastats.llm.enrichment_analysis import get_enrichment_data
from alphastats.llm.uniprot_utils import get_gene_function

GENERAL_FUNCTION_MAPPING = {
    "get_gene_function": get_gene_function,
    "get_enrichment_data": get_enrichment_data,
}


def get_general_assistant_functions() -> List[Dict]:
    """Get a list of general assistant functions (independent of the underlying DataSet) for function calling by the LLM.

    Returns:
        List[Dict]: A list of dictionaries desscribing the assistant functions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": get_gene_function.__name__,
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
                "name": get_enrichment_data.__name__,
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
    gene_to_prot_id_map: Dict,
    metadata: pd.DataFrame,
    subgroups_for_each_group: Dict,
) -> List[Dict]:
    """
    Get a list of assistant functions for function calling in the ChatGPT model.
    For more information on how to format functions for Assistants, see https://platform.openai.com/docs/assistants/tools/function-calling

    Args:
        gene_to_prot_id_map (dict): A dictionary with gene names as keys and protein IDs as values.
        metadata (pd.DataFrame): The metadata dataframe (which sample has which disease/treatment/condition/etc).
        subgroups_for_each_group (dict): A dictionary with the column names as keys and a list of unique values as values. Defaults to get_subgroups_for_each_group().
    Returns:
        List[Dict]: A list of dictionaries desscribing the assistant functions.
    """
    gene_names = list(gene_to_prot_id_map.keys())
    groups = [str(col) for col in metadata.columns.to_list()]
    return [
        {
            "type": "function",
            "function": {
                "name": DataSet.plot_intensity.__name__,
                "description": "Create an intensity plot based on protein data and analytical methods.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "protein_id": {  # LLM will provide gene_name, mapping to protein_id is done when calling the function
                            "type": "string",
                            "enum": gene_names,
                            "description": "Identifier for the gene of interest",
                        },
                        "group": {
                            "type": "string",
                            "enum": groups,
                            "description": "Column name in the dataset for the group variable",
                        },
                        "subgroups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Specific subgroups within the group to analyze. For each group you need to look up the subgroups in the dict"
                            f" {subgroups_for_each_group} or present user with them first if you are not sure what to choose",
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
                "name": DataSet.perform_dimensionality_reduction.__name__,
                "description": "Perform dimensionality reduction on a given dataset and generate a plot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group": {
                            "type": "string",
                            "description": "The name of the group column in the dataset",
                            "enum": groups,
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
                "name": DataSet.plot_sampledistribution.__name__,
                "description": "Generates a histogram plot for each sample in the dataset matrix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "The name of the group column in the dataset to color the samples by",
                            "enum": groups,
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
                "name": DataSet.plot_volcano.__name__,
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
