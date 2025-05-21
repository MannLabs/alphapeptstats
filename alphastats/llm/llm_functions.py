"""Module for defining assistant functions for the ChatGPT model."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from alphastats.dataset.dataset import DataSet
from alphastats.gui.utils.state_keys import StateKeys
from alphastats.llm.enrichment_analysis import get_enrichment_data, gprofiler_organisms
from alphastats.llm.uniprot_utils import (
    ExtractedUniprotFields,
    format_uniprot_annotation,
    get_annotations_for_feature,
)


def get_annotation_from_store_by_feature_list(
    features: list, annotation_store: dict
) -> dict | str | None:
    """
    Retrieve the annotation from the store based on a list of feauture ids.
    The first matching feature found in the annotation store, is returned. Don't use this for disparate features, it is meant to be used for features that have the same representation.
    Args:
        features (list): A list of features to search for.
        annotation_store (dict): A dictionary storing annotations for features.
    Returns:
        Union[dict, str, None]: The annotation corresponding to the feature representation if found, otherwise None.
    """
    for feature in features:
        if feature in annotation_store:
            return annotation_store[feature]
    return None


def get_annotation_from_uniprot_by_feature_list(
    features: list,
) -> tuple[dict | str, str]:
    """
    Retrieve annotation from UniProt based on a list of features.
    It retrieves the annotation for the most appropriate feature from UniProt. Most appropriate is defined as 1. the only feature, 2. any feature if they only differ in isoform ids, 3. the feature with the most base ids, as this is most likely to contain a well annotated protein.
    Args:
        features (list): A list of features to search for.
    Returns:
        Union[dict, str]: The annotation for the identified feature. The return type can be a dictionary or a string depending on the annotation retrieved.
        feature (str): The feature that was used to retrieve the annotation.
    """
    if len(features) == 0:
        raise ValueError("No features provided.")
    elif len(features) == 1:
        feature = features[0]
    else:
        baseid_sets = []
        for feature in features:
            baseid_sets.append(
                set([identifier.split("-")[0] for identifier in feature.split(";")])
            )
        if len(set.union(*baseid_sets)) == len(set.intersection(*baseid_sets)):
            feature = features[0]
        else:
            feature = features[baseid_sets.index(max(baseid_sets, key=len))]
    annotation = get_annotations_for_feature(feature)
    return annotation, feature


def get_uniprot_info_for_search_string(
    search_string: str, fields: Optional | list = None
) -> str:
    """Get the UniProt information from llm input. This can be either a feature representation, a gene identifier or a protein id.
    This is required so the LLM can feed the promt from the list of feature representations it is provided with.

    Args:
        search_string (str): Either a feature representation, feature id, gene symbol or protein identifier.

    Returns:
        str: The formatted UniProt information for the feature."""
    annotation_store = st.session_state[StateKeys.ANNOTATION_STORE]
    dataset: DataSet = st.session_state[StateKeys.DATASET]

    try:
        features = dataset._get_feature_ids_from_search_string(search_string)
    except ValueError:
        features = [feature.strip() for feature in search_string.split(",")]

    annotation = get_annotation_from_store_by_feature_list(features, annotation_store)
    if annotation is None:
        annotation, feature = get_annotation_from_uniprot_by_feature_list(features)
        annotation_store[feature] = annotation

    return (
        search_string
        + ": "
        + format_uniprot_annotation(
            annotation,
            fields=fields if fields is not None else list(annotation.keys()),
        )
    )


GENERAL_FUNCTION_MAPPING = {
    "get_uniprot_info_for_search_string": get_uniprot_info_for_search_string,
    "get_enrichment_data": get_enrichment_data,
}


def get_general_assistant_functions() -> list[dict]:
    """Get a list of general assistant functions (independent of the underlying DataSet) for function calling by the LLM.

    Returns:
        List[Dict]: A list of dictionaries desscribing the assistant functions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": get_uniprot_info_for_search_string.__name__,
                "description": "Get the gene function and description by UniProt lookup of gene identifier or protein id. When picking the representation from a comma separated list, always include the whole item, even if it contains semicolons or other separators. e.g. from `A;B, ids:123, C` submit `A:B` (not only `A` and/or `B`) and `ids:123`, not `123`.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_string": {
                            "type": "string",
                            "description": "Feature representation, gene identifier or protein id",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"A list of UniProt fields to include in the output. If empty, all fields are included. Available fields are {', '.join(ExtractedUniprotFields.get_values())}.",
                        },
                    },
                    "required": ["search_string"],
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
                            "description": "A list of differentially expressed gene names to search for. Use the exact representations supplied in the (initial) prompt.",
                        },
                        "organism_id": {
                            "type": "string",
                            "description": f"The organism ID to search in, use keys from this dictionary: '{gprofiler_organisms}'. If you are unsure which organism to use, ask the user.",
                        },
                        "tool": {
                            "type": "string",
                            "description": "The tool to use for enrichment analysis. String output includes names of input identifiers belonging to each enriched term and can therefore be more useful for downstream analysis. Gprofiler output is more concise.",
                            "enum": ["string", "gprofiler"],
                        },
                        "include_background": {
                            "type": "boolean",
                            "description": "Whether to include background genes. This can significantly alter results and should be turned on when the experiment has significantly lower depth than a the full organism annotation.",
                        },
                    },
                    "required": ["difexpressed", "organism_id", "tool"],
                },
            },
        },
    ]


def get_assistant_functions(
    genes_of_interest: list[str],
    metadata: pd.DataFrame,
    subgroups_for_each_group: dict,
) -> list[dict]:
    """
    Get a list of assistant functions for function calling in the ChatGPT model.
    For more information on how to format functions for Assistants, see https://platform.openai.com/docs/assistants/tools/function-calling

    Args:
        genes_of_interest (list): A list with gene names.
        metadata (pd.DataFrame): The metadata dataframe (which sample has which disease/treatment/condition/etc).
        subgroups_for_each_group (dict): A dictionary with the column names as keys and a list of unique values as values. Defaults to get_subgroups_for_each_group().
    Returns:
        List[Dict]: A list of dictionaries desscribing the assistant functions.
    """
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
                        "feature": {
                            "type": "string",
                            "description": "Identifier for the feature of interest. Use the same format as in the initial prompt, inidividual gene symbols, or individual protein ids.",
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
                    "required": ["feature", "group"],
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
