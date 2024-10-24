"""Utility functions for the LLM module."""

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
