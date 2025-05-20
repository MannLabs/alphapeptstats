"""Utility functions for the LLM module."""

from typing import Dict

import pandas as pd

from alphastats.dataset.keys import Cols


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
    groups = [
        str(group)
        for group in metadata.columns.to_list()
        if metadata[group].dtype == "object"
    ]
    group_to_subgroup_values = {
        group: [str(subgroup) for subgroup in metadata[group].unique().tolist()]
        for group in groups
    }
    group_to_subgroup_values[Cols.SAMPLE] = (
        f"{len(group_to_subgroup_values[Cols.SAMPLE])} samples"
    )
    return group_to_subgroup_values
