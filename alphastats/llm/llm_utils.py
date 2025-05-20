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
    groups = [str(group) for group in metadata.columns.to_list()]
    group_to_subgroup_values = {}
    for group in groups:
        if group == Cols.SAMPLE:
            group_to_subgroup_values[group] = f"{len(metadata[group].unique())} samples"
        elif metadata[group].dtype == "object":
            group_to_subgroup_values[group] = metadata[group].unique().tolist()
        else:
            values = metadata[group].unique().tolist()
            if len(values) > 10:
                group_to_subgroup_values[group] = (
                    f"{len(values)} unique values ranging from {min(values)} to {max(values)}"
                )
            else:
                group_to_subgroup_values[group] = [str(subgroup) for subgroup in values]
    return group_to_subgroup_values
