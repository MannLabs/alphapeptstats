import numpy as np
import pandas as pd

from alphastats.dataset.keys import Cols


def calculate_foldchange(
    mat_transpose: pd.DataFrame,
    group1_samples: list,
    group2_samples: list,
    is_log2_transformed: bool,
):
    group1_values = mat_transpose[group1_samples].T.mean().values
    group2_values = mat_transpose[group2_samples].T.mean().values
    if is_log2_transformed:
        fc = group1_values - group2_values

    else:
        fc = group1_values / group2_values
        fc = np.log2(fc)

    return fc


def add_metadata_column(metadata: pd.DataFrame, group1_list: list, group2_list: list):
    # create new column in metadata with defined groups

    sample_names = metadata[Cols.SAMPLE].to_list()
    misc_samples = list(set(group1_list + group2_list) - set(sample_names))
    if len(misc_samples) > 0:
        raise ValueError(f"Sample names: {misc_samples} are not described in Metadata.")

    column = "_comparison_column"
    conditions = [
        metadata[Cols.SAMPLE].isin(group1_list),
        metadata[Cols.SAMPLE].isin(group2_list),
    ]
    choices = ["group1", "group2"]
    metadata[column] = np.select(conditions, choices, default=np.nan)

    return metadata, column
