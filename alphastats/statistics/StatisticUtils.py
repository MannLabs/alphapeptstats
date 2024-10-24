import numpy as np
import pandas as pd


def _add_metadata_column(
    metadata: pd.DataFrame, sample: str, group1_list: list, group2_list: list
):
    # create new column in metadata with defined groups

    sample_names = metadata[sample].to_list()
    misc_samples = list(set(group1_list + group2_list) - set(sample_names))
    if len(misc_samples) > 0:
        raise ValueError(f"Sample names: {misc_samples} are not described in Metadata.")

    column = "_comparison_column"
    conditons = [
        metadata[sample].isin(group1_list),
        metadata[sample].isin(group2_list),
    ]
    choices = ["group1", "group2"]
    metadata[column] = np.select(conditons, choices, default=np.nan)

    return metadata, column
