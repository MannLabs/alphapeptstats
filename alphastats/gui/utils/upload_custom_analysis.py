"""Utility functions to handle custom analysis file uploads and parsing."""

from collections import defaultdict

import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from alphastats.dataset.id_holder import IdHolder
from alphastats.dataset.keys import Cols, Regulation
from alphastats.gui.utils.result import ResultComponent

GENE_NAMES = "Gene names"
MAJORITY_PROTEIN_IDS = "Majority Protein IDs"
DIFFERENCE = "Difference"
SIGNIFICANT = "Significant"


def parse_custom_analysis_file(uploaded_file: UploadedFile) -> pd.DataFrame:
    """Parse uploaded custom analysis file and extract relevant columns."""
    df = pd.read_csv(uploaded_file, sep="\t")

    required_columns = [
        SIGNIFICANT,
        DIFFERENCE,
        MAJORITY_PROTEIN_IDS,
        GENE_NAMES,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    parsed_df = df[required_columns].copy()

    parsed_df = parsed_df.rename(columns={MAJORITY_PROTEIN_IDS: Cols.INDEX})

    significant_clean = parsed_df[SIGNIFICANT].fillna("").astype(str)
    significance_mapping = defaultdict(lambda: False)
    significance_mapping["+"] = True
    is_significant = significant_clean.map(significance_mapping).fillna(False)  # noqa: FBT003

    parsed_df[Cols.SIGNIFICANT] = Regulation.NON_SIG

    up_mask = is_significant & (parsed_df[DIFFERENCE] > 0)
    parsed_df.loc[up_mask, Cols.SIGNIFICANT] = Regulation.UP

    down_mask = is_significant & (parsed_df[DIFFERENCE] < 0)
    parsed_df.loc[down_mask, Cols.SIGNIFICANT] = Regulation.DOWN

    return parsed_df


def create_custom_result_component(
    parsed_df: pd.DataFrame,
) -> tuple[ResultComponent, IdHolder]:
    """Create a simplified ResultComponent and an IdHolder from parsed custom analysis data."""
    result_component = ResultComponent(
        dataframe=parsed_df,
        preprocessing={},
        method={},
        feature_to_repr_map={},
        is_plottable=False,
    )

    result_component.annotated_dataframe = parsed_df.copy()

    id_holder = IdHolder(
        features_list=list(parsed_df[Cols.INDEX]),
        proteins_list=list(parsed_df[Cols.INDEX]),
        gene_names_list=list(parsed_df[GENE_NAMES]),
    )

    return result_component, id_holder
