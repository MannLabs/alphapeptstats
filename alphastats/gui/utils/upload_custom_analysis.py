"""Utility functions to handle custom analysis file uploads and parsing."""

import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from alphastats.dataset.id_holder import IdHolder
from alphastats.dataset.keys import Cols, Regulation
from alphastats.gui.utils.result import ResultComponent


def parse_custom_analysis_file(uploaded_file: UploadedFile) -> pd.DataFrame:
    """Parse uploaded custom analysis file and extract relevant columns."""
    # Read the uploaded file as tab-separated values
    df = pd.read_csv(uploaded_file, sep="\t")

    # Check if required columns exist
    required_columns = ["Significant", "Difference", "Protein IDs", "Gene names"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Extract only the required columns
    parsed_df = df[required_columns].copy()

    # Rename Protein IDs to index_
    parsed_df = parsed_df.rename(columns={"Protein IDs": Cols.INDEX})

    # Convert Significant column based on significance and difference direction
    # Handle both string values and NaN values
    significant_clean = parsed_df["Significant"].fillna("").astype(str)
    is_significant = significant_clean.map(
        {
            "+": True,
            "": False,
            " ": False,
            "nan": False,
        }
    ).fillna(False)  # noqa: FBT003

    # Create UP/DOWN based on significance and difference direction
    parsed_df[Cols.SIGNIFICANT] = "NON_SIG"  # Default value

    # Set UP for significant entries with positive difference
    up_mask = is_significant & (parsed_df["Difference"] > 0)
    parsed_df.loc[up_mask, Cols.SIGNIFICANT] = Regulation.UP

    # Set DOWN for significant entries with negative difference
    down_mask = is_significant & (parsed_df["Difference"] < 0)
    parsed_df.loc[down_mask, Cols.SIGNIFICANT] = Regulation.DOWN

    return parsed_df


def create_custom_result_component(
    parsed_df: pd.DataFrame,
) -> tuple[ResultComponent, IdHolder]:
    """Create a simplified ResultComponent from parsed custom analysis data."""
    # Create basic ResultComponent with minimal required attributes
    result_component = ResultComponent(
        dataframe=parsed_df,
        preprocessing={},
        method={},
        feature_to_repr_map={},
        is_plottable=False,
    )

    # Set annotated_dataframe to the same data (pre-filled as requested)
    result_component.annotated_dataframe = parsed_df.copy()

    id_holder = IdHolder(
        features_list=list(parsed_df["index_"]),
        proteins_list=list(parsed_df["Gene names"]),
    )

    return result_component, id_holder
