"""Harmonize the input data to a common format."""

from typing import Dict, Optional

import pandas as pd

from alphastats.dataset.keys import Cols
from alphastats.loader.base_loader import BaseLoader


class DataHarmonizer:
    """Harmonize input data to a common format."""

    def __init__(self, loader: BaseLoader, sample_column_name: Optional[str] = None):
        _rawinput_rename_dict = {
            loader.index_column: Cols.INDEX,
        }
        if loader.gene_names_column is not None:
            _rawinput_rename_dict[loader.gene_names_column] = Cols.GENE_NAMES

        self._rawinput_rename_dict = _rawinput_rename_dict

        self._metadata_rename_dict = (
            {
                sample_column_name: Cols.SAMPLE,
            }
            if sample_column_name is not None
            else {}
        )

    def get_harmonized_rawinput(self, rawinput: pd.DataFrame) -> pd.DataFrame:
        """Harmonize the rawinput data to a common format."""
        return self._get_harmonized_data(
            rawinput,
            self._rawinput_rename_dict,
        )

    def get_harmonized_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Harmonize the rawinput data to a common format."""
        return self._get_harmonized_data(
            metadata,
            self._metadata_rename_dict,
        )

    @staticmethod
    def _get_harmonized_data(
        input_df: pd.DataFrame, rename_dict: Dict[str, str]
    ) -> pd.DataFrame:
        """Harmonize data to a common format."""
        for target_name in rename_dict.values():
            if target_name in input_df.columns:
                raise ValueError(
                    f"Column name '{target_name}' already exists. Please rename the column in your input data."
                )

        return input_df.rename(
            columns=rename_dict,
            errors="raise",
        )
