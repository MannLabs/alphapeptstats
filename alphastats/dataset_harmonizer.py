"""Harmonize the input data to a common format."""

from typing import Dict, Optional

import pandas as pd

from alphastats import BaseLoader
from alphastats.keys import Cols


class DataHarmonizer:
    """Harmonize input data to a common format."""

    def __init__(self, loader: BaseLoader, sample_column: Optional[str] = None):
        rawinput_rename_dict = {
            loader.index_column: Cols.INDEX,
            loader.gene_names_column: Cols.GENE_NAMES,
        }

        shared_rename_dict = (
            {
                sample_column: Cols.SAMPLE,
            }
            if sample_column is not None
            else {}
        )

        self._rawinput_rename_dict = {**rawinput_rename_dict, **shared_rename_dict}
        self._metadata_rename_dict = shared_rename_dict

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
        input: pd.DataFrame, rename_dict: Dict[str, str]
    ) -> pd.DataFrame:
        """Harmonize data to a common format."""
        for target_name in rename_dict.values():
            if target_name in input.columns:
                raise ValueError(
                    f"Column name {target_name} already exists. Please rename the column."
                )

        return input.rename(
            columns=rename_dict,
            errors="ignore",
        )
