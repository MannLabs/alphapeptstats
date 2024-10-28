"""Harmonize the input data to a common format."""

import pandas as pd
from keys import Cols

from alphastats import BaseLoader


class DataHarmonizer:
    """Harmonize input data to a common format."""

    def __init__(self, loader: BaseLoader):
        self._rename_dict = {loader.index_column: Cols.INDEX}

    def get_harmonized_rawinput(self, rawinput: pd.DataFrame) -> pd.DataFrame:
        """Harmonize the rawinput data to a common format."""
        for target_name in self._rename_dict.values():
            if target_name in rawinput.columns:
                raise ValueError(
                    f"Column name {target_name} already exists in rawinput. Please rename the column."
                )

        return rawinput.rename(
            columns=self._rename_dict,
            errors="raise",
        )
