"""Harmonize the input data to a common format."""

import pandas as pd
from keys import Cols

from alphastats import BaseLoader


class DataHarmonizer:
    """Harmonize input data to a common format."""

    def __init__(self, loader: BaseLoader):
        self._rawinput = loader.rawinput
        self._index_column = loader.index_column

    def get_rawinput(self) -> pd.DataFrame:
        """Harmonize the rawinput data to a common format."""
        if Cols.INDEX in self._rawinput.columns:
            raise ValueError(
                f"Column name {Cols.INDEX} already exists in rawinput. Please rename the column."
            )

        return self._rawinput.rename(
            columns={self._index_column: Cols.INDEX},
            errors="raise",
        )
