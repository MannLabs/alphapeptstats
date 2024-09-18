from typing import List, Union, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import logging
import warnings


class DataSetFactory:
    """Create all 'heavy' data structures of a DataSet."""

    def __init__(
        self,
        *,
        rawinput: pd.DataFrame,
        index_column: str,
        intensity_column: Union[List[str], str],
        metadata_path: Union[str, pd.DataFrame],
        sample_column: str,
    ):
        self.rawinput: pd.DataFrame = rawinput
        self.sample_column: str = sample_column
        self.index_column: str = index_column
        self.intensity_column: Union[List[str], str] = intensity_column
        self.metadata_path: Union[str, pd.DataFrame] = metadata_path

    def create_matrix_from_rawinput(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Creates a matrix: features (Proteins) as columns, samples as rows."""

        df = self.rawinput
        df = df.set_index(self.index_column)

        if isinstance(self.intensity_column, str):
            regex_find_intensity_columns = self.intensity_column.replace(
                "[sample]", ".*"
            )
            df = df.filter(regex=regex_find_intensity_columns, axis=1)
            # remove Intensity so only sample names remain
            substring_to_remove = regex_find_intensity_columns.replace(".*", "")
            df.columns = df.columns.str.replace(substring_to_remove, "")

        else:
            df = df[self.intensity_column]

        rawmat = df.transpose()
        rawmat.replace([np.inf, -np.inf], np.nan, inplace=True)

        # remove proteins with only zero  # TODO this is re-done in preprocessing
        mat_no_zeros = rawmat.loc[:, (rawmat != 0).any(axis=0)].astype(float)

        self._check_matrix_values(mat_no_zeros)

        return rawmat, mat_no_zeros

    @staticmethod
    def _check_matrix_values(mat: pd.DataFrame) -> None:
        """Check for infinite values in the matrix."""
        if np.isinf(mat).values.sum() > 0:
            logging.warning("Data contains infinite values.")

    def create_metadata(self, mat: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create metadata DataFrame from metadata file or DataFrame."""

        if self.metadata_path is not None:
            sample = self.sample_column
            metadata = self._load_metadata(file_path=self.metadata_path)
            metadata = self._remove_missing_samples_from_metadata(mat, metadata, sample)
        else:
            sample = "sample"
            metadata = pd.DataFrame({"sample": list(mat.index)})

        return metadata, sample

    def _remove_missing_samples_from_metadata(
        self, mat: pd.DataFrame, metadata: pd.DataFrame, sample
    ) -> pd.DataFrame:
        """Remove samples from metadata that are not in the protein data."""
        samples_matrix = mat.index.to_list()
        samples_metadata = metadata[sample].to_list()
        misc_samples = list(set(samples_metadata) - set(samples_matrix))
        if len(misc_samples) > 0:
            metadata = metadata[~metadata[sample].isin(misc_samples)]
            logging.warning(
                f"{misc_samples} are not described in the protein data and"
                "are removed from the metadata."
            )
        return metadata

    def _load_metadata(
        self, file_path: Union[pd.DataFrame, str]
    ) -> Optional[pd.DataFrame]:
        """Load metadata either xlsx, txt, csv or txt file

        Args:
            file_path: path to metadata file or metadata DataFrame  # TODO disentangle this
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        elif file_path.endswith(".xlsx"):
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="openpyxl",
                # message=r"/extension is not supported and will be removed/",  # this somehow does not work here?
            )
            df = pd.read_excel(file_path)
            # find robust way to detect file format
            # else give file separation as variable
        elif file_path.endswith(".txt") or file_path.endswith(".tsv"):
            df = pd.read_csv(file_path, delimiter="\t")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            logging.warning(
                "WARNING: Metadata could not be read. \nMetadata has to be a .xslx, .tsv, .csv or .txt file"
            )
            return None

        if df is not None and self.sample_column not in df.columns:
            logging.error(
                f"sample_column: {self.sample_column} not found in {file_path}"
            )

        # check whether sample labeling matches protein data
        #  warnings.warn("WARNING: Sample names do not match sample labelling in protein data")
        df.columns = df.columns.astype(str)
        return df
