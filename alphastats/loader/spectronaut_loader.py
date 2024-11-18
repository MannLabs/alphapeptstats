import re
import warnings
from typing import Union

import numpy as np
import pandas as pd

from alphastats.loader.base_loader import BaseLoader

SPECTRONAUT_COLUMN_DELIM = "."


class SpectronautLoader(BaseLoader):
    """Loader for Spectronaut outputfiles"""

    def __init__(
        self,
        file,
        intensity_column="PG.Quantity",
        index_column="PG.ProteinGroups",
        sample_column="R.FileName",
        gene_names_column="PG.Genes",
        sep="\t",
    ):
        """Loads Spectronaut output. Will add contamination column for further analysis.

        Args:
            file (str): path to Spectronaut outputfile or pandas.DataFrame
            intensity_column (str, optional): columns where the intensity of the proteins are given. Defaults to "PG.Quantity".
            index_column (str, optional): column indicating the protein groups. Defaults to "PG.ProteinGroups".
            sample_column (str, optional): column that contains sample names used for downstream analysis. Defaults to "R.FileName".
            gene_names_column (str, optional): column with gene names. Defaults to "PG.Genes".
            filter_qvalue (bool, optional): will filter out the intensities that have greater than qvalue_cutoff in EG.Qvalue column. Those intensities will be replaced with zero and will be considered as censored missing values for imputation purpose.. Defaults to True.
            qvalue_cutoff (float, optional): cut off value. Defaults to 0.01.
            sep (str, optional): file separation of file. Defaults to "\t".
        """

        self.software = "Spectronaut"
        self.intensity_column = intensity_column
        self.index_column = index_column
        self.confidence_column = None
        self.filter_columns = []
        self.evidence_df = None

        column_selection_regex = (
            "^"
            + "$|^".join(
                [
                    ".*" + intensity_column,
                    index_column,
                    sample_column,
                    gene_names_column,
                ]
            )
            + "$"
        )

        self.rawinput = self._read_spectronaut_file(
            file=file, sep=sep, column_selection_regex=column_selection_regex
        )

        if gene_names_column in self.rawinput.columns.to_list():
            self.gene_names_column = gene_names_column
        else:
            self.gene_names_column = None
        if sample_column in self.rawinput.columns.to_list():
            self.sample_column = sample_column
        else:
            self.sample_column = None
            warnings.warn(
                f"No sample column was found so the data will be treated as wide format. If this is wrong make sure {sample_column} is present in the file."
            )

        self.rawinput = self._deduplicate_data(long=bool(self.sample_column))

        if self.sample_column is not None:
            self.rawinput = self._reshape_long_to_wide()

        self.intensity_column = (
            "[sample]" + SPECTRONAUT_COLUMN_DELIM + self.intensity_column
        )

        self._add_contamination_column()
        self._read_all_column_names_as_string()

    def _reshape_long_to_wide(self):
        """
        other proteomics softwares use a wide format (column for each sample)
        reshape to a wider format
        """
        self.rawinput["tmp_sample"] = (
            self.rawinput[self.sample_column]
            + SPECTRONAUT_COLUMN_DELIM
            + self.intensity_column
        )
        indexing_columns = [self.index_column]
        if self.gene_names_column is not None:
            indexing_columns.append(self.gene_names_column)

        df = self.rawinput.pivot(
            columns="tmp_sample", index=indexing_columns, values=self.intensity_column
        )
        df.reset_index(inplace=True)
        # get rid of tmp_sample again, which can cause troubles when working with indices downstream
        df.rename_axis(columns=None, inplace=True)

        return df

    def _deduplicate_data(self, long: bool):
        """Deduplicates the data based on the index column and the intensity column.

        If long format is used, the sample column is also used for indexing. Deduplicaiton is necessary because additional columns for lower level qunatification, i.e. peptides, can be present in the data, leading to duplication of the protein group data.

        Args:
            long (bool): if the data is in long format

        Returns:
            pd.DataFrame: deduplicated data
        """
        subset = [self.index_column] + [
            col for col in self.rawinput.columns if self.intensity_column in col
        ]
        if long:
            subset.append(self.sample_column)
        unique_df = self.rawinput.drop_duplicates(subset=subset)
        return unique_df

    def _read_spectronaut_file(
        self, file: Union[str, pd.DataFrame], sep: str, column_selection_regex: str
    ):
        """
        Reads the Spectronaut file and converts numeric columns to float.

        Some spectronaut files include european decimal separators.

        Args:
            file (Union[str, pd.DataFrame]): path to the file or pandas.DataFrame
            sep (str): separator of the file
            column_selection_regex (str): regex pattern for columns to read

        Returns:
            df (pd.DataFrame): Spectronaut data
        """
        if isinstance(file, pd.DataFrame):
            df = file[
                [
                    col
                    for col in file.columns
                    if bool(re.match(column_selection_regex, col))
                ]
            ]
        else:
            df = pd.read_csv(
                file,
                sep=sep,
                low_memory=False,
                usecols=lambda col: bool(re.match(column_selection_regex, col)),
            )

        for column in df.columns:
            try:
                if df[column].dtype == np.float64:
                    continue
                df[column] = df[column].str.replace(",", ".").astype(float)
                print("converted", column, df[column].dtype)
            except (ValueError, AttributeError):
                print("failed", column, df[column].dtype)

        return df
