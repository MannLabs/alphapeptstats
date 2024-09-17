import copy
from alphastats.loader.BaseLoader import BaseLoader
import pandas as pd
import numpy as np
import logging
import re
import warnings


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

        columns = (
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

        self.rawinput = self._read_spectronaut_file(file=file, sep=sep, columns=columns)

        if gene_names_column in self.rawinput.columns.to_list():
            self.gene_names = gene_names_column
        else:
            self.gene_names = None
        if sample_column in self.rawinput.columns.to_list():
            self.sample_column = sample_column
        else:
            self.sample_column = None
            warnings.warn(
                f"No sample column was found so the data will be treated as wide format. If this is wrong make sure {sample_column} is present in the file."
            )

        self.rawinput = self._deduplicate_data(long=self.sample_column)

        if self.sample_column is not None:
            self.rawinput = self._reshape_long_to_wide()

        self.intensity_column = "[sample]." + self.intensity_column

        self._add_contamination_column()
        self._read_all_columns_as_string()

    def _reshape_long_to_wide(self):
        """
        other proteomics softwares use a wide format (column for each sample)
        reshape to a wider format
        """
        self.rawinput["sample"] = (
            self.rawinput[self.sample_column] + "." + self.intensity_column
        )
        indexing_columns = [self.index_column]
        if self.gene_names is not None:
            indexing_columns.append(self.gene_names)

        df = self.rawinput.pivot(
            columns="sample", index=indexing_columns, values=self.intensity_column
        )
        df.reset_index(inplace=True)

        return df

    def _deduplicate_data(self, long):
        subset = [self.index_column] + [
            col for col in self.rawinput.columns if self.intensity_column in col
        ]
        if long:
            subset.append(self.sample_column)
        unique_df = self.rawinput.drop_duplicates(subset=subset)
        return unique_df

    def _read_spectronaut_file(self, file, sep, columns):
        # some spectronaut files include european decimal separators
        if isinstance(file, pd.DataFrame):
            df = file[[col for col in file.columns if bool(re.match(columns, col))]]
            for column in df.columns:
                try:
                    if df[column].dtype == np.float64:
                        continue
                    df[column] = df[column].str.replace(",", ".").astype(float)
                    print("converted", column, df[column].dtype)
                except (ValueError, AttributeError) as e:
                    print("failed", column, df[column].dtype)
        else:
            df = pd.read_csv(
                file,
                sep=sep,
                low_memory=False,
                usecols=lambda col: bool(re.match(columns, col)),
            )
            for column in df.columns:
                try:
                    if df[column].dtype == np.float64:
                        continue
                    df[column] = df[column].str.replace(",", ".").astype(float)
                    print("converted", column, df[column].dtype)
                except (ValueError, AttributeError) as e:
                    print("failed", column, df[column].dtype)

        return df
