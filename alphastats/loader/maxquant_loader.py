import re
from typing import Union

import numpy as np
import pandas as pd

from alphastats.loader.base_loader import BaseLoader


class MaxQuantLoader(BaseLoader):
    """Loader for MaxQuant outputfiles"""

    def __init__(
        self,
        file,
        intensity_column: Union[str, list] = "LFQ intensity [sample]",
        index_column: str = "Protein IDs",
        gene_names_column: str = "Gene names",
        filter_columns: list = None,
        confidence_column: str = "Q-value",
        evidence_file=None,
        sep: str = "\t",
        **kwargs,
    ):
        """Loader MaxQuant output

        Special handling for Maxquant data includes removal of additional input rows that stems from overflowing id references. This is done by checking if the Protein IDs contain at least one letter, as the overflow is a string of numebrs a semicolons. If numeric ids are used, filter the data for any non-nan values in the intensity columns.

        Args:
            file (str): ProteinGroups.txt file: http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable
            intensity_column (str, optional): columns with Intensity values for each sample. Defaults to "LFQ intentsity [experiment]".
            index_column (str, optional): column with Protein IDs . Defaults to "Protein IDs".
            filter_columns (list, optional): columns that should be used for filtering. Defaults to ["Only identified by site", "Reverse", "Potential contaminant"].
            confidence_column (str, optional): column with the Q-value given. Defaults to "Q-value".
            sep (str, optional): separation of the input file. Defaults to "\t".
        """

        if filter_columns is None:
            filter_columns = [
                "Only identified by site",
                "Reverse",
                "Potential contaminant",
            ]
        super().__init__(file, intensity_column, index_column, sep)
        self.filter_columns = filter_columns + self.filter_columns
        self.confidence_column = confidence_column
        self.software = "MaxQuant"
        self._set_filter_columns_to_true_false()
        self._read_all_column_names_as_string()

        intensity_columns = self._get_intensity_columns()
        if len(self.rawinput.dropna(subset=intensity_columns, how="all")) != len(
            self.rawinput
        ):
            # there are likely overflowing id rows
            valid_id = re.compile("[A-Z]")
            self.rawinput = self.rawinput[
                self.rawinput[self.index_column].apply(
                    lambda x: isinstance(x, str) and bool(valid_id.match(x))
                )
            ]

        if gene_names_column in self.rawinput.columns.to_list():
            self.gene_names_column = gene_names_column

        if evidence_file is not None:
            self._load_evidence(evidence_file=evidence_file)

    def _load_evidence(self, evidence_file, sep="\t"):
        self.evidence_df = pd.read_csv(evidence_file, sep=sep, low_memory=False)

        evi_sample_names = self.evidence_df["Raw file"].to_list()
        pg_sample_names = self._extract_sample_names()

        intersection_sample_names = list(set(evi_sample_names) & set(pg_sample_names))
        if len(intersection_sample_names) == 0:
            raise ValueError(
                "Sample names in proteinGroups.txt do not match"
                "sample names in evidence.txt file"
            )

    def _extract_sample_names(self):
        regex_find_intensity_columns = self.intensity_column.replace("[sample]", ".*")
        df = self.rawinput
        df = df.filter(regex=(regex_find_intensity_columns), axis=1)
        # remove Intensity so only sample names remain
        substring_to_remove = regex_find_intensity_columns.replace(".*", "")
        df.columns = df.columns.str.replace(substring_to_remove, "")
        return df.columns.to_list()

    def _set_filter_columns_to_true_false(self):
        """replaces the '+' with True, else False"""
        if len(self.filter_columns) > 0:
            for filter_column in self.filter_columns:
                self.rawinput[filter_column] = np.where(
                    self.rawinput[filter_column] == "+", True, False
                )
