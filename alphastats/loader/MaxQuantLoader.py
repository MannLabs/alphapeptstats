from alphastats.loader.BaseLoader import BaseLoader
import pandas as pd
import numpy as np


class MaxQuantLoader(BaseLoader):
    """Loader for MaxQuant outputfiles"""

    def __init__(
        self,
        file,
        intensity_column="LFQ intensity [sample]",
        index_column="Protein IDs",
        gene_names_column="Gene names",
        filter_columns=["Only identified by site", "Reverse", "Potential contaminant"],
        confidence_column="Q-value",
        evidence_file=None,
        sep="\t",
        **kwargs
    ):
        """Loader MaxQuant output

        Args:
            file (str): ProteinGroups.txt file: http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable
            intensity_column (str, optional): columns with Intensity values for each sample. Defaults to "LFQ intentsity [experiment]".
            index_column (str, optional): column with Protein IDs . Defaults to "Protein IDs".
            filter_columns (list, optional): columns that should be used for filtering. Defaults to ["Only identified by site", "Reverse", "Potential contaminant"].
            confidence_column (str, optional): column with the Q-value given. Defaults to "Q-value".
            sep (str, optional): separation of the input file. Defaults to "\t".
        """

        super().__init__(file, intensity_column, index_column, sep)
        self.filter_columns = filter_columns + self.filter_columns
        self.confidence_column = confidence_column
        self.software = "MaxQuant"
        self._set_filter_columns_to_true_false()

        if gene_names_column in self.rawinput.columns.to_list():
            self.gene_names = gene_names_column

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
