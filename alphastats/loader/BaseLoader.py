import pandas as pd
import logging
import os
import numpy as np
from iteration_utilities import duplicates
import pkg_resources


class BaseLoader:
    """Parent class of Loaders"""

    def __init__(self, file, intensity_column, index_column, sep):
        """BaseLoader for AlphaPept, MaxQuant, Fragpipe and DIANNLoader

        Args:
            file_path (str): path to file
            sep (str, optional): file separation. Defaults to "\t".
        """

        # self._check_if_file_exists(file=file)
        if isinstance(file, pd.DataFrame):
            self.rawinput = file
        else:
            self.rawinput = pd.read_csv(file, sep=sep, low_memory=False)
        self.intensity_column = intensity_column
        self.index_column = index_column
        self.filter_columns = []
        self.confidence_column = None
        self.software = None
        self.evidence_df = None
        self.gene_names = None
        self.ptm_df = None
        self._add_contamination_column()
        self._check_if_columns_are_present()

    def _check_if_columns_are_present(self):
        """check if given columns present in rawinput"""
        given_columns = list(filter(None, [self.index_column, self.confidence_column]))
        wrong_columns = list(set(given_columns) - set(self.rawinput.columns.to_list()))
        if len(wrong_columns) > 0:
            raise KeyError(
                ", ".join(wrong_columns) + " columns do not exist.\n"
                "Check the documtentation: \n"
                "AlphaPept Format: https://github.com/MannLabs/alphapept \n"
                "DIA-NN Format: https://github.com/vdemichev/DiaNN"
                "FragPipe Format: https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv"
                "MaxQuant Format: http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable"
            )

    def _check_if_indexcolumn_is_unique(self):
        # TODO make own duplicates functions to have less dependencies
        duplicated_values = list(duplicates(self.rawinput[self.index_column].to_list()))
        if len(duplicated_values) > 0:
            # error or warning, duplicates could be resolved with preprocessing/filtering
            logging.warning(
                f"Column {self.index_column} contains duplicated values: "
                + ", ".join(duplicated_values)
            )

    def _check_if_file_exists(self, file):
        if os.path.isfile(file) == False:
            raise OSError(f"{file} does not exist.")

    def _add_contamination_column(self):
        #  load df with potential contamination from fasta file
        contaminations_path = pkg_resources.resource_filename(
            __name__, "../data/contaminations.txt"
        )
        contaminations = pd.read_csv(contaminations_path, sep="\t")
        contaminations_ids = contaminations["Uniprot ID"].to_list()
        #  add column with True False

        self.rawinput["contamination_library"] = np.where(
            self.rawinput[self.index_column].isin(contaminations_ids), True, False
        )
        self.filter_columns = self.filter_columns + ["contamination_library"]

        logging.info(
            "Column 'contamination_library' has been added, to indicate contaminations.\n"
            + "The contaminant library was created by Frankenfield et al."
            + ":https://www.biorxiv.org/content/10.1101/2022.04.27.489766v2.full"
        )
