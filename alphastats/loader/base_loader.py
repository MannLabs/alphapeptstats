import importlib.resources as importlib_resources
import logging
from typing import Union

import numpy as np
import pandas as pd

from alphastats.dataset.utils import find_duplicates_in_list


# TODO make the child classes instantiated this class
class BaseLoader:
    """Parent class of Loaders"""

    def __init__(
        self,
        file: Union[str, pd.DataFrame],
        intensity_column: Union[str, list],
        index_column: str,
        sep: str,
    ):
        """BaseLoader for AlphaPept, MaxQuant, Fragpipe, Spectronau and DIANNLoader

        Args:
            file_path (str): path to file
            sep (str, optional): file separation. Defaults to "\t".
        """

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
        self.gene_names_column = None
        self.ptm_df = None
        self._add_contamination_column()
        self._check_if_columns_are_present()
        self._read_all_column_names_as_string()

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

    def _read_all_column_names_as_string(self):
        self.rawinput.columns = self.rawinput.columns.astype(str)

    # TODO unused
    def _check_if_indexcolumn_is_unique(self):
        duplicated_values = find_duplicates_in_list(
            self.rawinput[self.index_column].to_list()
        )
        if len(duplicated_values) > 0:
            # error or warning, duplicates could be resolved with preprocessing/filtering
            logging.warning(
                f"Column {self.index_column} contains duplicated values: "
                + ", ".join(duplicated_values)
            )

    def _add_contamination_column(self):
        # load df with potential contamination from fasta file
        contaminations_path = (
            importlib_resources.files(__package__) / "../data/contaminations.txt"
        )
        contaminations = pd.read_csv(contaminations_path, sep="\t")
        contaminations_ids = contaminations["Uniprot ID"].to_list()
        # add column with True False

        self.rawinput["contamination_library"] = np.where(
            self.rawinput[self.index_column].isin(contaminations_ids), True, False
        )
        self.filter_columns = self.filter_columns + ["contamination_library"]

        logging.info(
            "Column 'contamination_library' has been added, to indicate contaminations.\n"
            + "The contaminant library was created by Frankenfield et al."
            + ":https://www.biorxiv.org/content/10.1101/2022.04.27.489766v2.full"
        )

    def _get_intensity_columns(self):
        """Get the intensity columns"""
        if isinstance(self.intensity_column, list):
            return self.intensity_column
        else:
            return [
                col
                for col in self.rawinput.columns
                if self.intensity_column.replace("[sample]", "") in col
            ]
