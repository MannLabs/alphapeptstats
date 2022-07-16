import pandas as pd
import logging
import os
import numpy as np
from iteration_utilities import duplicates


class BaseLoader:
    """Parent class of Loaders
    """

    def __init__(self, file, intensity_column, index_column, sep):
        """_summary_

        Args:
            file_path (str): path to file 
            sep (str, optional): file separation. Defaults to "\t".
        """

        self.check_if_file_exists(file=file)
        self.rawdata = pd.read_csv(file, sep=sep, low_memory=False)
        self.intensity_column = intensity_column
        self.index_column = index_column
        self.filter_columns = []
        self.confidence_column = None
        self.software = None
        self.add_contamination_column()
        self.check_if_columns_are_present()

    def check_if_columns_are_present(self):
        """check if given columns present in rawdata
        """
        given_columns = list(filter(None, [self.index_column, self.confidence_column]))
        wrong_columns = list(set(given_columns) - set(self.rawdata.columns.to_list()))
        if len(wrong_columns) > 0:
            logging.error(", ".join(wrong_columns) + " columns do not exist.")
            # raise KeyError

    def check_if_indexcolumn_is_unique(self):
        # TODO make own duplicates functions to have less dependencies
        duplicated_values = list(duplicates(self.rawdata[self.index_column].to_list()))
        if len(duplicated_values) > 0:
            # error or warning, duplicates could be resolved with preprocessing/filtering
            logging.warning(
                f"Column {self.index_column} contains duplicated values: "
                + ", ".join(duplicated_values)
            )

    def check_if_file_exists(self, file):
        if os.path.isfile(file) == False:
            logging.error(f"{file} does not exist.")
            # raise OSError

    def add_contamination_column(self):
        #  load df with potential contamination from fasta file
        contaminations = pd.read_csv("misc/contaminations.txt", sep="\t")
        contaminations_ids = contaminations["Uniprot ID"].to_list()
        #  add column with True False

        self.rawdata["contamination_library"] = np.where(
            self.rawdata[self.index_column].isin(contaminations_ids), True, False
        )
        self.filter_columns = self.filter_columns + ["contamination_library"]

        logging.info(
            "Column 'Contamination_library' has been added, to indicate contaminations.\n"
            + "The contaminant library was created by Frankenfield et al. :https://www.biorxiv.org/content/10.1101/2022.04.27.489766v2.full"
        )
