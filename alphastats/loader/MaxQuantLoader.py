from alphastats.loader.BaseLoader import BaseLoader
import pandas as pd
import numpy as np


class MaxQuantLoader(BaseLoader):
    """Loader for MaxQuant outputfiles
    """

    def __init__(
        self,
        file,
        intensity_column="LFQ intensity [sample]",
        index_column="Protein IDs",
        filter_columns=["Only identified by site", "Reverse", "Potential contaminant"],
        confidence_column="Q-value",
        sep="\t",
        **kwargs
    ):
        """Loader MaxQuant output 

        Args:
            file (_type_): ProteinGroups.txt file: http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable
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
        self.set_filter_columns_to_true_false()

    def set_filter_columns_to_true_false(self):
        """replaces the '+' with True, else False
        """
        if len(self.filter_columns) > 0:
            for filter_column in self.filter_columns:
                self.rawdata[filter_column] = np.where(
                    self.rawdata[filter_column] == "+", True, False
                )
