from alphastats.loader import BaseLoader
import pandas as pd
import numpy as np

class MaxQuantLoader(BaseLoader):
    def __init__(self,
        file,
        intensity_column = "LFQ intentsity [experiment]",
        index_column = "Protein IDs",
        filter_columns = ["Only identified by site", "Reverse", "Potential contaminant"],
        confidence_column = "Q-value",
        sep = "\t"):
        """_summary_

        Args:
            intensity_column (str, optional): _description_. Defaults to "LFQ intentsity [experiment]".
            index_column (str, optional): _description_. Defaults to "Protein IDs".
            filter_columns (list, optional): _description_. Defaults to ["Only identified by site", "Reverse", "Potential contaminant"].
            qvalue_column (str, optional): _description_. Defaults to "Q-value".
        """
        super.__init__(file, intensity_column, index_column, sep)

        self.filter_columns = filter_columns
        self.confidence_column = confidence_column
        self.software = "MaxQuant"
        self.set_filter_columns_to_true_false()


    def set_filter_columns_to_true_false(self):
        """replaces the '+' with True, else False
        """
        if len(self.filter_columns) > 0:
            for filter_column in self.filter_columns:
                self.rawdata[filter_column] = np.where(self.rawdata[filter_column] == "+",
                True, False)

    
