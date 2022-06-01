import pandas as pd

class BaseLoader:
    """Parent class of Loaders
    """
    def __init__(self, file, intensity_column, index_column, sep):
        """_summary_

        Args:
            file_path (str): path to file 
            sep (str, optional): file separation. Defaults to "\t".
        """

        self.rawdata = pd.read_csv(file, sep = sep)
        self.intensity_column = intensity_column
        self.index_column = index_column

        self.sample_column = None
        self.value_column = None
        
        self.qvalue_column = None
        self.software = None

        # get different output formats in alpha stat format
        # self.value_column ="Precursor.Normalised" # "PG.Quantity",
        # self.index_column = "Protein.Group",
        # self.qvalue_column = ["PG.Q.Value", "Q.Value"] # both qvalues are used for filtering in R-package of DIA-NN
        # allow multiple Q-value colummns qvalue_
        # self.filter_column = []
        # filter column should be binary
        # allow multiple filter columns filter_
        
        