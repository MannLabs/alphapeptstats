import pandas as pd
import logging

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

        self.confidence_column = None
        self.software = None
        self.check_if_columns_are_present()


    def check_if_columns_are_present(self):
        """check if given columns present in rawdata
        """
        given_columns = list(filter(None,[self.index_column, self.confidence_column]))
        wrong_columns = list(set([given_columns]) - set(self.rawdata.columns.to_list()))
        if len(wrong_columns) > 0:
            logging.error(", ".join(wrong_columns) + " columns do not exist.")
    
    def add_contamination_column(self):
        # load dict with potential contamination from fasta file
        # add column with True False
        self.rawdata["Potential contaminant"]
        pass


        # get different output formats in alpha stat format
        # self.value_column ="Precursor.Normalised" # "PG.Quantity",
        # self.index_column = "Protein.Group",
        # self.qvalue_column = ["PG.Q.Value", "Q.Value"] # both qvalues are used for filtering in R-package of DIA-NN
        # allow multiple Q-value colummns qvalue_
        # self.filter_column = []
        # filter column should be binary
        # allow multiple filter columns 
        

# ALPHASTATS STANDARDS

# intensity_column  
# confidence_column -> Q-value column in MaxQuant, ProteinProbability in MSFragger, DIA-NN and AlphaPept don't contain confidence column
# filter_columns -> added by alphastats if not MaxQuant "Contaminations" - use fasta.database, filters are annotated with True/False