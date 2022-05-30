from alphastats.loader import BaseLoader
import pandas as pd

"""
- older versions of DIA-NN don't contain MaxLFQ normalization here Precursor.Normalised should be taken
- Several output files use: report.pg_matrix.tsv
- else report.tsv could be used but is long data
- wont work with older versions of DIA-NN

"""

class DIANNLoader(BaseLoader):

   def __init__(self,
                file,
                sample_column = "[experiment]", 
                index_column =  "Protein.Group"):
                #qvalue_column = ["PG.Q.Value", "Q.Value"],
                #qvalue_threshold = 0.01,
                #load_matrix = False): # add option so you can load matrix from diann-rpackage
        
        #self.rawdata = self.convert_to_wide_data(file = file, qvalue_threshold = qvalue_threshold)
        self.rawdata = pd.read_csv(file, sep = "\t")
        self.sample_column = sample_column
        self.index_column = index_column
        self.software = "DIA-NN"

   # def convert_to_wide_data(self, file, qvalue_threshold):
        # FDR needs to be filtered before converting to wide format
        # pass


    # https://github.com/vdemichev/DiaNN
    # https://github.com/vdemichev/diann-rpackage
    # normalized and non normalized column

    # SAMPLE
    # column = Run

    # FILTERS
    # Run specific q-values for both precursors and proteins column = "PG.Q.value"
    # Global q-values for both precursor and proteins column = "Global.PG.Q.value"
    # PTM q-values reflecting the confidence in identifiying a specific prote column = "PTM.Q.value"
    # PTM side confidence score column = "PTM.Site.Confidence"
    # set thresholds for filters

    # VALUE
    # column = "PG.Quantity"

    # use normalized and non normalized column (normalize later?)
   

