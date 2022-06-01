from operator import index
from alphastats.loader import BaseLoader
import pandas as pd
import numpy as np

class AlphaPeptLoader(BaseLoader):

    def __init__(self,
                file,
        intensity_column = "[experiment]_LFQ",
        index_column = "Unnamed: 0",
        sep = ","
       ):
        """_summary_

        Args:
            file (_type_): AlphaPept output, either results_proteins.csv file or the hdf_file with the protein_table given
            intensity_column (str, optional): columns where the intensity of the proteins are given. Defaults to "[experiment]_LFQ".
            index_column (str, optional): column indicating the protein groups. Defaults to "Unnamed: 0".
            sep (str, optional): file separation of file. Defaults to ",".
        """

        if file.endswith(".hdf"):
            self.load_hdf_protein_table(file = file) 
        else: 
            self.rawdata = pd.read_csv(file, sep = sep)

        self.intensity_column = intensity_column
        self.index_column = index_column
        # add contamination column "Reverse"
        self.add_contamination_column()
    
    def load_hdf_protein_table(self, file):
        """_summary_

        Args:
            file (_type_): _description_
        """
        self.rawdata = pd.read_hdf(file,  'protein_table')

    def add_contamination_column(self):
        """adds column 'Reverse' to the rawdata for filtering
        """
        self.rawdata["Reverse"] = np.where(
        self.rawdata[self.index_column].str.contains("REV_"), 
        True, 
        False
        )
        self.filter_column = ["Reverse"]
       
       


  # https://mannlabs.github.io/alphapept/file_formats.html#Output-Files


