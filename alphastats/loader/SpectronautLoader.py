from alphastats.loader.BaseLoader import BaseLoader
import pandas as pd
import numpy as np
import logging



class SpectronautLoader(BaseLoader):
    """Loader for Spectronaut outputfiles 
    """

    def __init__(
        self,
        file,
        intensity_column="PG.Quantity",
        index_column="PG.ProteinGroups", 
        sample_column = "R.FileName",
        gene_names_column="PG.Genes",
        filter_qvalue = True,
        qvalue_cutoff = 0.01,
        sep="\t",
        **kwargs
    ):

        """_summary_

        Args:
            file (_type_): _description_
            intensity_column (str, optional): 'F.PeakArea'(default) uses not normalized peak area. 'NormalizedPeakArea' uses peak area normalized by Spectronaut.. Defaults to "F.PeakArea".
            index_column (str, optional): _description_. Defaults to "PG.ProteinGroups".
            gene_names_column (str, optional): _description_. Defaults to "PG.Genes".
            sep (str, optional): _description_. Defaults to "\t".
        """
        
        self.software = "Spectronaut"
        self.intensity_column = intensity_column
        self.index_column = index_column

        self._read_spectronaut_file(file=file, sep=sep)

        if filter_qvalue:
            self._filter_qvalue(qvalue_cutoff=qvalue_cutoff)

        self._reshape_spectronaut(sample_column=sample_column)
        
        if gene_names_column in self.rawinput.columns.to_list():
            self.gene_names = gene_names_column

    def _reshape_spectronaut(self, sample_column):
        self.rawdata["sample"] = self.rawdata[sample_column] + "_" + self.intensity_column
        df = self.rawdata[self.intensity_column, self.index_column, "sample"].drop_duplicates()
        self.rawdata = df.pivot(columns='sample', index=self.index_column, values=self.intensity_column)
        
        self.intensity_column = "[sample]_" + self.intensity_column

    def _filter_qvalue(self, qvalue_cutoff):
        if "EG.Qvalue" not in self.rawinput.columns.to_list():
            raise Warning("Column EG.Qvalue not found in file. File will not be filtered according to q-value.")
        
        rows_before_filtering = self.rawdata.shape[0]
        self.rawdata = self.rawdata[self.rawdata["EG.Qvalue"] < qvalue_cutoff]
        rows_after_filtering = self.rawdata.shape[0]

        rows_removed = rows_before_filtering - rows_after_filtering
        logging.info(f"{rows_removed} identification with a qvalue below {qvalue_cutoff} have been removed")
        
    
    def _read_spectronaut_file(self, file, sep):
        # some spectronaut files include european decimal separators
        df = pd.read_csv(file, sep=sep)

        if df[self.intensity_column].dtype != np.float64:
            # load european
             df = pd.read_csv(file, sep=sep, decimal=",")

        if df[self.intensity_column].dtype != np.float64:
            raise ValueError(f"Error in file format. {self.intensity_column} does not contain float values (numbers).")

        self.rawdata = df
        
    

#filter_with_Qvalue	
#TRUE(default) will filter out the intensities that have greater than qvalue_cutoff in EG.Qvalue column. Those intensities will be replaced with zero and will be considered as censored missing values for imputation purpose.

#qvalue_cutoff	
#Cutoff for EG.Qvalue. default is 0.01.

# Protein Level
# PG.Quantity
# PG.ProteinGroups

# Peptide Level
# F.PeakArea
# PEP.StrippedSequence