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
        sep="\t"
    ):
        """Loads Spectronaut output. Will add contamination column for further analysis.

        Args:
            file (str): path to Spectronaut outputfile or pandas.DataFrame 
            intensity_column (str, optional): columns where the intensity of the proteins are given. Defaults to "PG.Quantity".
            index_column (str, optional): column indicating the protein groups. Defaults to "PG.ProteinGroups".
            sample_column (str, optional): column that contains sample names used for downstream analysis. Defaults to "R.FileName".
            gene_names_column (str, optional): column with gene names. Defaults to "PG.Genes".
            filter_qvalue (bool, optional): will filter out the intensities that have greater than qvalue_cutoff in EG.Qvalue column. Those intensities will be replaced with zero and will be considered as censored missing values for imputation purpose.. Defaults to True.
            qvalue_cutoff (float, optional): cut off vaéie. Defaults to 0.01.
            sep (str, optional): file separation of file. Defaults to "\t".
        """
        
        self.software = "Spectronaut"
        self.intensity_column = intensity_column
        self.index_column = index_column
        self.confidence_column = None
        self.filter_columns = []
        self.evidence_df = None 
        self.gene_names = None

        self._read_spectronaut_file(file=file, sep=sep)

        if filter_qvalue:
            self._filter_qvalue(qvalue_cutoff=qvalue_cutoff)

        self._reshape_spectronaut(sample_column=sample_column, gene_names_column=gene_names_column)
        self._add_contamination_column()
              

    def _reshape_spectronaut(self, sample_column, gene_names_column):
        """
        other proteomics softwares use a wide format (column for each sample)
        reshape to a wider format
        """
        self.rawinput["sample"] = self.rawinput[sample_column] + "_" + self.intensity_column
        
        indexing_columns = [self.index_column]
        
        if gene_names_column in self.rawinput.columns.to_list():
            self.gene_names = gene_names_column
            indexing_columns += [self.gene_names]
        
        keep_columns = [self.intensity_column, "sample"] + indexing_columns
        
        df = self.rawinput[keep_columns].drop_duplicates()
        df = df.pivot(columns='sample', index=indexing_columns, values=self.intensity_column)
        df.reset_index(inplace=True)
        
        self.rawinput = df
        
        self.intensity_column = "[sample]_" + self.intensity_column

    def _filter_qvalue(self, qvalue_cutoff):
        if "EG.Qvalue" not in self.rawinput.columns.to_list():
            raise Warning("Column EG.Qvalue not found in file. File will not be filtered according to q-value.")
        
        rows_before_filtering = self.rawinput.shape[0]
        self.rawinput = self.rawinput[self.rawinput["EG.Qvalue"] < qvalue_cutoff]
        rows_after_filtering = self.rawinput.shape[0]

        rows_removed = rows_before_filtering - rows_after_filtering
        logging.info(f"{rows_removed} identification with a qvalue below {qvalue_cutoff} have been removed")
        
    
    def _read_spectronaut_file(self, file, sep):
        # some spectronaut files include european decimal separators
        if isinstance(file, pd.DataFrame):
            df = file
        else:
            df = pd.read_csv(file, sep=sep, low_memory=False)

            if df[self.intensity_column].dtype != np.float64:
                # load european
                df = pd.read_csv(file, sep=sep, decimal=",")

        self.rawinput = df
        
    

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

