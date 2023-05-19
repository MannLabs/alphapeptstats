
from alphastats.loader.BaseLoader import BaseLoader

import pandas as pd
from typing import Union

class GenericLoader(BaseLoader):
    def __init__(self, file:Union[str, pd.DataFrame], intensity_column:list, index_column:str, sep:str):
        """Generic Loader for you proteomics data

        Args:
            file (Union[str, pd.DataFrame]): path to your proteomics file or pandas.DataFrame
            intensity_column (list): list of samples with intensity
            index_column (str): column with Protein IDs or Gene names, used for indexing
            sep (str): file separation
        """
        if isinstance(file, pd.DataFrame):
            self.rawinput = file
        else:
            self.rawinput = pd.read_csv(file, sep=sep, low_memory=False)
        self.intensity_column = "[sample]"
        self.intensity_column_list = intensity_column
        self.index_column = index_column
        self.filter_columns = []
        self.confidence_column = None
        self.software = "Generic"
        self.evidence_df = None
        self.gene_names = None
        self.ptm_df = None
        self._add_contamination_column()
        self._check_if_columns_are_present()
        self._read_all_columns_as_string()

    def _extract_sample_names(self, metadata:pd.DataFrame, sample_column:str):
        sample_names = metadata[sample_column].to_list()
        
        for intensity_column in self.intensity_column_list:
            for sample in sample_names:
                if sample in intensity_column:
                    sample_structure = intensity_column.replace(sample, "[sample]")
        
        self.intensity_column = sample_structure
        return sample_structure

