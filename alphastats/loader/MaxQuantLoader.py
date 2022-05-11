from alphastats.loader import BaseLoader

class MaxQuantLoader(BaseLoader):
    def __init__(self,
        intensity_column = "LFQ intentsity ",
        index_column = "Protein IDs"):
        self.software = "MaxQuant"

        
    def preprocess_contamination(self):
        contaminantion_columns = ["Only identified by site", "Reverse", "Potential contaminant"]
        # remove columns where +
        pass