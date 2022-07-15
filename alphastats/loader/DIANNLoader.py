from alphastats.loader.BaseLoader import BaseLoader
import pandas as pd


class DIANNLoader(BaseLoader):
    """Loader for DIA-NN output files
     https://github.com/vdemichev/DiaNN
     """

    def __init__(
        self,
        file,
        intensity_column="[sample]",
        index_column="Protein.Group",
        sep="\t",
        **kwargs
    ):
        """Import DIA-NN output data report.pg_matrix.tsv

        Args:
            file (_type_): DIA-NN output file report.pg_matrix.tsv
            intensity_column (str, optional): columns containing the intensity column for each experiment. Defaults to "[experiment]".
            index_column (str, optional): column with the Protein IDs. Defaults to "Protein.Group".
            sep (str, optional): file separation of the input file. Defaults to "\t".
        """

        super().__init__(file, intensity_column, index_column, sep)
        self.software = "DIA-NN"
        self.add_tag_to_sample_columns()
        self.add_contamination_column()

    def add_tag_to_sample_columns(self):
        # when creating matrix sample columns wont be found when it is only specified as [experiment]
        # TODO this is very fragile as changes in column names can break this
        no_sample_column = [
            "PG.Q.value",
            "Global.PG.Q.value",
            "PTM.Q.value",
            "PTM.Site.Confidence",
            "PG.Quantity",
            "Protein.Group",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "First.Protein.Description",
            "contamination_library"
        ]
        self.rawdata.columns = [
            str(col) + "_Intensity" if col not in no_sample_column else str(col)
            for col in self.rawdata.columns
        ]
        self.intensity_column = "[sample]_Intensity"


"""
- older versions of DIA-NN don't contain MaxLFQ normalization here Precursor.Normalised should be taken
- Several output files use: report.pg_matrix.tsv
- else report.tsv could be used but is long data
- wont work with older versions of DIA-NN

"""

# def convert_to_wide_data(self, file, qvalue_threshold):
#  FDR needs to be filtered before converting to wide format
# pass

# https://github.com/vdemichev/DiaNN
# https://github.com/vdemichev/diann-rpackage
# normalized and non normalized column

# SAMPLE
# column = Run

# FILTERS
# Run specific q-values for both precursors and proteins column = "PG.Q.value"
#  Global q-values for both precursor and proteins column = "Global.PG.Q.value"
# PTM q-values reflecting the confidence in identifiying a specific prote column = "PTM.Q.value"
#  PTM side confidence score column = "PTM.Site.Confidence"
# set thresholds for filters

# VALUE
# column = "PG.Quantity"

# use normalized and non normalized column (normalize later?)
# qvalue_column = ["PG.Q.Value", "Q.Value"],
# qvalue_threshold = 0.01,
# load_matrix = False): # add option so you can load matrix from diann-rpackage

# self.rawdata = self.convert_to_wide_data(file = file, qvalue_threshold = qvalue_threshold)
