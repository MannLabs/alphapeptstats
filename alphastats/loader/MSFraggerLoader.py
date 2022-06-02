from alphastats.loader import BaseLoader
import pandas as pd

#  Philosopher
# class name needs to be discussed whether MSFragger/Fragpipe/Philospher
class MSFraggerLoader(BaseLoader):
    def __init__(
        self,
        file,
        intensity_column="[experiment] MaxLFQ Intensity ",
        index_column="Protein",
        confidence_column="Protein Probability",
        sep="\t",
    ):  #
        """_summary_

        Args:
            file (_type_): _description_
            intensity_column (str, optional): _description_. Defaults to "[experiment] MaxLFQ Intensity ".
            index_column (str, optional): _description_. Defaults to "Protein".
            qvalue_column (str, optional): _description_. Defaults to "Protein Probability".
        """
        super.__init__(file, intensity_column, index_column, sep)
        self.confidence_column = confidence_column
        self.software = "MSFragger_Philosopher"


"""
- find way to deal with Indistinguishable Proteins, combine ProteinIDs?
"""
# https://github.com/Nesvilab/MSFragger/wiki/Interpreting-MSFragger-Output

#  SAMPLE
# columm Spectrum

# https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv
# Frag pipe
# https://github.com/Nesvilab/philosopher/wiki/Combined-protein-reports
#  ProteinProphet: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5863791/
