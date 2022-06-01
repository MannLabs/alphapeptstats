from alphastats.loader import BaseLoader
import pandas as pd

# Philosopher
class FragPipeLoader(BaseLoader):
    # https://github.com/Nesvilab/MSFragger/wiki/Interpreting-MSFragger-Output

    # SAMPLE
    # columm Spectrum

    # https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv
    # Frag pipe
    # https://github.com/Nesvilab/philosopher/wiki/Combined-protein-reports
    # ProteinProphet: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5863791/
    def __init__(
        self,
        file,
        intensity_column =  "[experiment] MaxLFQ Intensity ",
        index_column = "Protein",
        qvalue_column = "Protein Probability",
        sep = "\t"): # 
        """_summary_

        Args:
            file (_type_): _description_
            intensity_column (str, optional): _description_. Defaults to "[experiment] MaxLFQ Intensity ".
            index_column (str, optional): _description_. Defaults to "Protein".
            qvalue_column (str, optional): _description_. Defaults to "Protein Probability".
        """
        super.__init__(file, intensity_column, index_column, sep)
        self.qvalue_column = qvalue_column
        #self.filter_column = "Indistinguishable Proteins"? "Protein Probability"
        self.software = "MSFragger_Philosopher"