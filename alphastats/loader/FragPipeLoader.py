from alphastats.loader.BaseLoader import BaseLoader
import pandas as pd
from typing import Union

# Philosopher
# class name needs to be discussed whether MSFragger/Fragpipe/Philospher
class FragPipeLoader(BaseLoader):
    """Loader for FragPipe-Philosopheroutputfiles
    https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv
    """

    def __init__(
        self,
        file:Union[str, pd.DataFrame],
        intensity_column:str="[sample] MaxLFQ Intensity ",
        index_column:str="Protein",
        gene_names_column:str="Gene Names",
        confidence_column:str="Protein Probability",
        sep:str="\t",
        replace_zero_with_nan:bool = True,
        **kwargs
    ):
        """Loads FragPipe/Philosopher output: combined_protein.tsv
        Args:
            file (str, pd.DataFryame): FragPipe output, combined_protein.tsv.
            intensity_column (str, optional): columns where the intensity of the proteins are given. Defaults to "[sample] MaxLFQ Intensity ".
            index_column (str, optional): column indicating the protein groups. Defaults to "Protein".
            replace_zero_with_nan (bool, optional): whether zero values should be replaced with NaN when loading the data. Defaults to True.
            sep (str/optional): file separation. Defaults to "\t".
        """

        super().__init__(file, intensity_column, index_column, sep)

        if gene_names_column in self.rawinput.columns.to_list():
            self.gene_names = gene_names_column

        self.replace_zero_with_nan = replace_zero_with_nan
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
