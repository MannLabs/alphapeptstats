from pyteomics import mztab
from alphastats.loader.BaseLoader import BaseLoader

class mzTabLoader(BaseLoader):
    def __init__(self, file, intensity_column: str="protein_abundance_[sample]", index_column:str="accession"):
        """Load mzTab file. Will add contamination column for further analysis.

        Args:
            file (str): path to mzTab file.
            intensity_column (str, optional):  columns where the intensity of the proteins are given.. Defaults to "protein_abundance_[sample]".
            index_column (str, optional): column indicating the protein groups.  Defaults to "accession".
        """
        self.filter_columns = []
        self.gene_names = None
        self.intensity_column = intensity_column
        self.index_column = index_column
        self.confidence_column = None
        self.evidence_df = None
        self.gene_names = None
        self._load_protein_table(file=file)
        self._add_contamination_column()


    def _load_protein_table(self, file):
        tables = mztab.MzTab(file)
        self.rawinput = tables.protein_table
        self.mztab_metadata = tables.metadata
        self.software = tables.protein_table.search_engine[0]
        