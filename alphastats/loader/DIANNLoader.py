from inspect import getfile
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
            file (str): DIA-NN output file report.pg_matrix.tsv
            intensity_column (str, optional): columns containing the intensity column for each experiment. Defaults to "[experiment]".
            index_column (str, optional): column with the Protein IDs. Defaults to "Protein.Group".
            sep (str, optional): file separation of the input file. Defaults to "\t".
        """

        super().__init__(file, intensity_column, index_column, sep)
        self.software = "DIA-NN"
        self.no_sample_column = [
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
            "contamination_library",
        ]
        self._remove_filepath_from_name()
        self._add_tag_to_sample_columns()
        self._add_contamination_column()

    def _add_tag_to_sample_columns(self):
        """
        when creating matrix sample columns wont be found when it is only specified as [experiment]
        so tag will be added
        """
        # TODO this is very fragile as changes in column names can break this

        self.rawinput.columns = [
            str(col) + "_Intensity" if col not in self.no_sample_column else str(col)
            for col in self.rawinput.columns
        ]
        self.intensity_column = "[sample]_Intensity"

    @staticmethod
    def _split_path(file_path):
        """
        split file path for windows and macOS
        """
        # try:
        if "/" in file_path:
            file = file_path.split("/")[-1]

        else:
            file = file_path.split("\\")[-1]

        # windows path can cause error
        # except SyntaxError:
        #     file = file_path

        return file

    def _remove_filepath_from_name(self):
        """
        split filepath so only filename is used for analysis
        """

        self.rawinput.columns = [
            self._split_path(col) if col not in self.no_sample_column else str(col)
            for col in self.rawinput.columns
        ]
