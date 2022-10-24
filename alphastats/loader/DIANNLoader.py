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
        self._add_tag_to_sample_columns()
        self._add_contamination_column()

    def _add_tag_to_sample_columns(self):
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
            "contamination_library",
        ]
        self.rawinput.columns = [
            str(col) + "_Intensity" if col not in no_sample_column else str(col)
            for col in self.rawinput.columns
        ]
        self.intensity_column = "[sample]_Intensity"
