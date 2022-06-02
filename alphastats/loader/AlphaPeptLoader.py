from operator import index
from alphastats.loader import BaseLoader
import pandas as pd
import numpy as np


class AlphaPeptLoader(BaseLoader):
    def __init__(
        self,
        file,
        intensity_column="[experiment]_LFQ",
        index_column="Unnamed: 0",  # column name to be changed
        sep=",",
    ):
        """_summary_

        Args:
            file (_type_): AlphaPept output, either results_proteins.csv file or the hdf_file with the protein_table given
            intensity_column (str, optional): columns where the intensity of the proteins are given. Defaults to "[experiment]_LFQ".
            index_column (str, optional): column indicating the protein groups. Defaults to "Unnamed: 0".
            sep (str, optional): file separation of file. Defaults to ",".
        """

        if file.endswith(".hdf"):
            self.load_hdf_protein_table(file=file)
        else:
            self.rawdata = pd.read_csv(file, sep=sep)

        self.intensity_column = intensity_column
        self.index_column = index_column
        # add contamination column "Reverse"
        self.add_contamination_column()
        #  make ProteinGroup column
        self.rawdata["ProteinGroup"] = self.rawdata[self.index_column].map(
            self.standardize_protein_group_column
        )

    def load_hdf_protein_table(self, file):
        """_summary_

        Args:
            file (_type_): _description_
        """
        self.rawdata = pd.read_hdf(file, "protein_table")

    def add_contamination_column(self):
        """adds column 'Reverse' to the rawdata for filtering
        """
        self.rawdata["Reverse"] = np.where(
            self.rawdata[self.index_column].str.contains("REV_"), True, False
        )
        self.filter_column = ["Reverse"]

    def standardize_protein_group_column(self, entry):
        #  make column with ProteinGroup to make comparison between softwares possible
        #  'sp|P0DMV9|HS71B_HUMAN,sp|P0DMV8|HS71A_HUMAN', -> P0DMV9;P0DMV8

        # This needs a more beautiful and robuster solution

        # split proteins into list
        proteins = entry.split(",")
        protein_id_list = list()
        for protein in proteins:
            # 'sp|P0DMV9|HS71B_HUMAN,sp|P0DMV8|HS71A_HUMAN',
            fasta_header_split = protein.split("|")
            if len(fasta_header_split) == 1:
                #  'ENSEMBL:ENSBTAP00000007350',
                if "ENSEMBL:" in fasta_header_split:
                    protein_id = fasta_header_split.replace("ENSEMBL:", "")
                # if only protein id is given
                else:
                    protein_id = fasta_header_split[0]
            else:
                protein_id = fasta_header_split[1]
            protein_id_list.append(protein_id)
        protein_id_concentate = ";".join(protein_id_list)
        return protein_id_concentate


# https://mannlabs.github.io/alphapept/file_formats.html#Output-Files
