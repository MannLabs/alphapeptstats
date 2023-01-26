from alphastats.loader.BaseLoader import BaseLoader
import pandas as pd
import numpy as np
import logging


class AlphaPeptLoader(BaseLoader):
    """Loader for AlphaPept outputfiles
    https://github.com/MannLabs/alphapept
    """

    def __init__(
        self,
        file,
        intensity_column="[sample]_LFQ",
        index_column="Unnamed: 0",  # column name to be changed
        sep=",",
        **kwargs
    ):
        """Loads Alphapept output: results_proteins.csv. Will add contamination column for further analysis.

        Args:
            file (str): AlphaPept output, either results_proteins.csv file or the hdf_file with the protein_table given
            intensity_column (str, optional): columns where the intensity of the proteins are given. Defaults to "[sample]_LFQ".
            index_column (str, optional): column indicating the protein groups. Defaults to "Unnamed: 0".
            sep (str, optional): file separation of file. Defaults to ",".
        """

        if file.endswith(".hdf"):
            self._load_hdf_protein_table(file=file)
        else:
            self.rawinput = pd.read_csv(file, sep=sep)

        self.intensity_column = intensity_column
        self.index_column = index_column
        self.filter_columns = []
        self.confidence_column = None
        self.software = "AlphaPept"
        self.evidence_df = None
        self.gene_names = None
        # add contamination column "Reverse"
        self._add_contamination_reverse_column()
        self._add_contamination_column()
        #  make ProteinGroup column
        self.rawinput["ProteinGroup"] = self.rawinput[self.index_column].map(
            self._standardize_protein_group_column
        )
        self.index_column = "ProteinGroup"

    def _load_hdf_protein_table(self, file):
        self.rawinput = pd.read_hdf(file, "protein_table")

    def _add_contamination_reverse_column(self):
        """adds column 'Reverse' to the rawinput for filtering"""
        self.rawinput["Reverse"] = np.where(
            self.rawinput[self.index_column].str.contains("REV_"), True, False
        )
        self.filter_columns = ["Reverse"]
        logging.info(
            "Proteins with a peptide derived from the reversed part of the decoy database have been annotated"
            "These proteins should be filtered with `DataSet.preprocess(remove_contaminations=True)` later."
        )

    def _standardize_protein_group_column(self, entry):
        #  make column with ProteinGroup to make comparison between softwares possible
        #  'sp|P0DMV9|HS71B_HUMAN,sp|P0DMV8|HS71A_HUMAN', -> P0DMV9;P0DMV8

        # TODO this needs a more beautiful and robuster solution
        # split proteins into list
        proteins = entry.split(",")
        protein_id_list = []
        for protein in proteins:
            # 'sp|P0DMV9|HS71B_HUMAN,sp|P0DMV8|HS71A_HUMAN',
            if "|" in protein:
                fasta_header_split = protein.split("|")
            else:
                fasta_header_split = protein
            if isinstance(fasta_header_split, str):
                #  'ENSEMBL:ENSBTAP00000007350',
                if "ENSEMBL:" in fasta_header_split:
                    protein_id = fasta_header_split.replace("ENSEMBL:", "")
                else:
                    protein_id = fasta_header_split
            else:
                protein_id = fasta_header_split[1]
            protein_id_list.append(protein_id)
        protein_id_concentate = ";".join(protein_id_list)
        # ADD REV to the protein ID, else there will be duplicates in the ProteinGroup column
        if "REV_" in entry:
            protein_id_concentate = "REV_" + protein_id_concentate
        return protein_id_concentate


# https://mannlabs.github.io/alphapept/file_formats.html#Output-Files
