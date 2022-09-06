import streamlit as st
import os
import pandas as pd
import datetime
import yaml
from typing import Union, Tuple
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.DataSet import DataSet
import logging

software_options = {
    "MaxQuant": {
        "import_file": "proteinGroups.txt",
        "intensity_column": ["LFQ intensity [sample]"],
        "index_column": ["Protein IDs"],
        "loader_function": MaxQuantLoader,
    },
    "AlphaPept": {
        "import_file": "results_proteins.csv or results.hdf",
        "intensity_column": ["[sample]_LFQ"],
        "index_column": ["Unnamed: 0"],
        "loader_function": AlphaPeptLoader,
    },
    "DIANN": {
        "import_file": "report.pg_matrix.tsv",
        "intensity_column": ["[sample]"],
        "index_column": ["Protein.Group"],
        "loader_function": DIANNLoader,
    },
    "Fragpipe": {
        "import_file": "combined_protein.tsv",
        "intensity_column": ["[sample] MaxLFQ Intensity "],
        "index_column": ["Protein"],
        "loader_function": FragPipeLoader,
    },
}


class ImportData:
    def print_software_import_info(self):
        import_file = software_options.get(self.software).get("import_file")
        string_output = f"Please upload {import_file} file from {self.software}."
        return string_output

    def select_columns_for_loaders(self):
        """
        select intensity and index column depending on software
        will be saved in session state
        """
        st.write("Select intensity columns for further analysis")
        st.selectbox(
            "Intensity Column",
            options=software_options.get(self.software).get("intensity_column"),
            key="intensity_column",
        )

        st.write("Select index column (with ProteinGroups) for further analysis")
        st.selectbox(
            "Index Column",
            options=software_options.get(self.software).get("index_column"),
            key="index_column",
        )

   # @st.cache(allow_output_mutation=True)
    def load_proteomics_data(self, uploaded_file, intensity_column, index_column):
        """load software file into loader object from alphastats
        """
        loader = software_options.get(self.software)["loader_function"](
            uploaded_file, intensity_column, index_column
        )
        return loader

    def read_uploaded_file_into_df(self, file):
        filename = file.name
        if filename.endswith(".xlsx"):
            df = pd.read_excel(file)
        elif filename.endswith(".txt") or filename.endswith(".tsv"):
            df = pd.read_csv(file, delimiter="\t")
        elif filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = None
            logging.warning(
                "WARNING: File could not be read. \nFile has to be a .xslx, .tsv, .csv or .txt file"
            )
            return
        return df

    def select_sample_column_metadata(self, df):
        st.write(
            f"Select column that contains sample IDs matching the sample names described"
            + f"in {software_options.get(self.software).get('import_file')}"
        )
        with st.form("sample_column"):
            st.selectbox("Sample Column", options=df.columns.to_list(), key = "sample_column")
            submitted = st.form_submit_button("Submit")
        if submitted:
            return True

    def display_file(self, uploaded_file):
        """display first 5 rows of uploaded file
        """
        if isinstance(uploaded_file, pd.DataFrame):
            df = uploaded_file
        else:
            df = self.read_uploaded_file_into_df(uploaded_file)
        st.dataframe(df.head(5))

    def upload_softwarefile(self):
        softwarefile = st.file_uploader(
            self.print_software_import_info(), key="softwarefile"
        )
        # import data
        if "softwarefile" in st.session_state:
            softwarefile_df = self.read_uploaded_file_into_df(softwarefile)
            # display head a protein data
            self.display_file(softwarefile_df)
            #  select intensity and index will be saved in session state
            #  other ways throw an error when loading the metadata for some reason
            self.select_columns_for_loaders()
            if (
                "intensity_column" in st.session_state
                and "index_column" in st.session_state
            ):
                self.loader = self.load_proteomics_data(
                    softwarefile_df,
                    intensity_column=st.session_state.intensity_column,
                    index_column=st.session_state.index_column,
                )

    def upload_metadatafile(self):
        metadatafile = st.file_uploader(
            "Upload metadata file. with information about your samples",
            key="metadatafile",
        )
        if st.button("Continue without metadata"):
            #  create dataset
            self.dataset = DataSet(loader=self.loader)
        else:
            if "metadatafile" in st.session_state:
                metadatafile_df = self.read_uploaded_file_into_df(metadatafile)
                # display metadata
                self.display_file(metadatafile_df)
                # pick sample column
                if self.select_sample_column_metadata(metadatafile_df):
                # create dataset
                    st.session_state["dataset"] = DataSet(
                    loader=self.loader,
                    metadata_path=metadatafile_df,
                    sample_column=st.session_state.sample_column
                )

    def import_data(self):
        st.write("# Import Data")

        #  Import Protein Data
        software = st.selectbox(
            "Select your Proteomics Software",
            options=["<select>", "MaxQuant", "AlphaPept", "DIANN", "Fragpipe"],
        )
        self.software = software

        if software != "<select>":
            self.upload_softwarefile()

        if self.loader is not None:
            self.upload_metadatafile()

        if "dataset" in st.session_state:
            self.get_column_names_metadata()
            self.load_plotting_options()
            
