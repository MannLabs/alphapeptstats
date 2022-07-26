from ast import Not
from cmath import isinf
from importlib.abc import Loader
from multiprocessing.sharedctypes import Value
from random import sample
import re
import pandas as pd
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
import os
import numpy as np
import logging
from sklearn_pandas import DataFrameMapper
import warnings
from alphastats.DataSet_Plot import Plot
from alphastats.DataSet_Preprocess import Preprocess
from alphastats.DataSet_Statistics import Statistics
from alphastats.utils import LoaderError
# remove warning from openpyxl
# only appears on mac
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


class DataSet(Preprocess, Statistics, Plot):
    """Analysis Object
    """

    def __init__(self, loader, metadata_path: str = None, sample_column=None):
        """Create proteinObjet/AnalysisObject

        Args:
            loader (_type_): loader of class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader
            metadata_path (str, optional): path to metadata file. Defaults to None.
            sample_column (_type_, optional): column in metadata file indicating the sample IDs. Defaults to None.
        """
        self.check_loader(loader=loader)
        #  load data from loader object
        self.loader = loader
        self.rawdata = loader.rawdata
        self.software = loader.software
        self.index_column = loader.index_column
        self.intensity_column = loader.intensity_column
        self.filter_columns = loader.filter_columns

        # include filtering before
        self.create_matrix()
        self.check_matrix_values()
        self.metadata = None
        if metadata_path:
            self.load_metadata(file_path=metadata_path, sample_column=sample_column)

        # save preprocessing settings
        self.preprocessing = None
        # update normalization when self.matrix is normalized, filtered
        self.normalization, self.imputation, self.contamination_filter = (
            "Data is not normalized.",
            "Data is not imputed.",
            "Contaminations have not been removed.",
        )

    def check_loader(self, loader):
        """Checks if the Loader is from class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader

        Args:
            loader : loader
        """
        if not isinstance(
            loader, (AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader)
        ):
            raise LoaderError(

                "loader must be from class: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader. ADD LINK TO DOCUMENTATION"
            )

        if not isinstance(loader.rawdata, pd.DataFrame) or loader.rawdata.empty:
            raise ValueError(
                "Error in rawdata, consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader"
            )

        if not isinstance(loader.index_column, str):
            raise ValueError(
                "Invalid index_column: consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader"
            )

    def check_matrix_values(self):
        if np.isinf(self.mat).values.sum() > 0:
            logging.warning("Data contains infinite values.")

    def create_matrix(self):
        """Creates a matrix of the Outputfile, with columns displaying features (Proteins) and
        rows the samples.
        """

        regex_find_intensity_columns = self.intensity_column.replace("[sample]", ".*")

        df = self.rawdata
        df = df.set_index(self.index_column)
        df = df.filter(regex=(regex_find_intensity_columns), axis=1)
        # remove Intensity so only sample names remain
        substring_to_remove = regex_find_intensity_columns.replace(".*", "")
        df.columns = df.columns.str.replace(substring_to_remove, "")
        # transpose dataframe
        self.mat = df.transpose()
        # reset preproccessing info
        self.normalization, self.imputation, self.contamination_filter = (
            "Data is not normalized",
            "Data is not imputed",
            "Contaminations have not been removed.",
        )

    def load_metadata(self, file_path, sample_column):
        """Load metadata either xlsx, txt, csv or txt file

        Args:
            file_path (str): path to metadata file
            sample_column (str): column name with sample IDs
        """
        #  loading file needs to be more beautiful
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            # find robust way to detect file format
            # else give file separation as variable
        elif file_path.endswith(".txt") or file_path.endswith(".tsv"):
            df = pd.read_csv(file_path, delimiter="\t")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = None
            logging.warning(
                "WARNING: Metadata could not be read. \nMetadata has to be a .xslx, .tsv, .csv or .txt file"
            )
            return
        if df is not None and sample_column not in df.columns:
            logging.error(f"sample_column: {sample_column} not found in {file_path}")
        df.columns = df.columns.str.replace(sample_column, "sample")
        # check whether sample labeling matches protein data
        #  warnings.warn("WARNING: Sample names do not match sample labelling in protein data")
        self.metadata = df

    def summary(self):
        # print summary
        # TODO look at keras model.summary()
        pass
