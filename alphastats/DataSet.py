from typing import List, Union, Dict, Optional

import pandas as pd
import numpy as np
import logging
import warnings
import plotly

from alphastats import BaseLoader


from alphastats.DataSet_Plot import Plot
from alphastats.DataSet_Preprocess import Preprocess
from alphastats.DataSet_Pathway import Enrichment
from alphastats.DataSet_Statistics import Statistics
from alphastats.utils import LoaderError

plotly.io.templates["alphastats_colors"] = plotly.graph_objects.layout.Template(
    layout=plotly.graph_objects.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            "#009599",
            "#005358",
            "#772173",
            "#B65EAF",  # pink
            "#A73A00",
            "#6490C1",
            "#FF894F",
            "#2B5E8B",
            "#A87F32",
        ],
    )
)

plotly.io.templates.default = "simple_white+alphastats_colors"


class DataSet(Statistics, Plot, Enrichment):
    """Analysis Object"""

    def __init__(self, loader, metadata_path=None, sample_column=None):
        """Create DataSet

        Args:
            loader (_type_): loader of class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader, SpectronautLoader
            metadata_path (str, optional): path to metadata file. Defaults to None.
            sample_column (str, optional): column in metadata file indicating the sample IDs. Defaults to None.

        """
        self._check_loader(loader=loader)

        self.rawinput: pd.DataFrame = loader.rawinput
        self.software: str = loader.software
        self.index_column: str = loader.index_column
        self.intensity_column: Union[str, list] = loader.intensity_column
        self.filter_columns: List[str] = loader.filter_columns
        self.evidence_df: pd.DataFrame = loader.evidence_df
        self.gene_names: str = loader.gene_names

        self.metadata: Optional[pd.DataFrame] = None
        self.sample: Optional[str] = None

        # include filtering before
        self.create_matrix()
        self._check_matrix_values()

        if metadata_path is not None:
            self.sample = sample_column
            self.load_metadata(file_path=metadata_path)
            self._remove_misc_samples_in_metadata()

        else:
            self._create_metadata()

        if loader == "Generic":
            intensity_column = loader._extract_sample_names(
                metadata=self.metadata, sample_column=self.sample
            )
            self.intensity_column = intensity_column

        # save preprocessing settings
        self.preprocessing_info: Dict = self._save_dataset_info()
        self.preprocessed: bool = False

        print("DataSet has been created.")
        self.overview()

    def preprocess(
        self,
        log2_transform: bool = True,
        remove_contaminations: bool = False,
        subset: bool = False,
        data_completeness: float = 0,
        normalization: str = None,
        imputation: str = None,
        remove_samples: list = None,
        **kwargs,
    ) -> None:
        """A wrapper for the preprocess() method, see documentation in Preprocess.preprocess()."""
        pp = Preprocess(
            self.filter_columns,
            self.rawinput,
            self.index_column,
            self.sample,
            self.metadata,
            self.preprocessing_info,
            self.mat,
        )

        self.mat, self.metadata, self.preprocessing_info = pp.preprocess(
            log2_transform,
            remove_contaminations,
            subset,
            data_completeness,
            normalization,
            imputation,
            remove_samples,
            **kwargs,
        )
        self.preprocessed = True

    def reset_preprocessing(self):
        """Reset all preprocessing steps"""
        self.create_matrix()
        # TODO fix bug: metadata is not reset here
        print("All preprocessing steps are reset.")

    def batch_correction(self, batch: str) -> None:
        pp = Preprocess(
            self.filter_columns,
            self.rawinput,
            self.index_column,
            self.sample,
            self.metadata,
            self.preprocessing_info,
            self.mat,
        )
        self.mat = pp.batch_correction(batch)

    def _create_metadata(self):
        samples = list(self.mat.index)
        self.metadata = pd.DataFrame({"sample": samples})
        self.sample = "sample"

    def _check_loader(self, loader):
        """Checks if the Loader is from class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader

        Args:
            loader : loader
        """
        if not isinstance(loader, BaseLoader):
            raise LoaderError(
                "loader must be a subclass of BaseLoader, "
                f"got {loader.__class__.__name__}"
            )

        if not isinstance(loader.rawinput, pd.DataFrame) or loader.rawinput.empty:
            raise ValueError(
                "Error in rawinput, consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader, SpectronautLoader"
            )

        if not isinstance(loader.index_column, str):
            raise ValueError(
                "Invalid index_column: consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader, SpectronautLoader"
            )

    def _check_matrix_values(self):
        if np.isinf(self.mat).values.sum() > 0:
            logging.warning("Data contains infinite values.")

    def _remove_misc_samples_in_metadata(self):
        samples_matrix = self.mat.index.to_list()
        samples_metadata = self.metadata[self.sample].to_list()
        misc_samples = list(set(samples_metadata) - set(samples_matrix))
        if len(misc_samples) > 0:
            self.metadata = self.metadata[
                ~self.metadata[self.sample].isin(misc_samples)
            ]
            logging.warning(
                f"{misc_samples} are not described in the protein data and"
                "are removed from the metadata."
            )

    # TODO this is implemented in both preprocessing and here
    #  This is only needed in the DimensionalityReduction class and only if the step was not run during preprocessing.
    #  idea: replace the step in DimensionalityReduction with something like:
    #  mat = self.data.mat.loc[sample_names,:] after creating sample_names.
    def _subset(self):
        # filter matrix so only samples that are described in metadata are also found in matrix
        self.preprocessing_info.update(
            {"Matrix: Number of samples": self.metadata.shape[0]}
        )
        return self.mat[self.mat.index.isin(self.metadata[self.sample].tolist())]

    def create_matrix(self):
        """
        Creates a matrix of the Outputfile, with columns displaying features (Proteins) and
        rows the samples.
        """

        df = self.rawinput
        df = df.set_index(self.index_column)

        if isinstance(self.intensity_column, str):
            regex_find_intensity_columns = self.intensity_column.replace(
                "[sample]", ".*"
            )
            df = df.filter(regex=(regex_find_intensity_columns), axis=1)
            # remove Intensity so only sample names remain
            substring_to_remove = regex_find_intensity_columns.replace(".*", "")
            df.columns = df.columns.str.replace(substring_to_remove, "")

        else:
            df = df[self.intensity_column]

        # transpose dataframe
        mat = df.transpose()
        mat.replace([np.inf, -np.inf], np.nan, inplace=True)

        # remove proteins with only zero  # TODO this is re-done in preprocessing
        self.mat = mat.loc[:, (mat != 0).any(axis=0)]
        self.mat = self.mat.astype(float)

        # reset preproccessing info
        self.preprocessing_info = self._save_dataset_info()
        self.preprocessed = False
        self.rawmat = mat

    def load_metadata(self, file_path):
        """Load metadata either xlsx, txt, csv or txt file

        Args:
            file_path (str): path to metadata file
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        #  loading file needs to be more beautiful
        elif file_path.endswith(".xlsx"):
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="openpyxl",
                # message=r"/extension is not supported and will be removed/",  # this somehow does not work here?
            )
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
        if df is not None and self.sample not in df.columns:
            logging.error(f"sample_column: {self.sample} not found in {file_path}")

        # check whether sample labeling matches protein data
        #  warnings.warn("WARNING: Sample names do not match sample labelling in protein data")
        df.columns = df.columns.astype(str)
        self.metadata = df

    def _save_dataset_info(self):
        n_proteingroups = self.mat.shape[1]
        preprocessing_dict = {
            "Raw data number of Protein Groups": n_proteingroups,
            "Matrix: Number of ProteinIDs/ProteinGroups": self.mat.shape[1],
            "Matrix: Number of samples": self.mat.shape[0],
            "Intensity used for analysis": self.intensity_column,
            "Log2-transformed": False,
            "Normalization": None,
            "Imputation": None,
            "Contaminations have been removed": False,
            "Contamination columns": self.filter_columns,
            "Number of removed ProteinGroups due to contaminaton": 0,
            "Data completeness cut-off": 0,
        }
        return preprocessing_dict

    def overview(self):
        """Print overview of the DataSet"""
        dataset_overview = (
            "Attributes of the DataSet can be accessed using: \n"
            + "DataSet.rawinput:\t Raw Protein data.\n"
            + "DataSet.mat:\t\tProcessed data matrix with ProteinIDs/ProteinGroups as columns and samples as rows. All computations are performed on this matrix.\n"
            + "DataSet.metadata:\tMetadata for the samples in the matrix. Metadata will be matched with DataSet.mat when needed (for instance Volcano Plot)."
        )
        print(dataset_overview)
