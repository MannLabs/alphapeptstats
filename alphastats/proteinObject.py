from ast import Not
from cmath import isinf
from multiprocessing.sharedctypes import Value
from random import sample
import re
import pandas as pd
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from data_cache import pandas_cache
import os
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import scipy.stats
import dash_bio
import numpy as np
import logging
import sys
from sklearn_pandas import DataFrameMapper
import sklearn


class proteinObject:
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
        self.metadata = None

        if metadata_path:
            self.load_metadata(file_path=metadata_path, sample_column=sample_column)

        self.experiment_type = None
        self.data_format = None
        # save preprocessing settings
        self.preprocessing = None
        # update normalization when self.matrix is normalized, filtered
        self.normalization = None
        self.removed_protein_groups = None
        self.imputation = "Data is not imputed."
        self.removed_protein_groups = None

    def check_loader(self, loader):
        """Checks if the Loader is from class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader

        Args:
            loader (_type_): loader
        """
        if not isinstance(
            loader, (AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader)
        ):
            logging.error(
                "loader must be from class: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader. ADD LINK TO DOCUMENTATION"
            )

        # if not isinstance(loader.rawdata, pd.DataFrame) or loader.rawdata.empty:
        #    logging.error(
        #        "Error in rawdata, consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader"
        #    )
        # if not isinstance(loader.index_column, str):
        #    logging.error(
        #        "Invalid index_column: consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader"
        #    )

    def create_matrix(self):
        """Creates a matrix of the Outputfile, with columns displaying features(Proteins) and
        rows the samples.
        """
        regex_find_intensity_columns = self.intensity_column.replace(
            "[experiment]", ".*"
        )
        df = self.rawdata
        df = df.set_index(self.index_column)
        df = df.filter(regex=(regex_find_intensity_columns), axis=1)
        # remove Intensity so only sample names remain
        substring_to_remove = regex_find_intensity_columns.replace(".*", "")
        df.columns = df.columns.str.replace(substring_to_remove, "")
        # transpose dataframe
        self.mat = df.transpose()
        #  reset preproccessing info
        self.normalization = None
        self.imputation = None
        self.removed_protein_groups = None

    def preprocess_exclude_sampels(self, sample_list):
        # exclude samples for analysis
        pass

    def preprocess_print_info(self):
        """Print summary of preprocessing steps
        """
        n_proteins = self.rawdata.shape[0]
        n_samples = self.rawdata.shape[1]  #  remove filter columns etc.
        text = (
            f"Preprocessing: \nThe raw data contains {str(n_proteins)} Proteins and "
            + f"{str(n_samples)} samples.\n {str(len(self.removed_protein_groups))}"
            + f"rows with Proteins/Protein Groups have been removed."
        )
        if self.normalization is None:
            normalization_text = (
                f"Data has not been normalized, or has already been normalized by "
                f"{self.software}.\n"
            )
        else:
            normalization_text = (
                f"Data has been normalized using {self.normalization}.\n"
            )
        imputation_text = self.imputation
        preprocessing_text = text + normalization_text + imputation_text
        print(preprocessing_text)

    def preprocess_filter(self):
        """Removes all observations, that were identified as contaminations. 
        """
        if self.filter_columns is None:
            logging.info("No columns to filter.")
            return
        #  print column names with contamination
        logging.info(
            f"Contaminations indicated in following columns: {self.filter_columns} are removed"
        )
        protein_groups_to_remove = self.rawdata[
            (self.rawdata[self.filter_columns] == True).any(1)
        ][self.index_column].tolist()
        # remove columns with protin groups
        self.mat = self.mat.drop(protein_groups_to_remove, axis=1)
        self.removed_protein_groups = protein_groups_to_remove
        logging.info(
            f"{str(len(protein_groups_to_remove))} observations have been removed."
        )

    def preprocess_impute(self, method):
        """
        Impute Data
        For more information visit:
        SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
        k-Nearest Neighbors Imputation: https://scikit-learn.org/stable/modules/impute.html#impute

        Args:
            method (str): method to impute data: either "mean", "median" or "knn"
        """
        # Imputation using the mean
        if method == "mean":
            imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="mean")
            imputation_array = imp.fit_transform(self.mat.values)
        if method == "median":
            imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="median")
            imputation_array = imp.fit_transform(self.mat.values)
        # Imputation using Nearest neighbors imputation
        # default n_neighbors is 2  - should this optional?
        if method == "knn":
            imp = sklearn.impute.KNNImputer(n_neighbors=2, weights="uniform")
            imputation_array = imp.fit_transform(self.mat.values)

        self.mat = pd.DataFrame(
            imputation_array, index=self.mat.index, columns=self.mat.columns
        )
        self.imputation = f"Missing values were imputed using the {method}."

    def preprocess_normalization(self, method):
        """
        Normalize data using either zscore, quantile or linear (using l2 norm) Normalization.
        Z-score normalization equals standaridzation using StandardScaler: 
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        For more information visit.
        Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
        
        Args:
            method (str): method to normalize data: either "zscore", "quantile", "linear"
        """
        # zscore normalization == standardization
        if method == "zscore":
            scaler = sklearn.preprocessing.StandardScaler()
            normalized_array = scaler.fit_transform(self.mat.values)

        if method == "quantile":
            qt = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=10, random_state=0
            )
            normalized_array = qt.fit_transform(self.mat.values)

        if method == "linear":
            normalized_array = sklearn.preprocessing.normalize(
                self.mat.values, norm="l2"
            )

        self.mat = pd.DataFrame(
            normalized_array, index=self.mat.index, columns=self.mat.columns
        )
        self.normalization = f"Data has been normalized using {method} normalization"

    def preprocess(
        self,
        remove_contaminations=False,
        normalization=None,
        imputation=None,
        remove_samples=None,
        qvalue=0.01,
    ):
        """Preprocess Protein data

        Args:
            remove_contaminations (bool, optional): remove ProteinGroups that are identified as contamination
            . Defaults to False.
            normalization (str, optional): method to normalize data: either "zscore", "quantile", "linear". Defaults to None.
            remove_samples (_type_, optional): _description_. Defaults to None.
            imputation (str, optional):  method to impute data: either "mean", "median" or "knn". Defaults to None.
            qvalue (float, optional): _description_. Defaults to 0.01.

        Raises:
            NotImplementedError: _description_
        """
        if remove_contaminations:
            self.preprocess_filter()
        if normalization is not None:
            self.preprocess_normalization(method=normalization)
        if imputation is not None:
            self.preprocess_impute(method=imputation)
        if remove_samples is not None:
            raise NotImplementedError

    def load_metadata(self, file_path, sample_column):
        """Load metadata

        Args:
            file_path (str): path to metadata file
            sample_column (str): column with sample IDs
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
            logging.warn(
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

    def calculate_ttest_fc(self, column, group1, group2):
        """_summary_

        Args:
            column (_type_): _description_
            group1 (_type_): _description_
            group2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        # get samples names of two groupes
        group1_samples = self.metadata[self.metadata[column] == group1][
            "sample"
        ].tolist()
        group2_samples = self.metadata[self.metadata[column] == group2][
            "sample"
        ].tolist()
        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.mat.transpose()
        if self.normalization != "log":
            fc = (
                mat_transpose[group1_samples].T.mean().values
                / mat_transpose[group2_samples].T.mean().values
            )
        # calculate p-values
        # output needs to be checked
        p_values = self.mat.apply(
            lambda row: scipy.stats.ttest_ind(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
            )[1],
            axis=1,
        )
        df = pd.DataFrame()
        df["Protein IDs"] = p_values.index.tolist()
        df["fc"] = fc
        df["fc_log2"] = np.log2(fc)
        df["pvalue"] = p_values.values
        return df.dropna()

    def plot_pca(self, group=None):
        """plot PCA

        Parameters
        ----------
        group : _type_, optional
            _description_, by default None
        """
        if group:
            mat = self.mat[self.metadata["sample"].tolist()]
        else:
            mat = self.mat

        #  needs to be checked with publications
        # depends on normalization whether NA can be replaced with 0
        if self.imputation is None and self.mat.isna().values.any():
            logging.warn("Data contains missing values. Consider Imputation ")
        mat = mat.fillna(0)  # print warning depending on imputatio
        pipeline = Pipeline(
            [("scaling", StandardScaler()), ("pca", PCA(n_components=2))]
        )
        components = pipeline.fit_transform(mat.transpose())

        if group:
            fig = px.scatter(components, x=0, y=1, color=self.metadata[group])
        else:
            fig = px.scatter(components, x=0, y=1)
        return fig

    def plot_correlation_matrix(self, corr_method="pearson", save_figure=False):
        """_summary_

        Parameters
        ----------
        corr_method : str, optional
            _description_, by default "pearson"
        save_figure : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        corr_matrix = self.mat.corr(method=corr_method)
        plot = px.imshow(corr_matrix)
        return plot

    def plot_sampledistribution(self, group=None):
        df = self.mat.unstack().reset_index()
        fig = px.box(df, x="level_0", y=0)
        pass

    def plot_volcano(self, column, group1, group2):
        result = self.calculate_ttest_fc(column, group1, group2)
        volcano_plot = dash_bio.VolcanoPlot(
            dataframe=result,
            effect_size="fc_log2",
            p="pvalue",
            gene=None,
            snp=None,
            annotation="Protein IDs",
        )
        return volcano_plot
