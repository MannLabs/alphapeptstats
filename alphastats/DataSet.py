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
import scipy.stats
import dash_bio
import numpy as np
import logging
import sys
import sklearn
from sklearn_pandas import DataFrameMapper
import warnings

# remove warning from openpyxl
# only appears on mac
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


class DataSet:
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

        # save preprocessing settings
        self.preprocessing = None
        # update normalization when self.matrix is normalized, filtered
        self.normalization, self.imputation = (
            "Data is not normalized.",
            "Data is not imputed.",
        )
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
            return
        if not isinstance(loader.rawdata, pd.DataFrame) or loader.rawdata.empty:
            logging.error(
                "Error in rawdata, consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader"
            )
            return
        if not isinstance(loader.index_column, str):
            logging.error(
                "Invalid index_column: consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader"
            )
            return

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
        # reset preproccessing info
        self.normalization, self.imputation = (
            "Data is not normalized",
            "Data is not imputed",
        )
        self.removed_protein_groups = None

    def preprocess_remove_sampels(self, sample_list):
        # exclude samples for analysis
        self.mat = self.mat.drop(sample_list)

    def preprocess_subset(self):
        # filter matrix so only samples that are described in metadata
        # also found in matrix
        return self.mat[self.mat.index.isin(self.metadata["sample"].tolist())]

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
        preprocessing_text = text + self.normalization + self.imputation
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
        # remove ProteinGroups with only NA before
        protein_group_na = self.mat.columns[self.mat.isna().all()].tolist()
        if len(protein_group_na) > 0:
            self.mat = self.mat.drop(protein_group_na, axis=1)
            logging.info(f"{len(protein_group_na)} Protein Groups were removed.")
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
        # TODO logarithimic normalization
        self.mat = pd.DataFrame(
            normalized_array, index=self.mat.index, columns=self.mat.columns
        )
        self.normalization = f"Data has been normalized using {method} normalization"

    def preprocess(
        self,
        remove_contaminations=False,
        subset=False,
        normalization=None,
        imputation=None,
        remove_samples=None,
        qvalue=0.01,
    ):
        """Preprocess Protein data

        Args:
            remove_contaminations (bool, optional): remove ProteinGroups that are identified 
            as contamination. Defaults to False.
            normalization (str, optional): method to normalize data: either "zscore", "quantile", 
            "linear". Defaults to None.
            remove_samples (list, optional): list with sample ids to remove. Defaults to None.
            imputation (str, optional):  method to impute data: either "mean", "median" or "knn". 
            Defaults to None.
            qvalue (float, optional): _description_. Defaults to 0.01.

        Raises:
            NotImplementedError: _description_
        """
        if remove_contaminations:
            self.preprocess_filter()
        if subset:
            self.mat = self.preprocess_subset()
        if normalization is not None:
            self.preprocess_normalization(method=normalization)
        if imputation is not None:
            self.preprocess_impute(method=imputation)
        if remove_samples is not None:
            self.preprocess_remove_sampels(sample_list=remove_samples)

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

    def calculate_ttest_fc(self, column, group1, group2):
        """Calculate t-test and fold change between two groups

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str): name of group to compare needs to be present in column
            group2 (str): nme of group to compare needs to be present in column

        Returns:
            pandas Dataframe: pandas Dataframe with foldchange, foldchange_log2 and pvalue
            for each ProteinID/ProteinGroup between group1 and group2
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

        p_values = mat_transpose.apply(
            lambda row: scipy.stats.ttest_ind(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
            )[1],
            axis=1,
        )
        df = pd.DataFrame()
        df["Protein IDs"], df["pvalue"] = p_values.index.tolist(), p_values.values
        df["foldchange"], df["foldchange_log2"] = fc, np.log2(fc)
        return df

    def plot_pca(self, group=None):
        """Plot Principal Component Analysis (PCA)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.

        Returns:
            plotly Object: p
        """
        if self.imputation is None and self.mat.isna().values.any():
            logging.warning(
                "Data contains missing values. Missing values will replaced with 0. Consider Imputation."
            )

        if self.normalization == "Data is not normalized.":
            logging.info(
                "Data has not been normalized. Data will be normalized using zscore-Normalization"
            )
            self.preprocess(normalization="zscore")

        # subset matrix so it matches with metadata
        if group:
            mat = self.preprocess_subset()
            group_color = self.metadata[group]
        else:
            mat = self.mat
            group_color = group
        mat = mat.fillna(0)

        pca = sklearn.decomposition.PCA(n_components=2)
        components = pca.fit_transform(mat)

        fig = px.scatter(
            components,
            x=0,
            y=1,
            labels={
                "0": "PC 1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100),
                "1": "PC 2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100),
            },
            color=group_color,
        )
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
            effect_size="foldchange_log2",
            p="pvalue",
            gene=None,
            snp=None,
            annotation="Protein IDs",
        )
        return volcano_plot


# Plotly update figures
# https://maegul.gitbooks.io/resguides-plotly/content/content/plotting_locally_and_offline/python/methods_for_updating_the_figure_or_graph_objects.html
