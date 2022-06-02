from random import sample
import re
import pandas as pd
import seaborn as sn
from data_cache import pandas_cache
import os
import warnings
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import scipy.stats
import dash_bio
import numpy as np
import logging
from sklearn.impute import SimpleImputer


class proteinObject:
    """_summary_
    """

    def __init__(self, loader, metadata_path: str = None):

        """"
        """
        """Create a Protein Object containing the protein intensity and the corresponding metadata of the samples,
        ready for analyis 

        Parameters
        ----------
        rawfile_path : str
            path to Protein Intensity file
        metadata_path : str, optional
            path to metadata file (xlsx, csv or tsv), by default None
        intensity_column : str, optional
            , by default None
        software : str, optional
            _description_, by default None
        """

        #  load data from loader object
        self.loader = loader
        # check if loader is valid
        # self.check_loader()
        self.rawdata = loader.rawdata
        self.software = loader.software
        self.index_column = loader.index_column
        self.intensity_column = loader.intensity_column
        self.filter_columns = loader.filter_columns

        # include filtering before
        self.mat = self.create_matrix()
        self.metadata = None

        if metadata_path:
            self.metadata = self.load_metadata()

        self.experiment_type = None
        self.data_format = None
        # save preprocessing settings
        self.preprocessing = None
        # update normalization when self.matrix is normalized, filtered
        self.normalization = None
        self.removed_protein_groups = None
        self.imputation = None
        self.removed_protein_groups = None

    def check_loader(self):
        # check if loader is really from class loader and contains necessary columns
        pass

    @pandas_cache
    def create_matrix(self):
        """Creates a matrix out of the MaxQuant ProteinGroup Outputfile, with columns displaying samples and
        row the protein IDs.

        Parameters
        ----------
        df : df
        Pandas Dataframe of the MaxQuant Output ProteinGroup file
        intensity_col : str, optional
        columns , by default "LFQ intensity "
        proteinID_col : str, optional
        column in Dataframe containg the Protein IDs, must be unique, by default "Protein IDs"

        Returns
        -------
        _type_
        _description_
        """
        regex_find_intensity_columns = self.intensity_column.replace(
            "[experiment]", ".*"
        )
        df = self.rawdata.set_index(self.index_column)
        df = self.rawdata.filter(regex=(regex_find_intensity_columns), axis=1)
        # remove Intensity so only sample names remain
        df.columns = df.columns.str.replace(regex_find_intensity_columns[3:], "")
        self.mat = df
        self.normalization = None
        self.imputation = None
        self.removed_protein_groups = None

    def preprocess_exclude_sampels(self, sample_list):
        # exclude samples for analysis
        pass

    def preprocess_print_info(self):
        n_proteins = self.rawdata.shape[0]
        n_samples = self.rawdata.shape[1]  #  remove filter columns etc.

        text = (
            "Preprocessing: \nThe raw data contains "
            + str(n_proteins)
            + " Proteins and "
            + str(n_samples)
            + "samples.\n"
            + str(len(self.removed_protein_groups))
            + " rows with Proteins/Protein Groups have been removed."
        )

        if self.normalization is None:
            normalization_text = (
                "Data has not been normalized, or has already been normalized by "
                + self.software
                + ".\n"
            )
        else:
            normalization_text = (
                "Data has been normalized using " + self.normalization + ".\n"
            )

        if self.imputation is None:
            imputation_text = "Data is not imputed.\n"
        else:
            imputation_text = self.imputation

        preprocessing_text = text + normalization_text + imputation_text
        print(preprocessing_text)

    def preprocess_filter(self):
        """_summary_
        """
        if self.filter_columns is None:
            logging.error("No columns to filter.")
        #  print column names with contamination
        logging.info(
            "Contaminations indicated in following columns: ",
            self.filter_columns,
            "are removed",
        )

        protein_groups_to_remove = self.rawdata[
            (self.rawdata[self.filter_columns] == False).any(1)
        ][self.index_column].tolist()
        self.mat = self.drop(protein_groups_to_remove)
        self.removed_protein_groups = protein_groups_to_remove
        logging.info(len(protein_groups_to_remove), " observations have been removed.")

    @pandas_cache
    def preprocess(
        self,
        normalization=None,
        remove_contaminations=False,  #  needs to be changed when using different loaders
        remove_samples=None,
        impute=False,
        qvalue=0.01,
    ):

        if remove_contaminations:
            self.preprocess_filter()

        if normalization is not None:
            self.mat = normalization.normalize_data(
                self.mat,
                method=normalization,
                normalize="samples",
                max_iterations=250,
                linear_method="l1",
            )
            self.normalization = normalization

        if impute:
            imp = SimpleImputer(missing_values=np.nan, strategy="mean")
            imp.fit(self.mat.values)
            imputation_array = imp.transform(self.mat.values)
            # https://scikit-learn.org/stable/modules/impute.html
            self.mat = pd.DataFrame(
                imputation_array, index=self.mat.index, columns=self.mat.columns
            )
            self.imputation = "Missing values were imputed using the mean."

        if remove_samples is not None:
            raise NotImplementedError

    def load_metadata(file_path, sample_column=None):
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

        df.columns = df.columns.str.replace(sample_column, "sample")
        # check whether sample labeling matches protein data
        #  warnings.warn("WARNING: Sample names do not match sample labelling in protein data")
        return df

    def summary(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # print summary
        # look at keras model.summary()
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
        if self.normalization != "log":
            fc = (
                self.mat[group1_samples].T.mean().values
                / self.mat[group2_samples].T.mean().values
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
