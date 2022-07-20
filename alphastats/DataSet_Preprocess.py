import pandas as pd
import sklearn
import logging
import numpy as np


class Preprocess:
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
            f"{str(n_samples)} samples.\n {str(len(self.removed_protein_groups))}"
            f"rows with Proteins/Protein Groups have been removed."
        )
        preprocessing_text = text + self.normalization + self.imputation
        print(preprocessing_text)

    def preprocess_filter(self):
        """Removes all observations, that were identified as contaminations. 
        """
        if len(self.filter_columns) == 0:
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

    def preprocess_imputation(self, method):
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
            logging.info(
                f" {len(protein_group_na)} Protein Groups were removed due to missing values."
            )
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
            qt = sklearn.preprocessing.QuantileTransformer(random_state=0)
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
            as contamination. Calls preprocess_filter() Defaults to False.
            normalization (str, optional): method to normalize data: either "zscore", "quantile", 
            "linear", calls preprocess_normalizatin(). Defaults to None.
            remove_samples (list, optional): list with sample ids to remove. Defaults to None.
            imputation (str, optional):  method to impute data: either "mean", "median" or "knn", calls preprocess_imputation(). 
            Defaults to None.
            qvalue (float, optional): _description_. Defaults to 0.01.
        """
        if remove_contaminations:
            self.preprocess_filter()
        if subset:
            self.preprocess_subset()
        if normalization is not None:
            self.preprocess_normalization(method=normalization)
        if imputation is not None:
            self.preprocess_imputation(method=imputation)
        if remove_samples is not None:
            self.preprocess_remove_sampels(sample_list=remove_samples)
