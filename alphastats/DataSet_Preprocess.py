from random import random
import pandas as pd
import sklearn
import logging
import numpy as np
import sklearn.ensemble
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer
from alphastats.utils import ignore_warning


class Preprocess:
    def _remove_sampels(self, sample_list):
        # exclude samples for analysis
        self.mat = self.mat.drop(sample_list)
        self.metadata = self.metadata[~self.metadata[self.sample].isin(sample_list)]

    def _subset(self):
        # filter matrix so only samples that are described in metadata
        # also found in matrix
        self.preprocessing_info.update(
            {"Matrix: Number of samples": self.metadata.shape[0]}
        )
        return self.mat[self.mat.index.isin(self.metadata[self.sample].tolist())]

    def preprocess_print_info(self):
        """Print summary of preprocessing steps
        """
        print(pd.DataFrame(self.preprocessing_info.items()))

    def _filter(self):
        if len(self.filter_columns) == 0:
            logging.info("No columns to filter.")
            return

        if self.preprocessing_info.get("Contaminations have been removed") == True:
            logging.info("Contaminatons have already been filtered.")
            return

        #  print column names with contamination
        protein_groups_to_remove = self.rawinput[
            (self.rawinput[self.filter_columns] == True).any(1)
        ][self.index_column].tolist()

        protein_groups_to_remove = list(
            set(protein_groups_to_remove) & set(self.mat.columns.to_list())
        )

        # remove columns with protein groups
        self.mat = self.mat.drop(protein_groups_to_remove, axis=1)

        self.preprocessing_info.update(
            {
                "Number of removed ProteinGroups due to contaminaton": len(
                    protein_groups_to_remove
                ),
                "Contaminations have been removed": True,
                "Matrix: Number of ProteinIDs/ProteinGroups": self.mat.shape[1],
            }
        )

        filter_info = (
            f"Contaminations indicated in following columns: {self.filter_columns} were removed. "
            f"In total {str(len(protein_groups_to_remove))} observations have been removed."
        )
        logging.info(filter_info)

    @ignore_warning(RuntimeWarning)
    @ignore_warning(UserWarning)
    def _imputation(self, method):
        # Impute Data
        # For more information visit:
        # SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
        # k-Nearest Neighbors Imputation: https://scikit-learn.org/stable/modules/impute.html#impute

        # Args:
        #    method (str): method to impute data: either "mean", "median" or "knn"

        # remove ProteinGroups with only NA before
        protein_group_na = self.mat.columns[self.mat.isna().all()].tolist()

        if len(protein_group_na) > 0:
            self.mat = self.mat.drop(protein_group_na, axis=1)
            logging.info(
                f" {len(protein_group_na)} Protein Groups were removed due to missing values."
            )

        logging.info("Imputing data...")

        if method == "mean":
            imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="mean")
            imputation_array = imp.fit_transform(self.mat.values)

        elif method == "median":
            imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="median")
            imputation_array = imp.fit_transform(self.mat.values)

        elif method == "knn":
            # change for text
            method = "k-Nearest Neighbor"
            imp = sklearn.impute.KNNImputer(n_neighbors=2, weights="uniform")
            imputation_array = imp.fit_transform(self.mat.values)

        elif method == "randomforest":
            randomforest = sklearn.ensemble.RandomForestRegressor(
                max_depth=10,
                bootstrap=True,
                max_samples=0.5,
                n_jobs=2,
                random_state=0,
                verbose=0,  #  random forest takes a while print progress
            )
            imp = sklearn.impute.IterativeImputer(
                random_state=0, estimator=randomforest
            )

            # the random forest imputer doesnt work with float32/float16..
            #  so the values are multiplied and converted to integers
            array_multi_mio = self.mat.values * 1000000
            array_int = array_multi_mio.astype("int")

            imputation_array = imp.fit_transform(array_int)
            imputation_array = imputation_array / 1000000

        else:
            raise ValueError(
                "Imputation method: {method} is invalid."
                "Choose 'mean'. 'median', 'knn' (for k-nearest Neighbors) or "
                "'randomforest' for random forest imputation."
            )

        self.mat = pd.DataFrame(
            imputation_array, index=self.mat.index, columns=self.mat.columns
        )
        self.preprocessing_info.update({"Imputation": method})

    @ignore_warning(UserWarning)
    @ignore_warning(RuntimeWarning)
    def _normalization(self, method):
        # Normalize data using either zscore, quantile or linear (using l2 norm) Normalization.
        # Z-score normalization equals standaridzation using StandardScaler:
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        # For more information visit.
        # Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

        # Args:
        # method (str): method to normalize data: either "zscore", "quantile", "linear"

        # zscore normalization == standardization

        if method == "zscore":
            scaler = sklearn.preprocessing.StandardScaler()
            normalized_array = scaler.fit_transform(self.mat.values)

        elif method == "quantile":
            qt = sklearn.preprocessing.QuantileTransformer(random_state=0)
            normalized_array = qt.fit_transform(self.mat.values)

        elif method == "linear":
            normalized_array = sklearn.preprocessing.normalize(
                self.mat.values, norm="l2"
            )

        elif method == "vst":
            scaler = sklearn.preprocessing.PowerTransformer()
            normalized_array = scaler.fit_transform(self.mat.values)

        else:
            raise ValueError(
                "Normalization method: {method} is invalid"
                "Choose from 'zscore', 'quantile', 'linear' normalization. or 'vst' for variance stabilization transformation"
            )

        # TODO logarithimic normalization

        self.mat = pd.DataFrame(
            normalized_array, index=self.mat.index, columns=self.mat.columns
        )
        self.preprocessing_info.update({"Normalization": method})

    @ignore_warning(RuntimeWarning)
    def preprocess(
        self,
        remove_contaminations=False,
        subset=False,
        normalization=None,
        imputation=None,
        remove_samples=None,
    ):
        """Preprocess Protein data

        Removal of contaminations:

        Removes all observations, that were identified as contaminations. 

        Normalization:

        "zscore", "quantile", "linear", "vst"

        Normalize data using either zscore, quantile or linear (using l2 norm) Normalization.
        
        Z-score normalization equals standaridzation using StandardScaler: 
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        Variance stabilization transformation uses:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
        
        For more information visit. 
        Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

        Imputation:

        "mean", "median", "knn" or "randomforest"
        For more information visit:
        
        SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
        
        k-Nearest Neighbors Imputation: https://scikit-learn.org/stable/modules/impute.html#impute
        
        Random Forest Imputation: https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

        Args:
            remove_contaminations (bool, optional): remove ProteinGroups that are identified as contamination.
            normalization (str, optional): method to normalize data: either "zscore", "quantile", "linear". Defaults to None.
            remove_samples (list, optional): list with sample ids to remove. Defaults to None.
            imputation (str, optional):  method to impute data: either "mean", "median", "knn" or "randomforest". Defaults to None.
            subset (bool, optional): filter matrix so only samples that are described in metadata found in matrix. Defaults to False.
        """
        if remove_contaminations:
            self._filter()

        if subset:
            self.mat = self._subset()

        if normalization is not None:
            self._normalization(method=normalization)

        if imputation is not None:
            self._imputation(method=imputation)

        if remove_samples is not None:
            self._remove_sampels(sample_list=remove_samples)

        self.mat = self.mat.loc[:, (self.mat != 0).any(axis=0)]
