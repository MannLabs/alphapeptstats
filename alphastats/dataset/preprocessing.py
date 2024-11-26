import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.impute
import streamlit as st
from sklearn.experimental import enable_iterative_imputer  # noqa

from alphastats.dataset.keys import Cols, ConstantsClass
from alphastats.dataset.utils import ignore_warning


class PreprocessingStateKeys(metaclass=ConstantsClass):
    """Keys for accessing the dictionary holding the information about preprocessing."""

    # TODO disentangle these keys from the human-readably display strings
    PREPROCESSING_DONE = "Preprocessing done"

    RAW_DATA_NUM_PG = "Raw data number of Protein Groups"
    NUM_PG = "Matrix= Number of ProteinIDs/ProteinGroups"
    NUM_SAMPLES = "Matrix= Number of samples"
    INTENSITY_COLUMN = "Intensity used for analysis"
    REPLACE_ZEROES = "Replace zero values with nan"
    LOG2_TRANSFORMED = "Log2-transformed"
    NORMALIZATION = "Normalization"
    IMPUTATION = "Imputation"
    CONTAMINATIONS_REMOVED = "Contaminations have been removed"
    CONTAMINATION_COLUMNS = "Contamination columns"
    NUM_REMOVED_PG_DUE_TO_CONTAMINATION = (
        "Number of removed ProteinGroups due to contaminaton"
    )
    DATA_COMPLETENESS_CUTOFF = "Data completeness cut-off"
    NUM_PG_REMOVED_DUE_TO_DATA_COMPLETENESS_CUTOFF = (
        "Number of removed ProteinGroups due to data completeness cutoff"
    )
    MISSING_VALUES_REMOVED = "Missing values were removed"
    DROP_UNMEASURED_FEATURES = "Drop unmeasured features"


class Preprocess:
    imputation_methods = ["mean", "median", "knn", "randomforest"]
    normalization_methods = ["vst", "zscore", "quantile"]

    def __init__(
        self,
        filter_columns: List[str],
        rawinput: pd.DataFrame,
        metadata: pd.DataFrame,
        preprocessing_info: Dict,
        mat: pd.DataFrame,
    ):
        self.filter_columns = filter_columns

        self.rawinput = rawinput

        self.metadata = metadata
        self.preprocessing_info = preprocessing_info
        self.mat = mat

    @staticmethod
    def init_preprocessing_info(
        num_samples: int,
        num_protein_groups: int,
        intensity_column: str,
        filter_columns: List[str],
    ) -> Dict:
        """Initialize preprocessing info."""
        return {
            PreprocessingStateKeys.PREPROCESSING_DONE: False,
            PreprocessingStateKeys.RAW_DATA_NUM_PG: num_protein_groups,
            PreprocessingStateKeys.NUM_PG: num_protein_groups,
            PreprocessingStateKeys.NUM_SAMPLES: num_samples,
            PreprocessingStateKeys.INTENSITY_COLUMN: intensity_column,
            PreprocessingStateKeys.REPLACE_ZEROES: False,
            PreprocessingStateKeys.LOG2_TRANSFORMED: False,
            PreprocessingStateKeys.NORMALIZATION: None,
            PreprocessingStateKeys.IMPUTATION: None,
            PreprocessingStateKeys.CONTAMINATIONS_REMOVED: False,
            PreprocessingStateKeys.CONTAMINATION_COLUMNS: filter_columns,
            PreprocessingStateKeys.NUM_REMOVED_PG_DUE_TO_CONTAMINATION: 0,
            PreprocessingStateKeys.DATA_COMPLETENESS_CUTOFF: 0,
            PreprocessingStateKeys.NUM_PG_REMOVED_DUE_TO_DATA_COMPLETENESS_CUTOFF: 0,
            PreprocessingStateKeys.MISSING_VALUES_REMOVED: False,
            PreprocessingStateKeys.DROP_UNMEASURED_FEATURES: False,
        }

    def _remove_samples(self, sample_list: list):
        # exclude samples for analysis
        self.mat = self.mat.drop(sample_list)
        self.metadata = self.metadata[~self.metadata[Cols.SAMPLE].isin(sample_list)]

    @staticmethod
    def subset(
        mat: pd.DataFrame, metadata: pd.DataFrame, preprocessing_info: Dict
    ) -> pd.DataFrame:
        """Filter matrix so only samples that are described in metadata are also found in matrix."""
        preprocessing_info.update(
            {PreprocessingStateKeys.NUM_SAMPLES: metadata.shape[0]}
        )
        return mat[mat.index.isin(metadata[Cols.SAMPLE].tolist())]

    def _remove_na_values(self, cut_off):
        if (
            self.preprocessing_info.get(PreprocessingStateKeys.MISSING_VALUES_REMOVED)
            and self.preprocessing_info.get(
                PreprocessingStateKeys.DATA_COMPLETENESS_CUTOFF
            )
            == cut_off
        ):
            logging.info("Missing values have already been filtered.")
            st.warning(
                "Missing values have already been filtered. To apply another cutoff, reset preprocessing."
            )
            return

        cut = 1 - cut_off

        num_samples, num_proteins = self.mat.shape
        limit = num_samples * cut

        keep_list = list()
        invalid = 0
        for column_name in self.mat.columns:
            column = self.mat[column_name]
            count = column.isna().sum()
            try:
                count = count.item()
                if isinstance(count, int) and count < limit:
                    keep_list += [column_name]

            except ValueError:
                invalid += 1
                continue
        self.mat = self.mat[keep_list]

        self.preprocessing_info.update(
            {
                PreprocessingStateKeys.NUM_PG_REMOVED_DUE_TO_DATA_COMPLETENESS_CUTOFF: num_proteins
                - self.mat.shape[1],
                PreprocessingStateKeys.MISSING_VALUES_REMOVED: True,
                PreprocessingStateKeys.DATA_COMPLETENESS_CUTOFF: cut_off,
            }
        )

    def _filter(self):
        if len(self.filter_columns) == 0:
            logging.info("No columns to filter.")
            return

        if self.preprocessing_info.get(PreprocessingStateKeys.CONTAMINATIONS_REMOVED):
            logging.info("Contaminatons have already been filtered.")
            return

        # print column names with contamination
        protein_groups_to_remove = self.rawinput[
            self.rawinput[self.filter_columns].any(axis=1)
        ][Cols.INDEX].tolist()

        protein_groups_to_remove = list(
            set(protein_groups_to_remove) & set(self.mat.columns.to_list())
        )

        # remove columns with protein groups
        self.mat = self.mat.drop(protein_groups_to_remove, axis=1)

        self.preprocessing_info.update(
            {
                PreprocessingStateKeys.NUM_REMOVED_PG_DUE_TO_CONTAMINATION: len(
                    protein_groups_to_remove
                ),
                PreprocessingStateKeys.CONTAMINATIONS_REMOVED: True,
                PreprocessingStateKeys.NUM_PG: self.mat.shape[1],
            }
        )

        filter_info = (
            f"Contaminations indicated in following columns: {self.filter_columns} were removed. "
            f"In total {str(len(protein_groups_to_remove))} observations have been removed."
        )
        logging.info(filter_info)

    @ignore_warning(RuntimeWarning)
    @ignore_warning(UserWarning)
    def _imputation(self, method: str):
        # remove ProteinGroups with only NA before
        protein_group_na = self.mat.columns[self.mat.isna().all()].tolist()

        if len(protein_group_na) > 0:
            self.mat = self.mat.drop(protein_group_na, axis=1)
            logging.info(
                f" {len(protein_group_na)} Protein Groups were removed due to missing values."
            )
        logging.info("Imputing data...")

        if method == "mean":
            imp = sklearn.impute.SimpleImputer(
                missing_values=np.nan, strategy="mean", keep_empty_features=True
            )
            imputation_array = imp.fit_transform(self.mat.values)

        elif method == "median":
            imp = sklearn.impute.SimpleImputer(
                missing_values=np.nan, strategy="median", keep_empty_features=True
            )
            imputation_array = imp.fit_transform(self.mat.values)

        elif method == "knn":
            # change for text
            method = "k-Nearest Neighbor"
            imp = sklearn.impute.KNNImputer(n_neighbors=2, weights="uniform")
            imputation_array = imp.fit_transform(self.mat.values)

        elif method == "randomforest":
            imp = sklearn.ensemble.HistGradientBoostingRegressor(
                max_depth=10, max_iter=100, random_state=0
            )
            imp = sklearn.impute.IterativeImputer(
                random_state=0, estimator=imp, verbose=1
            )
            imputation_array = imp.fit_transform(self.mat.values)

        else:
            raise ValueError(
                "Imputation method: {method} is invalid."
                "Choose 'mean'. 'median', 'knn' (for k-nearest Neighbors) or "
                "'randomforest' for random forest imputation."
            )

        self.mat = pd.DataFrame(
            imputation_array, index=self.mat.index, columns=self.mat.columns
        )
        self.preprocessing_info.update({PreprocessingStateKeys.IMPUTATION: method})

    def _linear_normalization(self, dataframe: pd.DataFrame):
        """Normalize data using l2 norm without breaking when encoutering nones
        l2 = sqrt(sum(x**2))

        Args:
            dataframe (pd.DataFrame): dataframe to normalize

        Returns:
            np.array: normalized np.array
        """
        square_sum_per_row = dataframe.pow(2).sum(axis=1, skipna=True)

        l2_norms = np.sqrt(square_sum_per_row)
        normalized_vals = dataframe.div(l2_norms.replace(0, 1), axis=0)
        return normalized_vals.values

    @ignore_warning(UserWarning)
    @ignore_warning(RuntimeWarning)
    def _normalization(self, method: str) -> None:
        """Normalize across samples."""
        # TODO make both sample and protein normalization available
        if method == "zscore":
            scaler = sklearn.preprocessing.StandardScaler()
            # normalize samples => for preprocessing
            normalized_array = scaler.fit_transform(
                self.mat.values.transpose()
            ).transpose()
            # normalize proteins => for downstream processing
            # normalized_array = scaler.fit_transform(self.mat.values)

        elif method == "quantile":
            qt = sklearn.preprocessing.QuantileTransformer(random_state=0)
            normalized_array = qt.fit_transform(self.mat.values.transpose()).transpose()
            # normalized_array = qt.fit_transform(self.mat.values) # normalize proteins

        elif method == "linear":
            normalized_array = self._linear_normalization(self.mat)

            # normalized_array = self._linear_normalization(
            #     self.mat.transpose()
            # ).transpose() # normalize proteins

        elif method == "vst":
            minmax = sklearn.preprocessing.MinMaxScaler()
            scaler = sklearn.preprocessing.PowerTransformer()
            minmaxed_array = minmax.fit_transform(self.mat.values.transpose())
            normalized_array = scaler.fit_transform(minmaxed_array).transpose()
            # minmaxed_array = minmax.fit_transform(self.mat.values)  # normalize proteins
            # normalized_array = scaler.fit_transform(minmaxed_array)  # normalize proteins

        else:
            raise ValueError(
                f"Normalization method: {method} is invalid. "
                "Choose from 'zscore', 'quantile', 'linear' normalization. or 'vst' for variance stabilization transformation"
            )

        self.mat = pd.DataFrame(
            normalized_array, index=self.mat.index, columns=self.mat.columns
        )

        self.preprocessing_info.update({PreprocessingStateKeys.NORMALIZATION: method})

    # TODO this needs to be reimplemented
    # @ignore_warning(RuntimeWarning)
    # def _compare_preprocessing_modes(self, func, params_for_func) -> list:
    #     dataset = self
    #
    #     preprocessing_modes = list(
    #         itertools.product(self.normalization_methods, self.imputation_methods)
    #     )
    #
    #     results_list = []
    #
    #     del params_for_func["compare_preprocessing_modes"]
    #     params_for_func["dataset"] = params_for_func.pop("self")
    #
    #     # TODO: make this progress transparent in GUI
    #     for preprocessing_mode in tqdm(preprocessing_modes):
    #         # reset preprocessing
    #         dataset.reset_preprocessing()
    #         print(
    #             f"Normalization {preprocessing_mode[0]}, Imputation {str(preprocessing_mode[1])}"
    #         )
    #         dataset.mat.replace([np.inf, -np.inf], np.nan, inplace=True)
    #
    #         dataset.preprocess(
    #             subset=True,
    #             normalization=preprocessing_mode[0],
    #             imputation=preprocessing_mode[1],
    #         )
    #
    #         res = func(**params_for_func)
    #         results_list.append(res)
    #
    #         print("\t")
    #
    #     return results_list

    def _log2_transform(self):
        self.mat = np.log2(self.mat)
        self.mat = self.mat.replace([np.inf, -np.inf], np.nan)
        self.preprocessing_info.update({PreprocessingStateKeys.LOG2_TRANSFORMED: True})
        print("Data has been log2-transformed.")

    def batch_correction(self, batch: str) -> pd.DataFrame:
        """Correct for technical bias/batch effects

        Args:
            batch (str): column name in the metadata describing the different batches

        # TODO should the employed methods (and citations) be made transparent in the UI?
        Behdenna A, Haziza J, Azencot CA and Nordor A. (2020) pyComBat,
        a Python tool for batch effects correction in high-throughput molecular
        data using empirical Bayes methods. bioRxiv doi: 10.1101/2020.03.17.995431
        """
        from combat.pycombat import pycombat

        data = self.mat.transpose()
        series_of_batches = self.metadata.set_index(Cols.SAMPLE).reindex(
            data.columns.to_list()
        )[batch]

        batch_corrected_data = pycombat(data=data, batch=series_of_batches).transpose()

        return batch_corrected_data

    @ignore_warning(RuntimeWarning)
    def preprocess(
        self,
        log2_transform: bool = False,
        remove_contaminations: bool = False,
        subset: bool = False,
        replace_zeroes: bool = False,
        data_completeness: float = 0,
        normalization: str = None,
        imputation: str = None,
        remove_samples: list = None,
        drop_unmeasured_features: bool = False,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
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
            log2_transform (bool, optional): Log2 transform data. Default to True.
            normalization (str, optional): method to normalize data: either "zscore", "quantile", "linear". Defaults to None.
            data_completeness (float, optional): data completeness across all samples between 0-1. Defaults to 0.
            remove_samples (list, optional): list with sample ids to remove. Defaults to None.
            imputation (str, optional):  method to impute data: either "mean", "median", "knn" or "randomforest". Defaults to None.
            subset (bool, optional): filter matrix so only samples that are described in metadata found in matrix. Defaults to False.
        """
        for k in kwargs:
            if k not in [
                "batch",
            ]:
                raise ValueError(f"Invalid keyword argument: {k}")

        # TODO this is a stateful method as we change self.mat, self.metadata and self.processing_info
        #  refactor such that it does not change self.mat etc but just return the latest result
        if remove_contaminations:
            self._filter()

        if remove_samples is not None:
            self._remove_samples(sample_list=remove_samples)

        if subset:
            self.mat = self.subset(self.mat, self.metadata, self.preprocessing_info)

        if replace_zeroes:
            self.mat = self.mat.replace(0, np.nan)
            self.preprocessing_info.update(
                {
                    PreprocessingStateKeys.REPLACE_ZEROES: True,
                }
            )

        if data_completeness > 0:
            self._remove_na_values(cut_off=data_completeness)

        if (
            log2_transform
            and self.preprocessing_info.get(PreprocessingStateKeys.LOG2_TRANSFORMED)
            is False
        ):
            self._log2_transform()

        if normalization is not None:
            self._normalization(method=normalization)
            self.mat = self.mat.replace([np.inf, -np.inf], np.nan)

        if imputation is not None:
            self._imputation(method=imputation)

        if drop_unmeasured_features:
            n = self.mat.shape[1]
            self.mat = self.mat.loc[:, np.isfinite(self.mat).any(axis=0)]
            self.preprocessing_info.update(
                {
                    PreprocessingStateKeys.DROP_UNMEASURED_FEATURES: n
                    - self.mat.shape[1],
                }
            )

        self.preprocessing_info.update(
            {
                PreprocessingStateKeys.PREPROCESSING_DONE: True,
                PreprocessingStateKeys.NUM_PG: self.mat.shape[1],
            }
        )

        return self.mat, self.metadata, self.preprocessing_info
