from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
import scipy

from alphastats.dataset.keys import Cols
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.multicova import multicova
from alphastats.statistics.statistic_utils import (
    add_metadata_column,
    calculate_foldchange,
)


class DifferentialExpressionAnalysis:
    def __init__(
        self,
        mat: pd.DataFrame,
        metadata: pd.DataFrame,
        preprocessing_info: Dict,
        group1: Union[str, list],
        group2: Union[str, list],
        column: str = None,
        # TODO move these to perform()?
        method: str = "ttest",
        perm: int = 10,
        fdr: float = 0.05,
    ):
        self.mat = mat

        self.preprocessing_info = preprocessing_info

        self.method = method
        self.perm = perm
        self.fdr = fdr

        if isinstance(group1, list) and isinstance(group2, list):
            self.metadata, self.column = add_metadata_column(metadata, group1, group2)
            self.group1, self.group2 = "group1", "group2"
        else:
            self.metadata, self.column = metadata, column
            self.group1, self.group2 = group1, group2

        if self.column is None:
            raise ValueError(
                "Column containing group1 and group2 needs to be specified"
            )

    def _prepare_anndata(self):
        import anndata

        group_samples = self.metadata[
            (self.metadata[self.column] == self.group1)
            | (self.metadata[self.column] == self.group2)
        ][Cols.SAMPLE].tolist()

        # reduce matrix
        reduced_matrix = self.mat.loc[group_samples]
        reduced_matrix = reduced_matrix.loc[:, (reduced_matrix != 0).any(axis=0)]
        # sort metadata according to matrix values
        list_to_sort = reduced_matrix.index.to_list()
        # reduce metadata
        obs_metadata = (
            self.metadata[self.metadata[Cols.SAMPLE].isin(group_samples)]
            .set_index(Cols.SAMPLE)
            .loc[list_to_sort]
        )

        # change comparison group to 0/1
        obs_metadata[self.column] = np.where(
            obs_metadata[self.column] == self.group1, 1, 0
        )

        # create a annotated dataset
        anndata_data = anndata.AnnData(
            X=reduced_matrix.values,
            var=pd.DataFrame(index=reduced_matrix.columns.to_list()),
            obs=obs_metadata,
            dtype=reduced_matrix.values.dtype,
        )
        return anndata_data

    def sam(self) -> Tuple[pd.DataFrame, float]:
        transposed = self.mat.transpose()

        if not self.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED]:
            # needs to be lpog2 transformed for fold change calculations
            transposed = transposed.transform(lambda x: np.log2(x))

        transposed[Cols.INDEX] = transposed.index
        transposed = transposed.reset_index(drop=True)

        res_ttest, tlim_ttest = multicova.perform_ttest_analysis(
            transposed,
            c1=list(
                self.metadata[self.metadata[self.column] == self.group1][Cols.SAMPLE]
            ),
            c2=list(
                self.metadata[self.metadata[self.column] == self.group2][Cols.SAMPLE]
            ),
            # TODO: Remove hardcoded values
            s0=0.05,
            n_perm=self.perm,
            fdr=self.fdr,
            parallelize=True,
        )

        fdr_column = "FDR" + str(int(self.fdr * 100)) + "%"
        df = res_ttest[
            [
                Cols.INDEX,
                "fc",
                "tval",
                "pval",
                "tval_s0",
                "pval_s0",
                "qval",
            ]
        ]
        # TODO: these can just be a renames
        df["log2fc"] = res_ttest["fc"]
        df["FDR"] = res_ttest[fdr_column]

        return df, tlim_ttest

    def _wald(self) -> pd.DataFrame:
        import diffxpy.api as de

        d = self._prepare_anndata()
        formula_loc = "~ 1 +" + self.column

        test = de.test.wald(
            data=d, formula_loc=formula_loc, factor_loc_totest=self.column
        )
        df = test.summary().rename(columns={"gene": Cols.INDEX})
        return df

    def _welch_ttest(self) -> pd.DataFrame:
        import diffxpy.api as de

        d = self._prepare_anndata()

        # TODO: pass log flag correctly
        test = de.test.t_test(data=d, grouping=self.column)
        df = test.summary().rename(columns={"gene": Cols.INDEX})
        return df

    def _generic_ttest(self, test_fun: Callable) -> pd.DataFrame:
        """
        Perform a t-test between two groups, assuming log-normally distributed data.

        If the data was not already log transformed during preprocessing, it will be log2 transformed here. > Log2-transformed data will be used for the t-test

        Parameters:
            test_fun (Callable): A function that performs a t-test, e.g. scipy.stats.ttest_ind or scipy.stats.ttest_rel

        Returns:
            pd.DataFrame: DataFrame with index_column, p-value and log2 fold change.
        """
        group1_samples = self.metadata[self.metadata[self.column] == self.group1][
            Cols.SAMPLE
        ].tolist()
        group2_samples = self.metadata[self.metadata[self.column] == self.group2][
            Cols.SAMPLE
        ].tolist()
        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.mat.transpose()

        if not self.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED]:
            mat_transpose = mat_transpose.transform(lambda x: np.log2(x))
            mat_transpose = mat_transpose.replace([np.inf, -np.inf], np.nan)

        # TODO: return not only the p-value, but also the t-statistic
        p_values = mat_transpose.apply(
            lambda row: test_fun(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
                nan_policy="omit",
            )[1],
            axis=1,
        )

        df = pd.DataFrame()
        df[Cols.INDEX], df["pval"] = (
            p_values.index.tolist(),
            p_values.values,
        )
        df["log2fc"] = calculate_foldchange(
            mat_transpose=mat_transpose,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            is_log2_transformed=True,
        )
        return df

    def _ttest(self) -> pd.DataFrame:
        return self._generic_ttest(test_fun=scipy.stats.ttest_ind)

    def _pairedttest(self) -> pd.DataFrame:
        return self._generic_ttest(test_fun=scipy.stats.ttest_rel)

    def perform(self) -> pd.DataFrame:
        if self.method == "wald":
            df = self._wald()

        elif self.method == "ttest":
            df = self._ttest()

        elif self.method == "welch-ttest":
            df = self._welch_ttest()

        elif self.method == "sam":
            df, _ = self.sam()

        elif self.method == "paired-ttest":
            df = self._pairedttest()

        else:
            raise ValueError(
                f"{self.method} is invalid choose between 'wald' for Wald-test, 'sam',  and 'ttest', 'welch-ttest' or 'paired-ttest'"
            )

        return df
