from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
import scipy

from alphastats.DataSet_Preprocess import PreprocessingStateKeys
from alphastats.statistics.StatisticUtils import _add_metadata_column


class DifferentialExpressionAnalysis:
    def __init__(
        self,
        mat: pd.DataFrame,
        metadata: pd.DataFrame,
        sample: str,
        index_column: str,
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

        self.sample = sample
        self.index_column = index_column
        self.preprocessing_info = preprocessing_info

        self.method = method
        self.perm = perm
        self.fdr = fdr

        if isinstance(group1, list) and isinstance(group2, list):
            self.metadata, self.column = _add_metadata_column(
                metadata, sample, group1, group2
            )
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
        ][self.sample].tolist()

        # reduce matrix
        reduced_matrix = self.mat.loc[group_samples]
        reduced_matrix = reduced_matrix.loc[:, (reduced_matrix != 0).any(axis=0)]
        # sort metadata according to matrix values
        list_to_sort = reduced_matrix.index.to_list()
        # reduce metadata
        obs_metadata = (
            self.metadata[self.metadata[self.sample].isin(group_samples)]
            .set_index(self.sample)
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

    def _sam(self) -> pd.DataFrame:  # TODO duplicated? DUP1
        from alphastats.multicova import multicova

        transposed = self.mat.transpose()

        if not self.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED]:
            # needs to be lpog2 transformed for fold change calculations
            transposed = transposed.transform(lambda x: np.log2(x))

        transposed[self.index_column] = transposed.index
        transposed = transposed.reset_index(drop=True)

        res, _ = multicova.perform_ttest_analysis(
            transposed,
            c1=list(
                self.metadata[self.metadata[self.column] == self.group1][self.sample]
            ),
            c2=list(
                self.metadata[self.metadata[self.column] == self.group2][self.sample]
            ),
            s0=0.05,
            n_perm=self.perm,
            fdr=self.fdr,
            id_col=self.index_column,
            parallelize=True,
        )

        fdr_column = "FDR" + str(int(self.fdr * 100)) + "%"
        df = res[
            [
                self.index_column,
                "fc",
                "tval",
                "pval",
                "tval_s0",
                "pval_s0",
                "qval",
            ]
        ]
        df["log2fc"] = res["fc"]
        df["FDR"] = res[fdr_column]

        return df

    def _wald(self) -> pd.DataFrame:
        import diffxpy.api as de

        d = self._prepare_anndata()
        formula_loc = "~ 1 +" + self.column

        test = de.test.wald(
            data=d, formula_loc=formula_loc, factor_loc_totest=self.column
        )
        df = test.summary().rename(columns={"gene": self.index_column})
        return df

    def _welch_ttest(self) -> pd.DataFrame:
        import diffxpy.api as de

        d = self._prepare_anndata()

        test = de.test.t_test(data=d, grouping=self.column)
        df = test.summary().rename(columns={"gene": self.index_column})
        return df

    def _generic_ttest(self, test_fun: Callable) -> pd.DataFrame:
        group1_samples = self.metadata[self.metadata[self.column] == self.group1][
            self.sample
        ].tolist()
        group2_samples = self.metadata[self.metadata[self.column] == self.group2][
            self.sample
        ].tolist()
        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.mat.transpose()

        p_values = mat_transpose.apply(
            lambda row: test_fun(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
            )[1],
            axis=1,
        )

        df = pd.DataFrame()
        df[self.index_column], df["pval"] = (
            p_values.index.tolist(),
            p_values.values,
        )
        df["log2fc"] = self.calculate_foldchange(
            mat_transpose=mat_transpose,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            is_log2_transformed=self.preprocessing_info[
                PreprocessingStateKeys.LOG2_TRANSFORMED
            ],
        )
        return df

    def _ttest(self) -> pd.DataFrame:
        return self._generic_ttest(test_fun=scipy.stats.ttest_ind)

    def _pairedttest(self) -> pd.DataFrame:
        return self._generic_ttest(test_fun=scipy.stats.ttest_rel)

    @staticmethod
    def calculate_foldchange(
        mat_transpose: pd.DataFrame,
        group1_samples: list,
        group2_samples: list,
        is_log2_transformed: bool,
    ):
        group1_values = mat_transpose[group1_samples].T.mean().values
        group2_values = mat_transpose[group2_samples].T.mean().values
        if is_log2_transformed:
            fc = group1_values - group2_values

        else:
            fc = group1_values / group2_values
            fc = np.log2(fc)

        return fc

    def perform(self) -> pd.DataFrame:
        if self.method == "wald":
            df = self._wald()

        elif self.method == "ttest":
            df = self._ttest()

        elif self.method == "welch-ttest":
            df = self._welch_ttest()

        elif self.method == "sam":
            df = self._sam()

        elif self.method == "paired-ttest":
            df = self._pairedttest()

        else:
            raise ValueError(
                f"{self.method} is invalid choose between 'wald' for Wald-test, 'sam',  and 'ttest', 'welch-ttest' or 'paired-ttest'"
            )

        return df
