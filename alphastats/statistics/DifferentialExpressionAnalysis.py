import pandas as pd
import numpy as np

import scipy
from typing import Union

class DifferentialExpressionAnalysis:
    def __init__(
            self,
            dataset,
            group1: Union[str, list],
            group2: Union[str, list],
            column: str = None,
            method: str = "ttest",
            perm: int = 10,
            fdr: float = 0.05,
        ):

        self.dataset = dataset
        self.group1 = group1
        self.group2 = group2
        self.column = column
        self.method = method
        self.perm = perm
        self.fdr = fdr
    
    def _check_groups(self):
        if isinstance(self.group1, list) and isinstance(self.group2, list):
            self.column, self.group1, self.group2 = self._add_metadata_column(self.group1, self.group2)

        elif self.column is None:
            raise ValueError(
                "Column containing group1 and group2 needs to be specified"
            )

    def _prepare_anndata(self):
        import anndata
        group_samples = self.dataset.metadata[
            (self.dataset.metadata[self.column] == self.group1) | (self.dataset.metadata[self.column] == self.group2)
        ][self.dataset.sample].tolist()

        # reduce matrix
        reduced_matrix = self.dataset.mat.loc[group_samples]
        reduced_matrix = reduced_matrix.loc[:, (reduced_matrix != 0).any(axis=0)]
        # sort metadata according to matrix values
        list_to_sort = reduced_matrix.index.to_list()
        #  reduce metadata
        obs_metadata = (
            self.dataset.metadata[self.dataset.metadata[self.dataset.sample].isin(group_samples)]
            .set_index(self.dataset.sample)
            .loc[list_to_sort]
        )

        # change comparison group to 0/1
        obs_metadata[self.column] = np.where(obs_metadata[self.column] == self.group1, 1, 0)

        # create a annotated dataset
        anndata_data = anndata.AnnData(
            X=reduced_matrix.values,
            var=pd.DataFrame(index=reduced_matrix.columns.to_list()),
            obs=obs_metadata,
            dtype=reduced_matrix.values.dtype,
        )
        return anndata_data

    def _add_metadata_column(self, group1_list:list, group2_list:list):
        # create new column in metadata with defined groups
        metadata = self.dataset.metadata

        sample_names = metadata[self.dataset.sample].to_list()

        misc_samples = list(set(group1_list + group2_list) - set(sample_names))
        if len(misc_samples) > 0:
            raise ValueError(
                f"Sample names: {misc_samples} are not described in Metadata."
            )

        column = "_comparison_column"
        conditons = [
            metadata[self.dataset.sample].isin(group1_list),
            metadata[self.dataset.sample].isin(group2_list),
        ]
        choices = ["group1", "group2"]
        metadata[column] = np.select(conditons, choices, default=np.nan)
        self.dataset.metadata = metadata

        return column, "group1", "group2"

    def sam(self) -> pd.DataFrame:
        from alphastats.multicova import multicova
        transposed = self.dataset.mat.transpose()

        if self.dataset.preprocessing_info["Log2-transformed"] is None:
            # needs to be lpog2 transformed for fold change calculations
            transposed = transposed.transform(lambda x: np.log2(x))

        transposed[self.dataset.index_column] = transposed.index
        transposed = transposed.reset_index(drop=True)

        res, _ = multicova.perform_ttest_analysis(
            transposed,
            c1 =list(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1][self.dataset.sample]),                                      
            c2 =list(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2][self.dataset.sample]), 
            s0=0.05, 
            n_perm=self.perm,
            fdr=self.fdr,
            id_col=self.dataset.index_column,
            parallelize=True
        )

        fdr_column = "FDR"  + str(int(self.fdr*100)) + "%"
        df = res[[self.dataset.index_column, 'fc', 'tval', 'pval', 'tval_s0', 'pval_s0', 'qval']]
        df["log2fc"] = res["fc"]
        df["FDR"] = res[fdr_column]

        return df      

    def wald(self) -> pd.DataFrame:
        import diffxpy.api as de
        d = self._prepare_anndata()
        formula_loc = "~ 1 +" + self.column
        
        test = de.test.wald(
            data=d, formula_loc=formula_loc, factor_loc_totest=self.column
        )
        df = test.summary().rename(columns={"gene": self.dataset.index_column})
        return df
    
    def welch_ttest(self) -> pd.DataFrame:
        import diffxpy.api as de
        d = self._prepare_anndata()
        
        test = de.test.t_test(data=d, grouping=self.column)
        df = test.summary().rename(columns={"gene": self.dataset.index_column})
        return df

    def ttest(self) -> pd.DataFrame:
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column] == self.group1][
            self.dataset.sample
        ].tolist()
        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column] == self.group2][
            self.dataset.sample
        ].tolist()
        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.dataset.mat.transpose()
        
        p_values = mat_transpose.apply(
            lambda row: scipy.stats.ttest_ind(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
            )[1],
            axis=1,
        )

        fc = self._calculate_foldchange(
            mat_transpose=mat_transpose, 
            group1_samples=group1_samples, 
            group2_samples=group2_samples
        )
        df = pd.DataFrame()
        df[self.dataset.index_column], df["pval"] = p_values.index.tolist(), p_values.values
        df["log2fc"] = fc  
        return df

    def pairedttest(self) -> pd.DataFrame:
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column] == self.group1][
            self.dataset.sample
        ].tolist()
        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column] == self.group2][
            self.dataset.sample
        ].tolist()
        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.dataset.mat.transpose()
        
        p_values = mat_transpose.apply(
            lambda row: scipy.stats.ttest_rel(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
            )[1],
            axis=1,
        )

        fc = self._calculate_foldchange(
            mat_transpose=mat_transpose, 
            group1_samples=group1_samples, 
            group2_samples=group2_samples
        )
        df = pd.DataFrame()
        df[self.dataset.index_column], df["pval"] = p_values.index.tolist(), p_values.values
        df["log2fc"] = fc
        return df

    def _calculate_foldchange(
        self, mat_transpose: pd.DataFrame, group1_samples: list, group2_samples: list
    ):
        if self.dataset.preprocessing_info["Log2-transformed"]:
            fc = (
                mat_transpose[group1_samples].T.mean().values
                - mat_transpose[group2_samples].T.mean().values
            )
        
        else:
            fc = (
                mat_transpose[group1_samples].T.mean().values
                / mat_transpose[group2_samples].T.mean().values
            )
            fc = np.log2(fc)

        return fc

    def perform(self) -> pd.DataFrame:
        self._check_groups()

        if self.method == "wald":
            df = self.wald()
        
        elif self.method == "ttest":
            df = self.ttest()
        
        elif self.method == "welch-ttest":
            df = self.welch_ttest()
        
        elif self.method == "sam":
            df = self.sam()
        
        elif self.method == "paired-ttest":
            df = self.pairedttest()
        
        else:
            raise ValueError(
                f"{self.method} is invalid choose between 'wald' for Wald-test, 'sam',  and 'ttest', 'welch-ttest' or 'paired-ttest'"
            )
        
        return df




