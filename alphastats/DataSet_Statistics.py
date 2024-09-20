from codecs import ignore_errors
from itertools import permutations
import pandas as pd
import scipy.stats
import numpy as np
import pingouin
from alphastats.utils import ignore_warning
from tqdm import tqdm
from functools import lru_cache
from typing import Union

from alphastats.statistics.MultiCovaAnalysis import MultiCovaAnalysis
from alphastats.statistics.DifferentialExpressionAnalysis import DifferentialExpressionAnalysis
from alphastats.statistics.Anova import Anova


class Statistics:
    def _calculate_foldchange(
        self, mat_transpose: pd.DataFrame, group1_samples: list, group2_samples: list
    ) -> pd.DataFrame:
        if self.preprocessing_info["Log2-transformed"]:
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

        return pd.DataFrame({"log2fc": fc, self.index_column: mat_transpose.index})
    
    def _add_metadata_column(self, group1_list: list, group2_list: list):

        # create new column in metadata with defined groups
        metadata = self.metadata

        sample_names = metadata[self.sample].to_list()
        misc_samples = list(set(group1_list + group2_list) - set(sample_names))
        if len(misc_samples) > 0:
            raise ValueError(
                f"Sample names: {misc_samples} are not described in Metadata."
            )

        column = "_comparison_column"
        conditons = [
            metadata[self.sample].isin(group1_list),
            metadata[self.sample].isin(group2_list),
        ]
        choices = ["group1", "group2"]
        metadata[column] = np.select(conditons, choices, default=np.nan)
        self.metadata = metadata

        return column, "group1", "group2"
    
    @ignore_warning(RuntimeWarning)
    def diff_expression_analysis(
            self,
            group1: Union[str, list],
            group2: Union[str, list],
            column: str = None,
            method: str = "ttest",
            perm: int = 10,
            fdr: float = 0.05,
        ) -> pd.DataFrame:
        """Perform differential expression analysis doing a a t-test or Wald test. A wald test will fit a generalized linear model.

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str/list): name of group to compare needs to be present in column or list of sample names to compare
            group2 (str/list): name of group to compare needs to be present in column  or list of sample names to compare
            method (str,optional): statistical method to calculate differential expression, for Wald-test 'wald', paired t-test 'paired-ttest'. Default 'ttest'

        Returns:
            pandas.DataFrame:
            pandas Dataframe with foldchange, foldchange_log2 and pvalue
            for each ProteinID/ProteinGroup between group1 and group2.

            * ``'Protein ID'``: ProteinID/ProteinGroup
            * ``'pval'``: p-value of the ProteinID/ProteinGroup
            * ``'qval'``: multiple testing - corrected p-value
            * ``'log2fc'``: log2(foldchange)
            * ``'grad'``: the gradient of the log-likelihood
            * ``'coef_mle'``: the maximum-likelihood estimate of coefficient in liker-space
            * ``'coef_sd'``: the standard deviation of the coefficient in liker-space
            * ``'ll'``: the log-likelihood of the estimation
        """
        df = DifferentialExpressionAnalysis(
            dataset=self, 
            group1=group1, 
            group2=group2, 
            column=column, method=method,
            perm=perm, 
            fdr=fdr
        ).perform()
        return df

    @ignore_warning(RuntimeWarning)
    def tukey_test(self, protein_id:str, group:str, df: pd.DataFrame=None) -> pd.DataFrame:
        """Calculate Pairwise Tukey-HSD post-hoc test
        Wrapper around:
        https://pingouin-stats.org/generated/pingouin.pairwise_tukey.html#pingouin.pairwise_tukey

        Args:
            protein_id (str): ProteinID to calculate Pairwise Tukey-HSD post-hoc test - dependend variable
            group (str): A metadata column used calculate pairwise tukey
            df (pandas.DataFrame, optional): Defaults to None.

        Returns:
            pandas.DataFrame:
            * ``'A'``: Name of first measurement
            * ``'B'``: Name of second measurement
            * ``'mean(A)'``: Mean of first measurement
            * ``'mean(B)'``: Mean of second measurement
            * ``'diff'``: Mean difference (= mean(A) - mean(B))
            * ``'se'``: Standard error
            * ``'T'``: T-values
            * ``'p-tukey'``: Tukey-HSD corrected p-values
            * ``'hedges'``: Hedges effect size (or any effect size defined in
            ``effsize``)
            * ``'comparison'``: combination of measurment
            * ``'Protein ID'``: ProteinID/ProteinGroup
        """
        if df is None:
            df = (
                self.mat[[protein_id]]
                .reset_index()
                .rename(columns={"index": self.sample})
            )
            df = df.merge(self.metadata, how="inner", on=[self.sample])

        try:
            tukey_df = pingouin.pairwise_tukey(data=df, dv=protein_id, between=group)
            tukey_df["comparison"] = (
                tukey_df["A"] + " vs. " + tukey_df["B"] + " Tukey Test"
            )
            tukey_df[self.index_column] = protein_id

        except ValueError:
            tukey_df = pd.DataFrame()

        return tukey_df

    @ignore_warning(RuntimeWarning)
    def anova(self, column:str, protein_ids="all", tukey: bool=True) -> pd.DataFrame:
        """One-way Analysis of Variance (ANOVA)

        Args:
            column (str): A metadata column used to calculate ANOVA
            protein_ids (str or list, optional): ProteinIDs to calculate ANOVA for - dependend variable either ProteinID as string, several ProteinIDs as list or "all" to calculate ANOVA for all ProteinIDs. Defaults to "all".
            tukey (bool, optional): Whether to calculate a Tukey-HSD post-hoc test. Defaults to True.

        Returns:
            pandas.DataFrame:
            * ``'Protein ID'``: ProteinID/ProteinGroup
            * ``'ANOVA_pvalue'``: p-value of ANOVA
            * ``'A vs. B Tukey test'``: Tukey-HSD corrected p-values (each combination represents a column)
        """
        return Anova(dataset=self, column=column, protein_ids=protein_ids, tukey=tukey).perform()
        

    @lru_cache(maxsize=20)
    def ancova(self, protein_id:str, covar: Union[str, list], between:str) -> pd.DataFrame:
        """Analysis of covariance (ANCOVA) with on or more covariate(s).
        Wrapper around = https://pingouin-stats.org/generated/pingouin.ancova.html

        Args:
            protein_id (str): ProteinID/ProteinGroup - dependent variable
            covar (str or list):   Name(s) of column(s) in metadata with the covariate.
            between (str): Name of column in data with the between factor.

        Returns:
            pandas.Dataframe:

            ANCOVA summary:

            * ``'Source'``: Names of the factor considered
            * ``'SS'``: Sums of squares
            * ``'DF'``: Degrees of freedom
            * ``'F'``: F-values
            * ``'p-unc'``: Uncorrected p-values
            * ``'np2'``: Partial eta-squared
        """
        df = self.mat[protein_id].reset_index().rename(columns={"index": self.sample})
        df = self.metadata.merge(df, how="inner", on=[self.sample])
        ancova_df = pingouin.ancova(df, dv=protein_id, covar=covar, between=between)
        return ancova_df

    @ignore_warning(RuntimeWarning)
    def multicova_analysis(
        self,
        covariates: list,
        n_permutations: int = 3,
        fdr: float = 0.05,
        s0: float = 0.05,
        subset: dict = None,
    ) -> Union[pd.DataFrame, list]:
        """Perform Multicovariat Analysis
        will return a pandas DataFrame with the results and a list of volcano plots (for each covariat)

        Args:
            covariates (list): list of covariates, column names in metadata
            n_permutations (int, optional): number of permutations. Defaults to 3.
            fdr (float, optional): False Discovery Rate. Defaults to 0.05.
            s0 (float, optional): . Defaults to 0.05.
            subset (dict, optional): for categorical covariates . Defaults to None.

        Returns:
            pd.DataFrame: Multicova Analysis results
        """
        
        res, plot_list= MultiCovaAnalysis(
            dataset=self,
            covariates=covariates,
            n_permutations=n_permutations,
            fdr=fdr,
            s0=s0,
            subset=subset,
            plot=True
        ).calculate()
        return res, plot_list

        