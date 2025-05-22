from typing import Dict, Tuple, Union

import pandas as pd
import pingouin

from alphastats.dataset.keys import Cols
from alphastats.dataset.utils import ignore_warning
from alphastats.statistics.anova import Anova
from alphastats.statistics.differential_expression_analysis import (
    DifferentialExpressionAnalysis,
)
from alphastats.statistics.multicova_analysis import MultiCovaAnalysis


class Statistics:
    def __init__(
        self,
        *,
        mat: pd.DataFrame,
        metadata: pd.DataFrame,
        preprocessing_info: Dict,
    ):
        self.mat: pd.DataFrame = mat
        self.metadata: pd.DataFrame = metadata
        self.preprocessing_info: Dict = preprocessing_info

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
            mat=self.mat,
            metadata=self.metadata,
            preprocessing_info=self.preprocessing_info,
            group1=group1,
            group2=group2,
            column=column,
            method=method,
            perm=perm,
            fdr=fdr,
        ).perform()
        return df

    @ignore_warning(RuntimeWarning)
    def anova(self, column: str, protein_ids="all", tukey: bool = True) -> pd.DataFrame:
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
        return Anova(
            mat=self.mat,
            metadata=self.metadata,
            column=column,
            protein_ids=protein_ids,
            tukey=tukey,
        ).perform()

    def ancova(
        self, protein_id: str, covar: Union[str, list], between: str
    ) -> pd.DataFrame:
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
        df = self.mat[protein_id].reset_index().rename(columns={"index": Cols.SAMPLE})
        df = self.metadata.merge(df, how="inner", on=[Cols.SAMPLE])
        ancova_df = pingouin.ancova(df, dv=protein_id, covar=covar, between=between)
        return ancova_df

    @ignore_warning(RuntimeWarning)
    def multicova_analysis(  # TODO never used outside of tests .. how does this relate to multicova.py?
        self,
        covariates: list,
        n_permutations: int = 3,
        fdr: float = 0.05,
        s0: float = 0.05,
        subset: dict = None,
    ) -> Tuple[pd.DataFrame, list]:
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

        res, plot_list = MultiCovaAnalysis(
            mat=self.mat,
            metadata=self.metadata,
            covariates=covariates,
            n_permutations=n_permutations,
            fdr=fdr,
            s0=s0,
            subset=subset,
        ).calculate()

        return res, plot_list
