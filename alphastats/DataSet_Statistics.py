from codecs import ignore_errors
from itertools import permutations
import pandas as pd
import scipy.stats
import numpy as np
import logging
import pingouin
from alphastats.utils import ignore_warning
from tqdm import tqdm


class Statistics:
    @ignore_warning(RuntimeWarning)
    def calculate_ttest_fc(self, column, group1, group2):
        """Calculate t-test and fold change between two groups

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str): name of group to compare needs to be present in column
            group2 (str): name of group to compare needs to be present in column

        Returns:
            pandas.DataFrame: 
            
            pandas Dataframe with foldchange, foldchange_log2 and pvalue
            for each ProteinID/ProteinGroup between group1 and group2.

            * ``'Protein ID'``: ProteinID/ProteinGroup
            * ``'pvalue'``: p-value result of t-test
            * ``'foldchange'``: foldchange of the mean Protein Intensity of group1 vs. group2
            * ``'foldchange_log2'``: log2(foldchange)

        """
        # get samples names of two groupes
        group1_samples = self.metadata[self.metadata[column] == group1][
            "sample"
        ].tolist()
        group2_samples = self.metadata[self.metadata[column] == group2][
            "sample"
        ].tolist()

        if len(group1_samples) == 1 or len(group2_samples) == 1:
            raise NotImplementedError("Sample number too low to calculate t-test")

        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.mat.transpose()
        # add small value so no division by zero
        # https://www.researchgate.net/post/can-anyone-tell-me-how-to-Calculate-fold-change-in-RNA-seq-data-if-we-have-change-fromzero-tosomething-and-something-to-zero-in
        # https://github.com/swiri021/Difference_test_with_permutation
        # Theis Lab: https://github.com/theislab/diffxpy
        # https://github.com/staslist/A-Lister

        mat_transpose += 0.00001
        # np.aps()
        fc = (
            mat_transpose[group1_samples].T.mean().values
            / mat_transpose[group2_samples].T.mean().values
        )

        p_values = mat_transpose.apply(
            lambda row: scipy.stats.ttest_ind(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
                permutations=100,
            )[1],
            axis=1,
        )
        df = pd.DataFrame()
        df["Protein ID"], df["pvalue"] = p_values.index.tolist(), p_values.values
        df["foldchange"], df["foldchange_log2"] = fc, np.log2(fc)
        return df

    @ignore_warning(RuntimeWarning)
    def calculate_tukey(self, protein_id, group, df=None):
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
                self.mat[[protein_id]].reset_index().rename(columns={"index": "sample"})
            )
            df = df.merge(self.metadata, how="inner", on=["sample"])

        try:
            tukey_df = pingouin.pairwise_tukey(data=df, dv=protein_id, between=group)
            tukey_df["comparison"] = tukey_df["A"] + " vs. " + tukey_df["B"] + " Tukey Test"
            tukey_df["Protein ID"] = protein_id
        
        except ValueError:
            tukey_df =  pd.DataFrame()
        
        return tukey_df

    @ignore_warning(RuntimeWarning)
    def anova(self, group, protein_ids="all", tukey=True):
        """One-way Analysis of Variance (ANOVA)

        Args:
            group (str): A metadata column used to calculate ANOVA
            ids (str or list, optional): ProteinIDs to calculate ANOVA for - dependend variable either ProteinID as string, several ProteinIDs as list or "all" to calculate ANOVA for all ProteinIDs. Defaults to "all".
            tukey (bool, optional): Whether to calculate a Tukey-HSD post-hoc test. Defaults to True.

        Returns:
            pandas.DataFrame:
            * ``'Protein ID'``: ProteinID/ProteinGroup
            * ``'ANOVA_pvalue'``: p-value of ANOVA
            * ``'A vs. B Tukey test'``: Tukey-HSD corrected p-values (each combination represents a column)
        """
        if protein_ids == "all":
            protein_ids_list = self.mat.columns.tolist()
        elif isinstance(protein_ids, str):
            # convert id to list
            protein_ids_list = [protein_ids]
        else:
            protein_ids_list = protein_ids
        #  generated list of list with samples
        subgroup = self.metadata[group].unique().tolist()
        all_groups = []
        for sub in subgroup:
            group_list = self.metadata[self.metadata[group] == sub]["sample"].tolist()
            all_groups.append(group_list)

        mat_transpose = self.mat[protein_ids_list].transpose()
        #  perform rowwise anova
        p_values = mat_transpose.apply(
            # generate nested list with values for each group
            # TODO this can print long list of warnings work around this
            lambda row: scipy.stats.f_oneway(
                *[row[elem].values.flatten() for elem in all_groups]
            )[1],
            axis=1,
        )
        anova_df = pd.DataFrame()
        anova_df["Protein ID"], anova_df["ANOVA_pvalue"] = (
            p_values.index.tolist(),
            p_values.values,
        )

        if tukey:
            final_df = self._create_tukey_df(
                anova_df=anova_df, protein_ids_list=protein_ids_list, group=group
            )
        else:
            final_df = anova_df
        return final_df

    def _create_tukey_df(self, anova_df, protein_ids_list, group):
        #  combine tukey results with anova results
        df = (
            self.mat[protein_ids_list].reset_index().rename(columns={"index": "sample"})
        )
        df = df.merge(self.metadata, how="inner", on=["sample"])
        tukey_df_list = []
        for protein_id in tqdm(protein_ids_list):
            tukey_df_list.append(
                self.calculate_tukey(df=df, protein_id=protein_id, group=group)
            )
        # combine all tukey test results
        tukey_df = pd.concat(tukey_df_list)
        # combine anova and tukey test results
        final_df = anova_df.merge(
            tukey_df[["comparison", "p-tukey", "Protein ID"]],
            how="inner",
            on=["Protein ID"],
        )
        final_df = final_df.pivot(
            index=["Protein ID", "ANOVA_pvalue"],
            columns=["comparison"],
            values="p-tukey",
        )
        return final_df

    def ancova(self, protein_id, covar, between):
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
        df = self.mat[protein_id].reset_index().rename(columns={"index": "sample"})
        df = self.metadata.merge(df, how="inner", on=["sample"])
        ancova_df = pingouin.ancova(df, dv=protein_id, covar=covar, between=between)
        return ancova_df
