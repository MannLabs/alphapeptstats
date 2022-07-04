from itertools import permutations
import pandas as pd
import scipy.stats
import numpy as np
import logging


class Statistics:
    def calculate_ttest_fc(self, column, group1, group2):
        """Calculate t-test and fold change between two groups

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str): name of group to compare needs to be present in column
            group2 (str): nme of group to compare needs to be present in column

        Returns:
            pandas Dataframe: pandas Dataframe with foldchange, foldchange_log2 and pvalue
            for each ProteinID/ProteinGroup between group1 and group2
        """
        # get samples names of two groupes
        group1_samples = self.metadata[self.metadata[column] == group1][
            "sample"
        ].tolist()
        group2_samples = self.metadata[self.metadata[column] == group2][
            "sample"
        ].tolist()

        if len(group1_samples)==1 or len(group2_samples)==1:
            logging.warning("Sample number too low to calculate t-test")
            raise NotImplementedError

        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.mat.transpose()
        # add small value so no division by zero
        # https://www.researchgate.net/post/can-anyone-tell-me-how-to-Calculate-fold-change-in-RNA-seq-data-if-we-have-change-fromzero-tosomething-and-something-to-zero-in
        # TODO consider implemetation with permutations
        # https://github.com/swiri021/Difference_test_with_permutation
        # Theis Lab: https://github.com/theislab/diffxpy
        # https://github.com/staslist/A-Lister

        mat_transpose += 0.00000001
        if self.normalization != "log":
            fc = (
                mat_transpose[group1_samples].T.mean().values
                / mat_transpose[group2_samples].T.mean().values
            )

        p_values = mat_transpose.apply(
            lambda row: scipy.stats.ttest_ind(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
                permutations=100
            )[1],
            axis=1,
        )
        df = pd.DataFrame()
        df["Protein IDs"], df["pvalue"] = p_values.index.tolist(), p_values.values
        df["foldchange"], df["foldchange_log2"] = fc, np.log2(fc)
        return df

    def anova(self):
        # follow up tukey
        pass

    def anocova(self):
        pass
