import pandas as pd
import scipy.stats
import numpy as np

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
        # calculate fold change (if its is not logarithimic normalized)
        mat_transpose = self.mat.transpose()
        if self.normalization != "log":
            fc = (
                mat_transpose[group1_samples].T.mean().values
                / mat_transpose[group2_samples].T.mean().values
            )

        p_values = mat_transpose.apply(
            lambda row: scipy.stats.ttest_ind(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
            )[1],
            axis=1,
        )
        df = pd.DataFrame()
        df["Protein IDs"], df["pvalue"] = p_values.index.tolist(), p_values.values
        df["foldchange"], df["foldchange_log2"] = fc, np.log2(fc)
        return df