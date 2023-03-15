
import pandas as pd
import scipy
import tqdm

class Anova:
    def __init__(self, dataset, column, protein_ids):
        self.column = column
        self.protein_ids = protein_ids
        
    def _get_protein_ids_list(self):
        if self.protein_ids == "all":
            self.protein_ids_list = self.dataset.mat.columns.tolist()
        
        elif isinstance(self.protein_ids, str):
            # convert id to list
            self.protein_ids_list = [self.protein_ids]
       
        else:
            self.protein_ids_list = self.protein_ids

    def get_pvalues(self):
        p_values = self.mat_transpose.apply(
            # generate nested list with values for each group
            # TODO this can print long list of warnings work around this
            lambda row: scipy.stats.f_oneway(
                *[row[elem].values.flatten() for elem in self.all_groups]
            )[1],
            axis=1,
        )

    def _prepare_data(self):
        #  generated list of list with samples
        subgroup = self.dataset.metadata[self.column].unique().tolist()
        self.all_groups = []
        for sub in subgroup:
            group_list = self.metadata[self.dataset.metadata[self.column] == sub][
                self.dataset.sample
            ].tolist()
            self.all_groups.append(group_list)

        self.mat_transpose = self.dataset.mat[self.protein_ids_list].transpose()
           
    def _create_tukey_df(self, anova_df, protein_ids_list, group) -> pd.DataFrame:
        #  combine tukey results with anova results
        df = (
            self.mat[protein_ids_list]
            .reset_index()
            .rename(columns={"index": self.sample})
        )
        df = df.merge(self.metadata, how="inner", on=[self.sample])
        tukey_df_list = []
        for protein_id in tqdm(protein_ids_list):
            tukey_df_list.append(
                self.tukey_test(df=df, protein_id=protein_id, group=group)
            )
        # combine all tukey test results
        tukey_df = pd.concat(tukey_df_list)
        # combine anova and tukey test results
        final_df = anova_df.merge(
            tukey_df[["comparison", "p-tukey", self.index_column]],
            how="inner",
            on=[self.index_column],
        )
        final_df = final_df.pivot(
            index=[self.index_column, "ANOVA_pvalue"],
            columns=["comparison"],
            values="p-tukey",
        )
        return final_df
