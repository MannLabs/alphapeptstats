
import pandas as pd
import scipy
from tqdm import tqdm

class Anova:
    def __init__(self, dataset, column, protein_ids, tukey):
        self.dataset = dataset
        self.column = column
        self.protein_ids = protein_ids
        self.tukey = tukey
        
    def _get_protein_ids_list(self):
        if self.protein_ids == "all":
            self.protein_ids_list = self.dataset.mat.columns.tolist()
        
        elif isinstance(self.protein_ids, str):
            # convert id to list
            self.protein_ids_list = [self.protein_ids]
       
        else:
            self.protein_ids_list = self.protein_ids

    def perform_anova(self) -> pd.DataFrame:
        p_values = self.mat_transpose.apply(
            lambda row: scipy.stats.f_oneway(
                *[row[elem].values.flatten() for elem in self.all_groups]
            )[1],
            axis=1,
        )
        anova_df = pd.DataFrame()
        anova_df[self.dataset.index_column], anova_df["ANOVA_pvalue"] = (
            p_values.index.tolist(),
            p_values.values,
        )
        return anova_df

    def _prepare_data(self):
        #  generated list of list with samples
        subgroup = self.dataset.metadata[self.column].unique().tolist()
        self.all_groups = []
        for sub in subgroup:
            group_list = self.dataset.metadata[self.dataset.metadata[self.column] == sub][
                self.dataset.sample
            ].tolist()
            self.all_groups.append(group_list)

        self.mat_transpose = self.dataset.mat[self.protein_ids_list].transpose()
           
    def _create_tukey_df(self, anova_df: pd.DataFrame) -> pd.DataFrame:
        #  combine tukey results with anova results
        df = (
            self.dataset.mat[self.protein_ids_list]
            .reset_index()
            .rename(columns={"index": self.dataset.sample})
        )
        df = df.merge(self.dataset.metadata, how="inner", on=[self.dataset.sample])
        tukey_df_list = []
        for protein_id in tqdm(self.protein_ids_list):
            tukey_df_list.append(
                self.dataset.tukey_test(df=df, protein_id=protein_id, group=self.column)
            )
        # combine all tukey test results
        tukey_df = pd.concat(tukey_df_list)
        # combine anova and tukey test results
        final_df = anova_df.merge(
            tukey_df[["comparison", "p-tukey", self.dataset.index_column]],
            how="inner",
            on=[self.dataset.index_column],
        )
        final_df = final_df.pivot(
            index=[self.dataset.index_column, "ANOVA_pvalue"],
            columns=["comparison"],
            values="p-tukey",
        )
        return final_df
    
    def perform(self) -> pd.DataFrame:
        self._get_protein_ids_list()
        self._prepare_data()
        anova_df = self.perform_anova()
        
        if self.tukey:
            anova_df = self._create_tukey_df(anova_df=anova_df)
        
        return anova_df

