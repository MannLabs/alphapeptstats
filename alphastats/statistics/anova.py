from typing import List, Union

import pandas as pd
import scipy
from tqdm import tqdm

from alphastats.dataset.keys import Cols
from alphastats.statistics.tukey_test import tukey_test


class Anova:
    def __init__(
        self,
        mat: pd.DataFrame,
        metadata: pd.DataFrame,
        column: str,
        protein_ids: Union[str, List[str]],
        tukey: bool,
    ):
        self.mat: pd.DataFrame = mat
        self.metadata: pd.DataFrame = metadata

        # TODO move these to perform()?
        self.column: str = column
        self.protein_ids: Union[str, List[str]] = protein_ids
        self.tukey: bool = tukey

    def _get_protein_ids_list(self):
        if self.protein_ids == "all":
            self.protein_ids_list = self.mat.columns.tolist()

        elif isinstance(self.protein_ids, str):
            # convert id to list
            self.protein_ids_list = [self.protein_ids]

        else:
            self.protein_ids_list = self.protein_ids

    def _perform_anova(self) -> pd.DataFrame:
        p_values = self.mat_transpose.apply(
            lambda row: scipy.stats.f_oneway(
                *[row[elem].values.flatten() for elem in self.all_groups]
            )[1],
            axis=1,
        )
        anova_df = pd.DataFrame()
        anova_df[Cols.INDEX], anova_df["ANOVA_pvalue"] = (
            p_values.index.tolist(),
            p_values.values,
        )
        return anova_df

    def _prepare_data(self):
        # generated list of list with samples
        subgroup = self.metadata[self.column].unique().tolist()
        self.all_groups = []
        for sub in subgroup:
            group_list = self.metadata[self.metadata[self.column] == sub][
                Cols.SAMPLE
            ].tolist()
            self.all_groups.append(group_list)

        self.mat_transpose = self.mat[self.protein_ids_list].transpose()

    def _create_tukey_df(self, anova_df: pd.DataFrame) -> pd.DataFrame:
        # combine tukey results with anova results
        df = (
            self.mat[self.protein_ids_list]
            .reset_index()
            .rename(columns={"index": Cols.SAMPLE})
        )
        df = df.merge(self.metadata, how="inner", on=[Cols.SAMPLE])
        tukey_df_list = []
        for protein_id in tqdm(self.protein_ids_list):
            tukey_df_list.append(
                tukey_test(
                    df=df,
                    protein_id=protein_id,
                    group=self.column,
                )
            )
        # combine all tukey test results
        tukey_df = pd.concat(tukey_df_list)
        # combine anova and tukey test results
        final_df = anova_df.merge(
            tukey_df[["comparison", "p-tukey", Cols.INDEX]],
            how="inner",
            on=[Cols.INDEX],
        )
        final_df = final_df.pivot(
            index=[Cols.INDEX, "ANOVA_pvalue"],
            columns=["comparison"],
            values="p-tukey",
        )
        return final_df

    def perform(self) -> pd.DataFrame:
        self._get_protein_ids_list()
        self._prepare_data()
        anova_df = self._perform_anova()

        if self.tukey:
            anova_df = self._create_tukey_df(anova_df=anova_df)

        return anova_df
