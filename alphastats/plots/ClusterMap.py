from alphastats.plots.PlotUtils import PlotUtils, seaborn_object

import seaborn as sns
import random


class ClusterMap(PlotUtils):
    def __init__(
        self,
        dataset,
        label_bar,
        only_significant,
        group,
        subgroups
        ):
        self.dataset = dataset
        self.label_bar = label_bar
        self.only_significant = only_significant
        self.group = group
        self.subgroups = subgroups

        self._prepare_df()
        self._plot()


    def _prepare_df(self):
        df = self.dataset.mat.loc[:, (self.dataset.mat != 0).any(axis=0)]

        if self.group is not None and self.subgroups is not None:
            metadata_df = self.dataset.metadata[
                self.dataset.metadata[self.group].isin(self.subgroups + [self.dataset.sample])
            ]
            samples = metadata_df[self.dataset.sample]
            df = df.filter(items=samples, axis=0)

        else:
            metadata_df = self.dataset.metadata

        if self.only_significant and self.group is not None:
            anova_df = self.dataset.anova(column=self.group, tukey=False)
            significant_proteins = anova_df[anova_df["ANOVA_pvalue"] < 0.05][
                self.dataset.index_column
            ].to_list()
            df = df[significant_proteins]

        if self.label_bar is not None:
            self._create_label_bar(
               metadata_df
            )

        self.prepared_df = self.dataset.mat.loc[:, (self.dataset.mat != 0).any(axis=0)].transpose()
  

    def _plot(self):
        fig = sns.clustermap(self.prepared_df, col_colors=self.label_bar)

        if self.label_bar is not None:
           fig = self._add_label_bar(fig)

        # set attributes
        setattr(fig, "plotting_data", self.prepared_df)
        setattr(fig, "preprocessing", self.dataset.preprocessing_info)
        setattr(fig, "method", "clustermap")

        self.plot = fig

    def _add_label_bar(self, fig):
        for label in self.s.unique():
                fig.ax_col_dendrogram.bar(
                    0, 0, color=self.lut[label], label=label, linewidth=0
                )
                fig.ax_col_dendrogram.legend(loc="center", ncol=6)
        return fig

    def _create_label_bar(self, metadata_df):
        colorway = [
            "#009599",
            "#005358",
            "#772173",
            "#B65EAF",
            "#A73A00",
            "#6490C1",
            "#FF894F",
        ]

        self.s = metadata_df[self.label_bar]
        su = self.s.unique()
        colors = sns.light_palette(random.choice(colorway), len(su))
        self.lut = dict(zip(su, colors))
        self.label_bar = self.s.map(self.lut)

       
