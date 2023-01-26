import sklearn
import logging
import plotly.express as px
import plotly
import scipy
import sklearn.manifold

import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import random
import itertools
import plotly.figure_factory

from alphastats.plots.DimensionalityReduction import DimensionalityReduction
from alphastats.plots.VolcanoPlot import VolcanoPlot
from alphastats.utils import ignore_warning, check_for_missing_values


class plotly_object(plotly.graph_objs._figure.Figure):
    plotting_data = None
    preprocessing = None
    method = None


class seaborn_object(sns.matrix.ClusterGrid):
    plotting_data = None
    preprocessing = None
    method = None


class Plot:
    def _update_figure_attributes(self, figure_object, plotting_data, method=None):
        setattr(figure_object, "plotting_data", plotting_data)
        setattr(figure_object, "preprocessing", self.preprocessing_info)
        setattr(figure_object, "method", method)
        return figure_object

    @check_for_missing_values
    def plot_pca(self, group=None, circle=False):
        """Plot Principal Component Analysis (PCA)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: PCA plot
        """
        dimensionality_reduction = DimensionalityReduction(
            dataset=self, group=group, method="pca", circle=circle
        )
        return dimensionality_reduction.plot

    @check_for_missing_values
    def plot_tsne(self, group=None, circle=False, perplexity=5, n_iter=1000):
        """Plot t-distributed stochastic neighbor embedding (t-SNE)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: t-SNE plot
        """
        dimensionality_reduction = DimensionalityReduction(
            dataset=self,
            group=group,
            method="tsne",
            circle=circle,
            perplexity=perplexity,
            n_iter=n_iter,
        )
        return dimensionality_reduction.plot

    @check_for_missing_values
    def plot_umap(self, group=None, circle=False):
        """Plot Uniform Manifold Approximation and Projection for Dimension Reduction

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: UMAP plot
        """
        dimensionality_reduction = DimensionalityReduction(
            dataset=self, group=group, method="umap", circle=circle
        )
        return dimensionality_reduction.plot

    def plot_volcano(
        self,
        group1,
        group2,
        column=None,
        method="ttest",
        labels=False,
        min_fc=1,
        alpha=0.05,
        draw_line=True,
    ):
        """Plot Volcano Plot

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str/list): name of group to compare needs to be present in column or list of sample names to compare
            group2 (str/list): name of group to compare needs to be present in column  or list of sample names to compare
            method (str): "anova", "wald", "ttest", Defaul ttest.
            labels (bool): Add text labels to significant Proteins, Default False.
            alpha(float,optional): p-value cut off.
            min_fc (float): Minimum fold change
            draw_line(boolean): whether to draw cut off lines.


        Returns:
            plotly.graph_objects._figure.Figure: Volcano Plot
        """

        volcano_plot = VolcanoPlot(
            dataset=self,
            group1=group1,
            group2=group2,
            column=column,
            method=method,
            labels=labels,
            min_fc=min_fc,
            alpha=alpha,
            draw_line=draw_line,
        )

        return volcano_plot.plot

    def plot_correlation_matrix(self, method="pearson"):
        """Plot Correlation Matrix

        Args:
            method (str, optional): orrelation coefficient "pearson", "kendall" (Kendall Tau correlation)
            or "spearman" (Spearman rank correlation). Defaults to "pearson".

        Returns:
            plotly.graph_objects._figure.Figure: Correlation matrix
        """
        corr_matrix = self.mat.transpose().corr(method=method)
        plot = px.imshow(corr_matrix)
        return plot

    def plot_sampledistribution(self, method="violin", color=None, log_scale=False):
        """Plot Intensity Distribution for each sample. Either Violin or Boxplot

        Args:
            method (str, optional): Violinplot = "violin", Boxplot = "box". Defaults to "violin".
            color (str, optional): A metadata column used to color the boxes. Defaults to None.
            log_scale (bool, optional): yaxis in logarithmic scale. Defaults to False.

        Returns:
             plotly.graph_objects._figure.Figure: Plotly Sample Distribution Plot
        """

        # create long df
        df = self.mat.unstack().reset_index()
        df.rename(columns={"level_1": self.sample, 0: "Intensity"}, inplace=True)

        if color is not None:
            df = df.merge(self.metadata, how="inner", on=[self.sample])

        if method == "violin":
            fig = px.violin(df, x=self.sample, y="Intensity", color=color)

        elif method == "box":
            fig = px.box(df, x=self.sample, y="Intensity", color=color)

        else:
            raise ValueError(
                f"{method} is not available."
                + "Please select from 'violin' for Violinplot or 'box' for Boxplot."
            )

        if log_scale:
            fig.update_layout(yaxis=dict(type="log"))

        fig = plotly_object(fig)
        fig = self._update_figure_attributes(
            figure_object=fig, plotting_data=df, method=method
        )
        return fig

    @staticmethod
    def _add_significance(plot):
        # add sginficance pvalue, and stars to pairwise intensity plot
        plot_dict = plot.to_plotly_json()
        data = plot_dict.get("data")

        if len(data) != 2:
            logging.warning(
                "Signficane can only be estimated when there are two groups plotted."
            )
            return plot

        group1, group2 = data[0]["name"], data[1]["name"]
        y_array1, y_array2 = data[0]["y"], data[1]["y"]
        #  do ttest
        pvalue = scipy.stats.ttest_ind(y_array1, y_array2).pvalue

        pvalue_text = "<i>p=" + str(round(pvalue, 4)) + "</i>"

        if pvalue < 0.001:
            significance_level = "***"
            pvalue_text = "<i>p<0.001</i>"
        elif pvalue < 0.01:
            significance_level = "**"
        elif pvalue < 0.05:
            significance_level = "*"
        else:
            significance_level = "-"

        y_max = np.concatenate((y_array1, y_array2)).max()
        # add connecting bar for pvalue
        plot.add_trace(
            go.Scatter(
                x=[group1, group1, group2, group2],
                y=[y_max * 1.1, y_max * 1.15, y_max * 1.15, y_max * 1.1],
                fill=None,
                mode="lines",
                line=dict(color="rgba(0,0,0,1)", width=1),
                showlegend=False,
            )
        )

        # Add p-values
        plot.add_annotation(
            text=pvalue_text,
            name="p-value",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.95,
            showarrow=False,
            font=dict(size=12, color="black"),
        )

        plot.add_annotation(
            text=significance_level,
            name="significance",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.002,
            showarrow=False,
            font=dict(size=12, color="black"),
        )

        plot.update_layout(width=600, height=700)
        return plot

    def plot_intensity(
        self,
        protein_id,
        group=None,
        subgroups=None,
        method="box",
        add_significance=False,
        log_scale=False,
    ):
        """Plot Intensity of individual Protein/ProteinGroup

        Args:
            ID (str): ProteinGroup ID
            group (str, optional): A metadata column used for grouping. Defaults to None.
            subgroups (list, optional): Select variables from the group column. Defaults to None.
            method (str, optional):  Violinplot = "violin", Boxplot = "box", Scatterplot = "scatter". Defaults to "box".
            add_significance (bool, optional): add p-value bar, only possible when two groups are compared. Defaults False.
            log_scale (bool, optional): yaxis in logarithmic scale. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: Plotly Plot
        """
        #  TODO use difflib to find similar ProteinId if ProteinGroup is not present
        df = self.mat[[protein_id]].reset_index().rename(columns={"index": self.sample})
        df = df.merge(self.metadata, how="inner", on=[self.sample])

        if subgroups is not None:
            df = df[df[group].isin(subgroups)]

        y_label = protein_id + " - " + self.intensity_column.replace("[sample]", "")

        if method == "violin":
            fig = px.violin(
                df, y=protein_id, x=group, color=group, labels={protein_id: y_label}
            )

        elif method == "box":
            fig = px.box(
                df, y=protein_id, x=group, color=group, labels={protein_id: y_label}
            )

        elif method == "scatter":
            fig = px.scatter(
                df, y=protein_id, x=group, color=group, labels={protein_id: y_label}
            )

        else:
            raise ValueError(
                f"{method} is not available."
                + "Please select from 'violin' for Violinplot, 'box' for Boxplot and 'scatter' for Scatterplot."
            )

        if log_scale:
            fig.update_layout(yaxis=dict(type="log"))

        if add_significance:
            fig = self._add_significance(fig)

        fig = plotly_object(fig)
        fig = self._update_figure_attributes(
            figure_object=fig, plotting_data=df, method=method
        )

        return fig

    def _clustermap_create_label_bar(self, label, metadata_df):
        colorway = [
            "#009599",
            "#005358",
            "#772173",
            "#B65EAF",
            "#A73A00",
            "#6490C1",
            "#FF894F",
        ]

        s = metadata_df[label]
        su = s.unique()
        colors = sns.light_palette(random.choice(colorway), len(su))
        lut = dict(zip(su, colors))
        color_label = s.map(lut)

        return color_label, lut, s

    @ignore_warning(UserWarning)
    @check_for_missing_values
    def plot_clustermap(
        self, label_bar=None, only_significant=False, group=None, subgroups=None
    ):
        """Plot a matrix dataset as a hierarchically-clustered heatmap

        Args:
            label_bar (str, optional): column/variable name described in the metadata. Will be plotted as bar above the heatmap to see wheteher groups are clustering together. Defaults to None.. Defaults to None.
            only_significant (bool, optional): performs ANOVA and only signficantly different proteins will be clustered (p<0.05). Defaults to False.
            group (str, optional): group containing subgroups that should be clustered. Defaults to None.
            subgroups (list, optional): variables in group that should be plotted. Defaults to None.

        Returns:
             ClusterGrid: Clustermap
        """

        df = self.mat.loc[:, (self.mat != 0).any(axis=0)]

        if group is not None and subgroups is not None:
            metadata_df = self.metadata[
                self.metadata[group].isin(subgroups + [self.sample])
            ]
            samples = metadata_df[self.sample]
            df = df.filter(items=samples, axis=0)

        else:
            metadata_df = self.metadata

        if only_significant and group is not None:
            anova_df = self.anova(column=group, tukey=False)
            significant_proteins = anova_df[anova_df["ANOVA_pvalue"] < 0.05][
                self.index_column
            ].to_list()
            df = df[significant_proteins]

        if label_bar is not None:
            label_bar, lut, s = self._clustermap_create_label_bar(
                label_bar, metadata_df
            )

        df = self.mat.loc[:, (self.mat != 0).any(axis=0)]

        fig = sns.clustermap(df.transpose(), col_colors=label_bar)

        if label_bar is not None:
            for label in s.unique():
                fig.ax_col_dendrogram.bar(
                    0, 0, color=lut[label], label=label, linewidth=0
                )
                fig.ax_col_dendrogram.legend(loc="center", ncol=6)

        fig = self._update_figure_attributes(
            figure_object=fig, plotting_data=df, method="clustermap"
        )
        return fig

    @check_for_missing_values
    def plot_dendrogram(
        self, linkagefun=lambda x: scipy.cluster.hierarchy.linkage(x, "complete")
    ):
        """Plot Hierarichical Clustering Dendrogram. This is a wrapper around:
        https://plotly.com/python-api-reference/generated/plotly.figure_factory.create_dendrogram.html

        Args:
            linkagefun (_type_, optional): Function to compute the linkage matrix from
                               the pairwise distance. Defaults to lambdax:scipy.cluster.hierarchy.linkage(x, "complete").

        Raises:
            ValueError: If data contains missing values, is not imputed

        Returns:
            plotly.figure_factory.create_dendrogram: dendrogram Plotly figure object
        """
        # of anova results
        # general of a subset of proteins
        fig = plotly.figure_factory.create_dendrogram(
            self.mat, labels=self.mat.index, linkagefun=linkagefun
        )

        fig = plotly_object(fig)
        fig = self._update_figure_attributes(
            figure_object=fig, plotting_data=self.mat, method="dendrogram"
        )
        return fig

    def plot_imputed_values(self):
        # get coordinates of missing values
        df = self.mat
        s = df.stack(dropna=False)
        missing_values_coordinates = [list(x) for x in s.index[s.isna()]]

        # get all coordinates
        coordinates = list(
            itertools.product(list(self.mat.index), list(self.mat.columns))
        )

        # needs to be speed up
        imputed_values, original_values = [], []
        for coordinate in coordinates:
            coordinate = list(coordinate)
            if coordinate in missing_values_coordinates:
                value = self.mat.loc[coordinate[0], coordinate[1]]
                imputed_values.append(value)
            else:
                original_values.append(value)

        label = ["imputed values"] * len(imputed_values) + ["non imputed values"] * len(
            original_values
        )
        values = imputed_values + original_values

        plot_df = pd.DataFrame(
            list(zip(label, values)), columns=["Imputation", "values"]
        )

        fig = px.histogram(
            plot_df,
            x="values",
            color="Imputation",
            opacity=0.8,
            hover_data=plot_df.columns,
        )

        pass
