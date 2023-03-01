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
from alphastats.plots.IntensityPlot import IntensityPlot
from alphastats.plots.ClusterMap import ClusterMap
from alphastats.utils import ignore_warning, check_for_missing_values


class plotly_object(plotly.graph_objs._figure.Figure):
    plotting_data = None
    preprocessing = None
    method = None


class seaborn_object(sns.matrix.ClusterGrid):
    plotting_data = None
    preprocessing = None
    method = None


plotly.io.templates["alphastats_colors"] = plotly.graph_objects.layout.Template(
    layout=plotly.graph_objects.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            "#009599",
            "#005358",
            "#772173",
            "#B65EAF",  # pink
            "#A73A00",
            "#6490C1",
            "#FF894F",
            "#2B5E8B",
            "#A87F32",
        ],
    )
)

plotly.io.templates.default = "simple_white+alphastats_colors"

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
        perm=100, 
        fdr=0.05,
        compare_preprocessing_modes=False
    ):
        """Plot Volcano Plot

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str/list): name of group to compare needs to be present in column or list of sample names to compare
            group2 (str/list): name of group to compare needs to be present in column  or list of sample names to compare
            method (str): "anova", "wald", "ttest", "SAM" Defaul ttest.
            labels (bool): Add text labels to significant Proteins, Default False.
            alpha(float,optional): p-value cut off.
            min_fc (float): Minimum fold change.
            draw_line(boolean): whether to draw cut off lines.
            perm(float,optional): number of permutations when using SAM as method. Defaults to 100.
            fdr(float,optional): FDR cut off when using SAM as method. Defaults to 0.05.
            compare_preprocessing_modes(bool): Will iterate through normalization and imputation modes and return a list of VolcanoPlots in different settings, Default False.


        Returns:
            plotly.graph_objects._figure.Figure: Volcano Plot
        """

        if compare_preprocessing_modes:
            params_for_func = locals()
            results = self._compare_preprocessing_modes(func=VolcanoPlot,params_for_func=params_for_func)
            return results
        
        else:
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
                perm=perm, 
                fdr=fdr
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

    def plot_intensity(
        self,
        protein_id,
        group=None,
        subgroups=None,
        method="box",
        add_significance=False,
        log_scale=False,
        compare_preprocessing_modes=False
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
        if compare_preprocessing_modes:
            params_for_func = locals()
            results = self._compare_preprocessing_modes(func=IntensityPlot,params_for_func=params_for_func)
            return results
        
        intensity_plot = IntensityPlot(
            dataset = self,
            protein_id=protein_id,
            group=group,
            subgroups=subgroups,
            method=method,
            add_significance=add_significance,
            log_scale=log_scale
        )

        return intensity_plot.plot

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

        clustermap = ClusterMap(
            dataset = self,
            label_bar=label_bar,
            only_significant=only_significant,
            group=group,
            subgroups=subgroups
        )
        return  clustermap.plot

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
