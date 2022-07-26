from audioop import add
import sklearn
import logging
import plotly.express as px
import plotly
import dash_bio
import scipy
import sklearn.manifold
from alphastats.utils import ignore_warning, check_for_missing_values

# make own alphastats theme
plotly.io.templates["alphastats_colors"] = plotly.graph_objects.layout.Template(
    layout=plotly.graph_objects.Layout(
        colorway=[
            "#009599",
            "#005358",
            "#772173",
            "#B65EAF",
            "#A73A00",
            "#6490C1",
            "#FF894F",
            "#2B5E8B",
            "#A87F32",
        ]
    )
)

plotly.io.templates.default = "simple_white+alphastats_colors"


class Plot:
    @staticmethod
    def _add_circles_to_scatterplot(fig):
        # called by _plot_dimensionality_reduction()
        # convert figure to dict and extract information
        fig_dict = fig.to_plotly_json().get("data")
        for group in fig_dict:
            # get coordinates for the group
            x_vector = group.get("x")
            y_vector = group.get("y")
            # get color of the group to color circle in the same color
            group_color = group.get("marker").get("color")
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=min(x_vector),
                y0=min(y_vector),
                x1=max(x_vector),
                y1=max(y_vector),
                opacity=0.2,
                fillcolor=group_color,
                line_color=group_color,
            )
        return fig

    @check_for_missing_values
    def _plot_dimensionality_reduction(self, group, method, circle, **kwargs):
        # function for plot_pca and plot_tsne
        if self.normalization == "Data is not normalized.":
            logging.info(
                "Data has not been normalized. Data will be normalized using zscore-Normalization"
            )
            self.preprocess(normalization="zscore")

        # subset matrix so it matches with metadata
        if group:
            mat = self._preprocess_subset()
            group_color = self.metadata[group]
        else:
            mat = self.mat
            group_color = group
        mat = mat.fillna(0)

        if method == "pca":
            pca = sklearn.decomposition.PCA(n_components=2)
            components = pca.fit_transform(mat)
            labels = {
                "0": "PC 1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100),
                "1": "PC 2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100),
            }

        elif method == "tsne":
            tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, **kwargs)
            components = tsne.fit_transform(mat)
            labels = {
                "0": "Dimension 1",
                "1": "Dimension 2",
            }

        else:
            # TODO implement UMAP??
            return

        fig = px.scatter(components, x=0, y=1, labels=labels, color=group_color,)

        # draw circles around plotted groups
        if circle is True and group is not None:
            fig = self._add_circles_to_scatterplot(fig)

        return fig

    def plot_pca(self, group=None, circle=False):
        """Plot Principal Component Analysis (PCA)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: PCA plot
        """
        return self._plot_dimensionality_reduction(
            group=group, method="pca", circle=circle
        )

    def plot_tsne(self, group=None, circle=False, perplexity=30, n_iter=1000):
        """Plot t-distributed stochastic neighbor embedding (t-SNE)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: t-SNE plot
        """
        return self._plot_dimensionality_reduction(
            group=group,
            method="tsne",
            circle=circle,
            perplexity=perplexity,
            n_iter=n_iter,
        )

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
            color (_type_, optional): A metadata column used to color the boxes. Defaults to None.
            log_scale (bool, optional): yaxis in logarithmic scale. Defaults to False.

        Returns:
             plotly.graph_objects._figure.Figure: Plotly Sample Distribution Plot
        """

        # create long df
        df = self.mat.unstack().reset_index()
        df.rename(columns={"level_1": "sample", 0: "Intensity"}, inplace=True)

        if color is not None:
            df = df.merge(self.metadata, how="inner", on=["sample"])

        if method == "violin":
            fig = px.violin(df, x="sample", y="Intensity", color=color)

        elif method == "box":
            fig = px.box(df, x="sample", y="Intensity", color=color)

        else:
            raise ValueError(
                f"{method} is not available."
                + "Please select from 'violin' for Violinplot or 'box' for Boxplot."
            )

        if log_scale:
            fig.update_layout(yaxis=dict(type="log"))
        return fig

    def plot_intensity(self, id, group=None, method="violin", log_scale=False):
        """Plot Intensity of individual Protein/ProteinGroup 

        Args:
            id (str): ProteinGroup ID
            group (str, optional): A metadata column used for grouping. Defaults to None.
            method (str, optional):  Violinplot = "violin", Boxplot = "box", Scatterplot = "scatter". Defaults to "violin".
            log_scale (bool, optional): yaxis in logarithmic scale. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: Plotly Plot
        """
        #  TODO use difflib to find similar ProteinId if ProteinGroup is not present
        df = self.mat[[id]].reset_index().rename(columns={"index": "sample"})
        df = df.merge(self.metadata, how="inner", on=["sample"])

        if method == "violin":
            fig = px.violin(df, x=id, y=group, color=group)

        elif method == "box":
            fig = px.box(df, x=id, y=group, color=group)

        elif method == "scatter":
            fig = px.scatter(df, x=id, y=group, color=group)

        else:
            raise ValueError(
                f"{method} is not available."
                + "Please select from 'violin' for Violinplot, 'box' for Boxplot and 'scatter' for Scatterplot."
            )

        if log_scale:
            fig.update_layout(yaxis=dict(type="log"))

        return fig

    @ignore_warning(RuntimeWarning)
    def plot_volcano(self, column, group1, group2):
        """Plot Volcano Plot

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str): name of group to compare needs to be present in column
            group2 (str): name of group to compare needs to be present in column

        Returns:
            _dash_bio: Volcano Plot
        """
        result = self.calculate_ttest_fc(column, group1, group2)
        result = result.dropna().reset_index(drop=True)

        volcano_plot = dash_bio.VolcanoPlot(
            dataframe=result,
            effect_size="foldchange_log2",
            p="pvalue",
            gene=None,
            snp=None,
            annotation="Protein ID",
        )
        return volcano_plot

    @check_for_missing_values
    def plot_heatmap(self):
        """Plot Heatmap with samples as columns and Proteins as rows

        Returns:
            _dash_bio.Clustergram: Dash Bio Clustergram object
        """
        df = self.mat.transpose()
        columns = list(df.columns.values)
        rows = list(df.index)

        plot = dash_bio.Clustergram(
            data=df.loc[rows].values,
            row_labels=rows,
            column_labels=columns,
            color_threshold={"row": 250, "col": 700},
            height=800,
            width=1000,
            color_map=[
                [0.0, "#D0ECE7"],
                [0.25, "#5AA28A"],
                [0.5, "#6C79BB"],
                [0.75, "#8B6CBB"],
                [1.0, "#5B2C6F"],
            ],
            line_width=2,
        )
        return plot

    @check_for_missing_values
    def plot_dendogram(
        self, linkagefun=lambda x: scipy.cluster.hierarchy.linkage(x, "complete")
    ):
        """Plot Hierarichical Clustering Dendogram. This is a wrapper around: 
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
        return fig

    # def plot_line(self):
    #   pass

    # def plot_upset(self):
    #    pass
    # Plotly update figures
    # https://maegul.gitbooks.io/resguides-plotly/content/content/plotting_locally_and_offline/python/methods_for_updating_the_figure_or_graph_objects.html
