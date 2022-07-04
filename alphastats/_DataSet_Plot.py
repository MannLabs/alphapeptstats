import sklearn
import logging
import plotly.express as px
import plotly
import dash_bio

# make own alphastats theme
plotly.io.templates["alphastats_colors"] = plotly.graph_objects.layout.Template(
    layout=plotly.graph_objects.Layout(
        colorway=[
            "#489B80",
            "#B65EAF",
            "#7593D6",
            "#A475D6",
            "#C177B3",
            "#468189",
            "#9DBEBB",
            "#65676F",
            "#ABCCA6",
            "#B395F2",
            "#535F97",
            "#A7D87F",
            "#F3A020",
            "#20CAF3",
        ]
    )
)
plotly.io.templates.default = "simple_white+alphastats_colors"


class Plot:
    def plot_pca(self, group=None):
        """Plot Principal Component Analysis (PCA)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.

        Returns:
            plotly.graph_objects._figure.Figure: PCA plot
        """
        if self.imputation is None and self.mat.isna().values.any():
            logging.warning(
                "Data contains missing values. Missing values will be replaced with 0. Consider Imputation:"
                "for instance `DataSet.preprocess(imputation='mean')`."
            )

        if self.normalization == "Data is not normalized.":
            logging.info(
                "Data has not been normalized. Data will be normalized using zscore-Normalization"
            )
            self.preprocess(normalization="zscore")

        # subset matrix so it matches with metadata
        if group:
            mat = self.preprocess_subset()
            group_color = self.metadata[group]
        else:
            mat = self.mat
            group_color = group
        mat = mat.fillna(0)

        pca = sklearn.decomposition.PCA(n_components=2)
        components = pca.fit_transform(mat)

        fig = px.scatter(
            components,
            x=0,
            y=1,
            labels={
                "0": "PC 1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100),
                "1": "PC 2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100),
            },
            color=group_color,
        )
        return fig

    def plot_correlation_matrix(self, method="pearson", save_figure=False):
        """Plot Correlation Matrix

        Args:
            method (str, optional): orrelation coefficient "pearson", "kendall" (Kendall Tau correlation) 
            or "spearman" (Spearman rank correlation). Defaults to "pearson".
            save_figure (bool, optional): _description_. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: Correlation matrix
        """
        corr_matrix = self.mat.transpose().corr(method=method)
        plot = px.imshow(corr_matrix)
        return plot

    def plot_sampledistribution(self, method="violin", color=None, log_scale=True):
        """Plot Intesity Distribution for each sample. Either Violin or Boxplot

        Args:
            method (str, optional): Violinplot = "violin", Boxplot = "box". Defaults to "violin".
            color (_type_, optional): A metadata column used to color the boxes. Defaults to None.
            log_scale (bool, optional): plot in logarithmic scale. Defaults to True.

        Returns:
             plotly.graph_objects._figure.Figure: Sample Distribution Plot
        """

        # create long df
        df = self.mat.unstack().reset_index()
        df.rename(columns={"level_1": "sample", 0: "Intensity"}, inplace=True)

        if color is not None:
            df = df.merge(self.metadata, how="inner", on=["sample"])

        if method == "violin":
            fig = px.violin(df, x="sample", y="Intensity", color=color)
        if method == "box":
            fig = px.box(df, x="sample", y="Intensity", color=color)

        if log_scale:
            fig.update_layout(yaxis=dict(type="log"))
        return fig

    def plot_volcano(self, column, group1, group2):
        """Plot Volcano Plot

        Args:
            column (_type_): _description_
            group1 (_type_): _description_
            group2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        result = self.calculate_ttest_fc(column, group1, group2)
        result = result.drop_na()
        volcano_plot = dash_bio.VolcanoPlot(
            dataframe=result,
            effect_size="foldchange_log2",
            p="pvalue",
            gene=None,
            snp=None,
            annotation="Protein IDs",
        )
        return volcano_plot

    def plot_hierarchialclustering(self):
        # of anova results
        # general of a subset of proteins

        # use dash bio
        pass

    def plot_line(self):
        pass

    def plot_box(self):
        pass

    def plot_upset(self):
        pass
        # Plotly update figures
        # https://maegul.gitbooks.io/resguides-plotly/content/content/plotting_locally_and_offline/python/methods_for_updating_the_figure_or_graph_objects.html

        """Plot sample distribution

        Returns:
            plotly.graph_objects._figure.Figure: Plot
        """
