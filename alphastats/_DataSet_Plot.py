import sklearn
import logging
import plotly.express as px
import plotly
import dash_bio
import scipy

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
        if self.imputation == "Data is not imputed." and self.mat.isna().values.any():
            logging.warning(
                "Data contains missing values. Missing values will be replaced with 0. Consider Imputation instead:"
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

    def plot_sampledistribution(self, method="violin", color=None, log_scale=False):
        """Plot Intesity Distribution for each sample. Either Violin or Boxplot

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
        if method == "box":
            fig = px.box(df, x="sample", y="Intensity", color=color)

        if log_scale:
            fig.update_layout(yaxis=dict(type="log"))
        return fig

    def plot_intensity(self, id, group=None, method="violin", log_scale=False):
        """Plot Intensity of individual Protein/ProteinGroup 

        Args:
            id (str): ProteinGroup ID
            group (str, optional): A metadata column used for grouping. Defaults to None.
            method (str, optional):  Violinplot = "violin", Boxplot = "box". Defaults to "violin".
            log_scale (bool, optional): yaxis in logarithmic scale. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: Plotly Plot
        """
        #  TODO use difflib to find similar ProteinId if ProteinGroup is not present
        df = self.mat[[id]].reset_index().rename(columns={"index": "sample"})
        df = df.merge(self.metadata, how="inner", on=["sample"])

        if method == "violin":
            fig = px.violin(df, x=id, y=group, color=group)
        if method == "box":
            fig = px.box(df, x=id, y=group, color=group)

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
        # TODO add option to load DeSeq results???
        result = self.calculate_ttest_fc(column, group1, group2)
        result = result.dropna()
        volcano_plot = dash_bio.VolcanoPlot(
            dataframe=result,
            effect_size="foldchange_log2",
            p="pvalue",
            gene=None,
            snp=None,
            annotation="Protein IDs",
        )
        return volcano_plot

    def plot_heatmap(self):
        """Plot Heatmap with samples as columns and Proteins as rows

        Returns:
            _dash_bio.Clustergram: Dash Bio Clustergram object
        """
        if self.mat.isna().values.any() is True:
            raise ValueError(
                "Data contains missing values. Impute data before plotting: "
                "for instance `DataSet.preprocess(imputation='mean')` or replace NAs with 0."
            )

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
        if self.mat.isna().values.any() is True:
            raise ValueError(
                "Data contains missing values. Impute data before plotting: "
                "for instance `DataSet.preprocess(imputation='mean')` or replace NAs with 0."
            )
        fig = plotly.figure_factory.create_dendrogram(
            self.mat, labels=self.mat.index, linkagefun=linkagefun
        )
        return fig

    def plot_line(self):
        pass

    def plot_upset(self):
        pass
        # Plotly update figures
        # https://maegul.gitbooks.io/resguides-plotly/content/content/plotting_locally_and_offline/python/methods_for_updating_the_figure_or_graph_objects.html
