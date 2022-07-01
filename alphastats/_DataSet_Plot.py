import sklearn
import logging
import plotly.express as px
import plotly
import dash_bio

plotly.io.templates.default = "simple_white"

class Plot:
    def plot_pca(self, group=None):
        """Plot Principal Component Analysis (PCA)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.

        Returns:
            plotly Object: p
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
            _type_: _description_
        """
        corr_matrix = self.mat.transpose().corr(method=method)
        plot = px.imshow(corr_matrix)
        return plot

    def plot_sampledistribution(self, group=None):
        df = self.mat.unstack().reset_index()
        fig = px.box(df, x="level_0", y=0)
        

    def plot_volcano(self, column, group1, group2):
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

# Plotly update figures
# https://maegul.gitbooks.io/resguides-plotly/content/content/plotting_locally_and_offline/python/methods_for_updating_the_figure_or_graph_objects.html