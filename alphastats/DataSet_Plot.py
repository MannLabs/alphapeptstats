from audioop import add
from turtle import color
import sklearn
import logging
import plotly.express as px
import plotly
import scipy
import sklearn.manifold
from alphastats.utils import ignore_warning, check_for_missing_values
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import random
from umap import UMAP

# make own alphastats theme
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


class plotly_object(plotly.graph_objs._figure.Figure):
    plotting_data = None
    preprocessing = None
    method = None


class seaborn_object(plotly.graph_objs._figure.Figure):
    plotting_data = None
    preprocessing = None
    method = None


class Plot:
    @staticmethod
    def _update_colors_plotly(fig, color_dict):
        #  plotly doesnt allow to assign color to certain group
        # update instead the figure in form of a dict
        # color_dict with group_variable/legendgroup as key, and corresponding color as value
        fig_dict = fig.to_plotly_json()
        data_dict_list = fig_dict.get("data")
        for count, group in enumerate(data_dict_list):
            group_variable = group.get("legendgroup")
            group_color = color_dict.get(group_variable)
            fig_dict["data"][count]["marker"]["color"] = group_color
        # convert dict back to plotly figure
        return go.Figure(fig_dict)

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

    def _update_figure_attributes(self, figure_object, plotting_data, method=None):
        setattr(figure_object, "plotting_data", plotting_data)
        setattr(figure_object, "preprocessing", self.preprocessing_info)
        setattr(figure_object, "method", method)
        return figure_object

    def _volcano_add_labels(self, result_df, figure_object):

        if self.gene_names is not None:
            label_column = self.gene_names
        else:
            label_column = self.index_column

        result_df["label"] = np.where(
            result_df.color != "non-significant", result_df[label_column], ""
        )
        #  replace nas with empty string (can cause error when plotting with gene names)
        result_df["label"] = result_df["label"].fillna("")
        result_df = result_df[result_df["label"] != ""]

        for x, y, label_column in result_df[
            ["log2fc", "-log10(p-value)", label_column]
        ].itertuples(index=False):
            figure_object.add_annotation(
                x=x, y=y, text=label_column, showarrow=False, yshift=10
            )
        return figure_object

    @check_for_missing_values
    def _plot_dimensionality_reduction(self, group, method, circle, **kwargs):
        # function for plot_pca and plot_tsne
        # subset matrix so it matches with metadata
        if group:
            mat = self._subset()
            self.metadata[group] = self.metadata[group].apply(str)
            group_color = self.metadata[group]
            sample_names = self.metadata[self.sample].to_list()
        else:
            mat = self.mat
            group_color = group
            sample_names = mat.reset_index(level=0)["index"].to_list()
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

        elif method == "umap":
            umap_2d = UMAP(n_components=2, init="random", random_state=0)
            components = umap_2d.fit_transform(mat)
            labels = {
                "0": "",
                "1": "",
            }

        components = pd.DataFrame(components)
        components[self.sample] = sample_names

        fig = px.scatter(
            components,
            x=0,
            y=1,
            labels=labels,
            color=group_color,
            hover_data=[components[self.sample]],
        )

        # rename hover_data_0 to sample
        fig_dict = fig.to_plotly_json()
        data = fig_dict.get("data")

        for count, d in enumerate(data):
            hover = d.get("hovertemplate").replace("hover_data_0", "sample")
            fig_dict["data"][count]["hovertemplate"] = hover
        fig = go.Figure(fig_dict)

        #  save plotting data in figure object
        fig = plotly_object(fig)
        fig = self._update_figure_attributes(
            figure_object=fig, plotting_data=pd.DataFrame(components), method=method
        )

        # draw circles around plotted groups
        if circle is True and group is not None:
            fig = self._add_circles_to_scatterplot(fig)

        if group:
            fig.update_layout(legend_title_text=group)

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

    def plot_umap(self, group=None, circle=False):
        """Plot Uniform Manifold Approximation and Projection for Dimension Reduction

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: UMAP plot
        """
        return self._plot_dimensionality_reduction(
            group=group, method="umap", circle=circle
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

        return fig

    @ignore_warning(UserWarning)
    @ignore_warning(RuntimeWarning)
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

        if isinstance(group1, list) and isinstance(group2, list):
            column, group1, group2 = self._add_metadata_column(group1, group2)

        if column is None:
            raise ValueError(
                "Column containing group1 and group2 needs to be specified"
            )

        if method == "wald":

            print(
                "Calculating differential expression analysis using wald test. Fitting generalized linear model..."
            )
            result_df = self.perform_diff_expression_analysis(
                column=column, group1=group1, group2=group2, method="wald"
            )
            pvalue_column = "qval"

        elif method == "ttest":

            print("Calculating t-test...")
            result_df = self.perform_diff_expression_analysis(
                column=column, group1=group1, group2=group2, method="ttest"
            )
            pvalue_column = "pval"

        elif method == "anova":

            print("Calculating ANOVA with follow-up tukey test...")

            result_df = self.anova(column=column, protein_ids="all", tukey=True)
            group1_samples = self.metadata[self.metadata[column] == group1][
                self.sample
            ].tolist()
            group2_samples = self.metadata[self.metadata[column] == group2][
                self.sample
            ].tolist()

            mat_transpose = self.mat.transpose()
            fc = self._calculate_foldchange(
                mat_transpose, group1_samples, group2_samples
            )

            #  check how column is ordered
            pvalue_column = group1 + " vs. " + group2 + " Tukey Test"

            if pvalue_column not in fc.columns:
                pvalue_column = group2 + " vs. " + group1 + " Tukey Test"

            result_df = result_df.reset_index().merge(
                fc.reset_index(), on=self.index_column
            )

        else:
            raise ValueError(
                f"{method} is not available."
                + "Please select from 'ttest' or 'anova' for anova with follow up tukey or 'wald' for wald-test using."
            )

        result_df = result_df[(result_df["log2fc"] < 10) & (result_df["log2fc"] > -10)]
        result_df["-log10(p-value)"] = -np.log10(result_df[pvalue_column])

        alpha = -np.log10(alpha)
        # add color variable to plot
        condition = [
            (result_df["log2fc"] < -min_fc) & (result_df["-log10(p-value)"] > alpha),
            (result_df["log2fc"] > min_fc) & (result_df["-log10(p-value)"] > alpha),
        ]
        value = ["down", "up"]
        result_df["color"] = np.select(condition, value, default="non-significant")

        # additional labeling with gene names
        hover_data = [self.index_column]

        if self.gene_names is not None:
            result_df = pd.merge(
                result_df,
                self.rawinput[[self.gene_names, self.index_column]],
                on=self.index_column,
                how="left",
            )
            hover_data.append(self.gene_names)

        # create volcano plot
        volcano_plot = px.scatter(
            result_df,
            x="log2fc",
            y="-log10(p-value)",
            color="color",
            hover_data=hover_data,
        )

        if labels:
            volcano_plot = self._volcano_add_labels(
                result_df=result_df, figure_object=volcano_plot
            )

        #  save plotting data in figure object
        volcano_plot = plotly_object(volcano_plot)
        volcano_plot = self._update_figure_attributes(
            figure_object=volcano_plot, plotting_data=result_df, method=method
        )

        if draw_line:
            volcano_plot.add_hline(
                y=alpha, line_width=1, line_dash="dash", line_color="#8c8c8c"
            )
            volcano_plot.add_vline(
                x=min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
            )
            volcano_plot.add_vline(
                x=-min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
            )

        # update coloring
        color_dict = {"non-significant": "#404040", "up": "#B65EAF", "down": "#009599"}
        volcano_plot = self._update_colors_plotly(volcano_plot, color_dict=color_dict)
        volcano_plot.update_layout(showlegend=False)
        volcano_plot.update_layout(width=600, height=700)
        return volcano_plot

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
        return fig
