import plotly.express as px
import plotly
import scipy

import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import plotly.figure_factory
from typing import Dict

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
    def __init__(
        self,
        mat: pd.DataFrame,
        rawmat: pd.DataFrame,
        metadata: pd.DataFrame,
        sample: str,
        preprocessing_info: Dict,
    ):
        self.mat: pd.DataFrame = mat
        self.rawmat: pd.DataFrame = rawmat
        self.metadata: pd.DataFrame = metadata
        self.sample: str = sample
        self.preprocessing_info: Dict = preprocessing_info

    def _update_figure_attributes(self, figure_object, plotting_data, method=None):
        setattr(figure_object, "plotting_data", plotting_data)
        setattr(figure_object, "preprocessing", self.preprocessing_info)
        setattr(figure_object, "method", method)
        return figure_object

    def plot_correlation_matrix(self, method: str = "pearson"):  # TODO unused
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

    def plot_sampledistribution(
        self,
        method: str = "violin",
        color: str = None,
        log_scale: bool = False,
        use_raw: bool = False,
    ):
        """Plot Intensity Distribution for each sample. Either Violin or Boxplot

        Args:
            method (str, optional): Violinplot = "violin", Boxplot = "box". Defaults to "violin".
            color (str, optional): A metadata column used to color the boxes. Defaults to None.
            log_scale (bool, optional): yaxis in logarithmic scale. Defaults to False.
            use_raw (bool, optional): use raw data instead of processed data. Defaults to False.

        Returns:
             plotly.graph_objects._figure.Figure: Plotly Sample Distribution Plot
        """

        # create long df
        matrix = self.mat if not use_raw else self.rawmat
        df = matrix.unstack().reset_index()
        df.rename(columns={"level_1": self.sample, 0: "Intensity"}, inplace=True)

        if color is not None:
            df = df.merge(self.metadata, how="inner", on=[self.sample])

        if method == "violin":
            fig = px.violin(
                df,
                x=self.sample,
                y="Intensity",
                color=color,
                template="simple_white+alphastats_colors",
            )

        elif method == "box":
            fig = px.box(
                df,
                x=self.sample,
                y="Intensity",
                color=color,
                template="simple_white+alphastats_colors",
            )

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

    @check_for_missing_values
    def plot_dendrogram(
        self, linkagefun=lambda x: scipy.cluster.hierarchy.linkage(x, "complete")
    ):
        """Plot Hierarchical Clustering Dendrogram. This is a wrapper around:
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

    def plot_imputed_values(self):  # not used
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
            template="simple_white+alphastats_colors",
        )

        pass
