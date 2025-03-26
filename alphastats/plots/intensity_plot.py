import logging
from typing import Dict

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scipy

from alphastats.dataset.keys import Cols
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.plots.plot_utils import PlotlyObject, PlotUtils

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


class IntensityPlot(PlotUtils):
    def __init__(
        self,
        *,
        mat: pd.DataFrame,
        metadata: pd.DataFrame,
        intensity_column: str,
        preprocessing_info: Dict,
        protein_id,
        feature_to_repr_map: Dict,
        group,
        subgroups=None,
        method,
        add_significance,
        log_scale,
    ) -> None:
        self.mat = mat
        self.metadata = metadata
        self.intensity_column = intensity_column
        self.preprocessing_info = preprocessing_info

        self.protein_id = [protein_id] if isinstance(protein_id, str) else protein_id
        self.feature_to_repr_map = feature_to_repr_map
        self.group = group
        self.subgroups = subgroups
        self.method = method
        self.add_significance = add_significance
        self.log_scale = log_scale
        self.y_axis_label = self.intensity_column.replace("[sample]", "").strip()
        if self.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED]:
            self.y_axis_label = "log2(" + self.y_axis_label + ")"

        self.prepared_df = None
        self._prepare_data()
        self._plot()

    @staticmethod
    def _add_significance(plot):
        # add sginficance pvalue, and stars to pairwise intensity plot
        plot_dict = plot.to_plotly_json()
        data = plot_dict.get("data")

        if len(data) != 2:
            logging.warning(
                "Significance can only be estimated when there are two groups plotted."
            )
            return plot

        group1, group2 = data[0]["name"], data[1]["name"]
        y_array1, y_array2 = data[0]["y"], data[1]["y"]
        # do ttest
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

        return plot

    def _prepare_data(self):
        df = (
            self.mat[self.protein_id].melt(
                ignore_index=False,
                value_name=self.y_axis_label,
                var_name=Cols.INDEX,
            )
        ).dropna()
        df = df.reset_index().rename(columns={"index": Cols.SAMPLE})
        df = df.merge(self.metadata, how="inner", on=[Cols.SAMPLE])

        if self.subgroups is not None:
            df = df[df[self.group].isin(self.subgroups)]

        self.prepared_df = df

    def _plot(self):
        if self.method == "violin":
            fig = px.violin(
                self.prepared_df,
                y=self.y_axis_label,
                x=self.group,
                facet_col=Cols.INDEX,
                color=self.group,
                template="simple_white+alphastats_colors",
            )

        elif self.method == "box":
            fig = px.box(
                self.prepared_df,
                y=self.y_axis_label,
                x=self.group,
                facet_col=Cols.INDEX,
                color=self.group,
                template="simple_white+alphastats_colors",
            )

        elif self.method == "scatter":
            fig = px.scatter(
                self.prepared_df,
                y=self.y_axis_label,
                x=self.group,
                facet_col=Cols.INDEX,
                color=self.group,
                template="simple_white+alphastats_colors",
            )

        elif self.method == "all":
            fig = px.violin(
                self.prepared_df,
                y=self.y_axis_label,
                x=self.group,
                facet_col=Cols.INDEX,
                color=self.group,
                box=True,
                points="all",
                template="simple_white+alphastats_colors",
            )

        else:
            raise ValueError(
                f"{self.method} is not available."
                + "Please select from 'violin' for Violinplot, 'box' for Boxplot and 'scatter' for Scatterplot."
            )

        fig.for_each_annotation(
            lambda a: a.update(text=self.feature_to_repr_map[a.text.split("=")[-1]])
        )

        if self.log_scale:
            fig.update_layout(yaxis=dict(type="log"))

        if self.add_significance:
            fig = self._add_significance(fig)

        fig.update_layout(
            width=100
            + len(self.protein_id) * self.prepared_df[self.group].nunique() * 50,
            height=500,
        )

        fig = PlotlyObject(fig)
        self._update_figure_attributes(
            fig,
            plotting_data=self.prepared_df,
            preprocessing_info=self.preprocessing_info,
            method=self.method,
        )

        self.plot = fig
