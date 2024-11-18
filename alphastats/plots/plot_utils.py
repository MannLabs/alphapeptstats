from typing import Dict, Optional

import pandas as pd
import plotly
import plotly.graph_objects as go

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


class PlotlyObject(plotly.graph_objs._figure.Figure):
    plotting_data = None
    preprocessing = None
    method = None


# TODO unused?
# class seaborn_object(sns.matrix.ClusterGrid):
#     plotting_data = None
#     preprocessing = None
#     method = None


class PlotUtils:
    @staticmethod
    def _update_colors_plotly(fig, color_dict):
        # TODO revisit this comment:
        #  plotly doesnt allow to assign color to certain group
        #  update instead the figure in form of a dict
        #  color_dict with group_variable/legendgroup as key, and corresponding color as value
        # update:
        # https://plotly.com/python-api-reference/generated/generated/plotly.graph_objects.Figure.update_traces.html
        # + selector to set individual color or something like:
        # plot.for_each_trace(lambda t: t.update(marker_color=color_dict.get(t.legendgroup))
        fig_dict = fig.to_plotly_json()
        data_dict_list = fig_dict.get("data")
        for count, group in enumerate(data_dict_list):
            group_variable = group.get("legendgroup")
            group_color = color_dict.get(group_variable)
            fig_dict["data"][count]["marker"]["color"] = group_color
        # convert dict back to plotly figure
        return go.Figure(fig_dict)

    def _update_figure_attributes(
        self,
        figure_object: PlotlyObject,
        *,
        plotting_data: pd.DataFrame,
        preprocessing_info: Dict,
        method: Optional[str] = None,
    ) -> None:
        figure_object.plotting_data = plotting_data
        figure_object.preprocessing = preprocessing_info
        figure_object.method = method
