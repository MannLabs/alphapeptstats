import plotly
import seaborn as sns
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


class PlotUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _update_colors_plotly(fig, color_dict):
        # plotly doesnt allow to assign color to certain group
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

    def _update_figure_attributes(
        self, figure_object, plotting_data, preprocessing_info, method=None
    ):
        setattr(figure_object, "plotting_data", plotting_data)
        setattr(figure_object, "preprocessing", preprocessing_info)
        setattr(figure_object, "method", method)
        return figure_object


class plotly_object(plotly.graph_objs._figure.Figure):
    plotting_data = None
    preprocessing = None
    method = None


class seaborn_object(sns.matrix.ClusterGrid):
    plotting_data = None
    preprocessing = None
    method = None
