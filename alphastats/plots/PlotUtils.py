import plotly
import seaborn as sns
import plotly.graph_objects as go

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

    def _update_figure_attributes(self, figure_object, plotting_data, preprocessing_info, method=None):
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
