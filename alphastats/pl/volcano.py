from typing import Dict, Literal, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objs._figure import Figure

from alphastats.gui.utils.ui_helper import StateKeys
from alphastats.pl.display_utils import ResultObject
from alphastats.tl.differential_expression_analysis import (
    DeaColumns,
    DifferentialExpressionAnalysis,
)


def _plot_volcano(
    df_plot: pd.DataFrame,
    log2name: str,
    group1: str,
    group2: str,
    qvalue_cutoff: float,
    log2fc_cutoff: Union[float, None],
    drawlines: bool,
    label_significant: bool,
    flip_xaxis: bool,
    renderer: Literal["webgl", "svg"],
    **kwargs,
) -> Figure:
    """Create the volcano plot.

    Parameters
    ----------
    df_plot : pd.DataFrame
        The dataframe to plot.
    log2name : str
        The name of the log2 fold change column.
    group1 : str
        The name of the first group.
    group2 : str
        The name of the second group.
    qvalue_cutoff : float
        The significance cutoff for the q-values.
    log2fc_cutoff : Union[float, None]
        The fold cutoff for the log2 fold changes.
    drawlines : bool
        Whether to draw the significance and fold change cutoff lines.
    label_significant : bool
        Whether to label significant points.
    flip_xaxis : bool
        Whether to flip the x-axis.
    renderer : Literal['webgl', 'svg']
        The renderer to use for the plot. webgl or svg.

    Returns
    -------
    go.Figure
        The volcano plot.
    """
    # calculate x_range
    factor = 1.1 if not label_significant else 1.3
    x_range = (
        max([-20, np.round(df_plot[log2name].min() * factor, 1)]),
        min([20, np.round(df_plot[log2name].max() * factor, 1)]),
    )

    fig = px.scatter(
        df_plot,
        x=log2name,
        y="-log10(q-value)",
        hover_data=[
            column
            for column in df_plot.columns
            if column not in [log2name, "-log10(q-value)", "significant"]
        ],
        color="significant",
        color_discrete_map={"non_sig": "#404040", "up": "#B65EAF", "down": "#009599"},
        template="simple_white",
        render_mode=renderer,
    )
    fig.update_layout(
        showlegend=False,
        width=600,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Volcano plot of {group1} vs {group2}",
    )
    fig.add_annotation(
        y=1,
        x=x_range[0],
        xref="x",
        yref="paper",
        text=f"<b>{group2 if not flip_xaxis else group1}</b>",
        xanchor="left",
        showarrow=False,
    )
    fig.add_annotation(
        y=1,
        x=x_range[1],
        xref="x",
        yref="paper",
        text=f"<b>{group1 if not flip_xaxis else group2}</b>",
        xanchor="right",
        showarrow=False,
    )

    if drawlines:
        fig.add_hline(
            y=-np.log10(qvalue_cutoff),
            line_width=1,
            line_dash="dash",
            line_color="#8c8c8c",
        )
        if log2fc_cutoff is not None:
            fig.add_vline(
                x=log2fc_cutoff, line_width=1, line_dash="dash", line_color="#8c8c8c"
            )
            fig.add_vline(
                x=-log2fc_cutoff, line_width=1, line_dash="dash", line_color="#8c8c8c"
            )

    if label_significant:
        for x, y, significant, label in zip(
            df_plot[log2name],
            df_plot["-log10(q-value)"],
            df_plot["significant"],
            df_plot["label"],
        ):
            if significant == "up":
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=label if len(label) < 10 else label[:10] + "...",
                    showarrow=False,
                    xanchor="left",
                    xshift=5,
                )
            if significant == "down":
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=label if len(label) < 10 else label[:10] + "...",
                    showarrow=False,
                    xanchor="right",
                    xshift=-5,
                )

    fig.update_layout(**kwargs)
    return fig


def prepare_result_df(
    statistics_results: pd.DataFrame,
    feature_to_repr_map: dict,
    group1: str,
    group2: str,
    qvalue_cutoff: float,
    log2fc_cutoff: Union[float, None],
    flip_xaxis: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Prepare the dataframe for plotting or display/download.

    Parameters
    ----------
    statistics_results : pd.DataFrame
        The results of the differential expression analysis.
    feature_to_repr_map : dict
        A dictionary mapping feature names to their representations.
    group1 : str
        The name of the first group.
    group2 : str
        The name of the second group.
    qvalue_cutoff : float
        The significance cutoff for the q-values.
    log2fc_cutoff : float
        The fold cutoff for the log2 fold changes.
    flip_xaxis : bool, default=False
        Whether to flip the x-axis.

    Returns
    -------
    pd.DataFrame
        The prepared dataframe.
    str
        The name of the log2 fold change column.
    """
    result_df = statistics_results.copy()

    # get significant q-values
    result_df = result_df.join(
        DifferentialExpressionAnalysis.get_significance_qvalue(
            result_df, qvalue_cutoff=qvalue_cutoff
        ),
        how="left",
    )

    if not flip_xaxis:
        left_label = group2
        right_label = group1
    else:
        result_df[DeaColumns.LOG2FC] = -result_df[DeaColumns.LOG2FC]
        left_label = group1
        right_label = group2
    log2name = f"log2({right_label}/{left_label})"
    result_df = result_df.rename(columns={DeaColumns.LOG2FC: log2name})

    # get significant column
    if log2fc_cutoff is not None:
        result_df["significant"] = (
            result_df[log2name].abs() >= log2fc_cutoff
        ) & result_df[DeaColumns.SIGNIFICANTQ]
    else:
        result_df["significant"] = result_df[DeaColumns.SIGNIFICANTQ]
    result_df["significant"] = np.where(
        result_df["significant"],
        (result_df[log2name] > 0).map({True: "up", False: "down"}),
        "non_sig",
    )

    # transform q-values to -log10
    result_df["-log10(q-value)"] = -np.log10(result_df[DeaColumns.QVALUE])

    # map feature names to representations
    result_df["label"] = result_df.index.map(feature_to_repr_map)

    return result_df, log2name


class DifferentialExpressionTwoGroupsResult(ResultObject):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessing: Dict,
        method: Dict,
    ):
        super().__init__(
            dataframe, plottable=True, preprocessing=preprocessing, method=method
        )
        self.log2name = ""

    def _get_data_annotation_options(self) -> Dict:
        return {
            "qvalue_cutoff": st.number_input(
                "Q-value cutoff", 0.0, 1.0, 0.05, 0.01, format="%.2f"
            ),
            "log2fc_cutoff": st.number_input(
                "Log2FC cutoff", 0.0, 10.0, 1.0, 0.1, format="%.1f"
            ),
            "flip_xaxis": st.checkbox("Flip groups", False),
        }

    def _update_data_annotation(
        self,
        qvalue_cutoff: float,
        log2fc_cutoff: float,
        flip_xaxis: bool,
    ) -> pd.DataFrame:
        formatted_df, log2name = prepare_result_df(
            statistics_results=self.dataframe,
            feature_to_repr_map=st.session_state[StateKeys.DATASET].feature_to_repr_map,
            group1=self.method["group1"],
            group2=self.method["group2"],
            qvalue_cutoff=qvalue_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            flip_xaxis=flip_xaxis,
        )
        self.log2name = log2name
        return formatted_df

    def _get_plot_options(self) -> Dict:
        return {
            **{
                "drawlines": st.checkbox(
                    "Draw significance and fold change lines", True
                ),
                "label_significant": st.checkbox("Label significant points", True),
                "renderer": st.radio("Renderer", ["webgl", "svg"], index=0),
            },
            **self.get_standard_layout_options(),
        }

    def _update_plot(
        self,
        drawlines: bool,
        label_significant: bool,
        renderer: Literal["webgl", "svg"],
        **kwargs,
    ) -> Figure:
        return _plot_volcano(
            df_plot=self.dataframe,
            log2name=self.log2name,
            group1=self.method["group1"],
            group2=self.method["group2"],
            qvalue_cutoff=self.data_annotation_options["qvalue_cutoff"],
            log2fc_cutoff=self.data_annotation_options["log2fc_cutoff"],
            flip_xaxis=self.data_annotation_options["flip_xaxis"],
            drawlines=drawlines,
            label_significant=label_significant,
            renderer=renderer,
            **kwargs,
        )
