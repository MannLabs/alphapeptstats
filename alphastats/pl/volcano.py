"""Module for creating volcano plots of differential expression analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.express as px

from alphastats.tl.differential_expression_analysis import (
    DeaColumns,
    DifferentialExpressionAnalysis,
)

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objs as go


def plot_volcano(  # noqa: PLR0913
    statistics_results: pd.DataFrame,
    feature_to_repr_map: dict,
    group1: str,
    group2: str,
    qvalue_cutoff: float = 0.05,
    log2fc_cutoff: float | None = 1,
    renderer: Literal["webgl", "svg"] = "webgl",
    *,
    draw_lines: bool = True,
    label_significant: bool = True,
    flip_xaxis: bool = False,
) -> tuple(go.Figure, pd.DataFrame):
    """Create a volcano plot of the differential expression analysis results by first formatting the data and then creating the plot.

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
    qvalue_cutoff : float, default=0.05
        The significance cutoff for the q-values.
    log2fc_cutoff : Union[float, None], default=1
        The fold cutoff for the log2 fold changes.
    renderer : Literal['webgl', 'svg']
        The renderer to use for the plot. webgl or svg.
    draw_lines : bool, default=True
        Whether to draw the significance and fold change cutoff lines.
    label_significant : bool, default=True
        Whether to label significant points.
    flip_xaxis : bool, default=False
        Whether to flip the x-axis.

    Returns
    -------
    go.Figure
        The volcano plot.
    pd.DataFrame
        The prepared dataframe.

    """
    df_plot = prepare_result_df(
        statistics_results_df=statistics_results,
        feature_to_repr_map=feature_to_repr_map,
        group1=group1,
        group2=group2,
        qvalue_cutoff=qvalue_cutoff,
        log2fc_cutoff=log2fc_cutoff,
        flip_xaxis=flip_xaxis,
    )

    fig = _plot_volcano(
        df_plot=df_plot,
        group1=group1,
        group2=group2,
        qvalue_cutoff=qvalue_cutoff,
        log2fc_cutoff=log2fc_cutoff,
        draw_lines=draw_lines,
        label_significant=label_significant,
        flip_xaxis=flip_xaxis,
        renderer=renderer,
    )
    return fig, df_plot


def _plot_volcano(  # noqa: PLR0913
    df_plot: pd.DataFrame,
    group1: str,
    group2: str,
    qvalue_cutoff: float,
    log2fc_cutoff: float | None,
    renderer: Literal["webgl", "svg"],
    *,
    draw_lines: bool,
    label_significant: bool,
    flip_xaxis: bool,
    **layout_options,
) -> go.Figure:
    """Create the volcano plot from formatted data.

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
    renderer : Literal['webgl', 'svg']
        The renderer to use for the plot. webgl or svg.
    draw_lines : bool
        Whether to draw the significance and fold change cutoff lines.
    label_significant : bool
        Whether to label significant points.
    flip_xaxis : bool
        Whether to flip the x-axis.
    layout_options : dict
        Additional layout options for the plot. Passed directly to update_layout.

    Returns
    -------
    go.Figure
        The volcano plot.

    """
    log2name = get_foldchange_column_name(group1, group2, flip_xaxis=flip_xaxis)

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
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
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

    if draw_lines:
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
            max_chars = 10
            if significant == "up":
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=label if len(label) < max_chars else label[:max_chars] + "...",
                    showarrow=False,
                    xanchor="left",
                    xshift=5,
                )
            if significant == "down":
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=label if len(label) < max_chars else label[:max_chars] + "...",
                    showarrow=False,
                    xanchor="right",
                    xshift=-5,
                )

    fig.update_layout(**layout_options)
    return fig


def prepare_result_df(  # noqa: PLR0913
    statistics_results_df: pd.DataFrame,
    feature_to_repr_map: dict,
    group1: str,
    group2: str,
    qvalue_cutoff: float,
    log2fc_cutoff: float | None,
    *,
    flip_xaxis: bool = False,
) -> tuple(pd.DataFrame, str):
    """Prepare the dataframe for plotting or display/download.

    Parameters
    ----------
    statistics_results_df : pd.DataFrame
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
    result_df = statistics_results_df.copy()

    # get significant q-values
    result_df = result_df.join(
        DifferentialExpressionAnalysis.get_significance_qvalue(
            result_df, qvalue_cutoff=qvalue_cutoff
        ),
        how="left",
        validate="one_to_one",
    )

    log2name = get_foldchange_column_name(group1, group2, flip_xaxis=flip_xaxis)
    if flip_xaxis:
        result_df[DeaColumns.LOG2FC] = -result_df[DeaColumns.LOG2FC]
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
    result_df["-log10(q-value)"] = [
        -np.log10(qvalue) for qvalue in result_df[DeaColumns.QVALUE]
    ]

    # map feature names to representations
    result_df["label"] = result_df.index.map(feature_to_repr_map)

    return result_df.reset_index()


def get_foldchange_column_name(
    group1: str,
    group2: str,
    *,
    flip_xaxis: bool,
) -> str:
    """Get the descriptive name of the log2 fold change column."""
    if not flip_xaxis:
        left_label = group2
        right_label = group1
    else:
        left_label = group1
        right_label = group2
    return f"log2({right_label}/{left_label})"
