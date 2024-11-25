from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go

from alphastats.dataset.keys import Cols
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.dataset.statistics import Statistics
from alphastats.dataset.utils import ignore_warning
from alphastats.multicova import multicova
from alphastats.plots.plot_utils import PlotlyObject, PlotUtils
from alphastats.statistics.differential_expression_analysis import (
    DifferentialExpressionAnalysis,
)
from alphastats.statistics.statistic_utils import (
    add_metadata_column,
    calculate_foldchange,
)

# TODO this is repeated and needs to go elsewhere!
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


class VolcanoPlot(PlotUtils):
    def __init__(
        self,
        *,
        mat: pd.DataFrame,
        rawinput: pd.DataFrame,
        metadata: pd.DataFrame,
        preprocessing_info: Dict,
        feature_to_repr_map: Dict,
        group1: Union[List[str], str],
        group2: Union[List[str], str],
        column: str = None,
        method: str = None,
        labels=None,
        min_fc=None,
        alpha=None,
        draw_line=None,
        perm=100,
        fdr=0.05,
        color_list=None,
    ):
        if color_list is None:
            color_list = []
        self.mat: pd.DataFrame = mat
        self.rawinput = rawinput
        self.metadata: pd.DataFrame = metadata
        self.preprocessing_info: Dict = preprocessing_info
        self.feature_to_repr_map = feature_to_repr_map

        self.method = method
        self.labels = labels
        self.min_fc = min_fc
        self.fdr = fdr
        self.alpha = alpha
        self.draw_line = draw_line
        self.hover_data = None

        self.perm = perm
        self.color_list = color_list

        if isinstance(group1, list) and isinstance(group2, list):
            self.metadata, self.column = add_metadata_column(metadata, group1, group2)
            self.group1, self.group2 = "group1", "group2"
        else:
            self.metadata, self.column = metadata, column
            self.group1, self.group2 = group1, group2

        if self.column is None:
            raise ValueError(
                "Column containing group1 and group2 needs to be specified"
            )

        self._statistics = Statistics(
            mat=self.mat,
            metadata=self.metadata,
            preprocessing_info=self.preprocessing_info,
        )

        self.res, self.pvalue_column, self.tlim_ttest = (
            self._perform_differential_expression_analysis()
        )
        self._annotate_result_df()
        self._add_hover_data_columns()
        self._plot()

    @ignore_warning(UserWarning)
    @ignore_warning(RuntimeWarning)
    def _perform_differential_expression_analysis(
        self,
    ) -> Tuple[pd.DataFrame, str, float]:
        """Wrapper for differential expression analysis."""

        tlim_ttest = None

        # Note: all the called methods were decorated with @lru_cache(maxsize=20), reimplement if there's performance issues
        if self.method in ["wald", "ttest", "welch-ttest", "paired-ttest"]:
            pvalue_column = "qval" if self.method == "wald" else "pval"

            res = self._statistics.diff_expression_analysis(
                column=self.column,
                group1=self.group1,
                group2=self.group2,
                method=self.method,
            )

        elif self.method == "anova":
            res, pvalue_column = self._anova()

        elif self.method == "sam":
            # TODO this is a bit of a hack, but currently diff_expression_analysis() returns only the df, not the tlim_ttest
            #  To remedy, make it return (df, {}), the latter being a dictionary containing optional additional data.
            res, tlim_ttest = DifferentialExpressionAnalysis(
                mat=self.mat,
                metadata=self.metadata,
                preprocessing_info=self.preprocessing_info,
                group1=self.group1,
                group2=self.group2,
                column=self.column,
                method="sam",
            ).sam()

            pvalue_column = "pval"

        else:
            raise ValueError(
                f"{self.method} is not available. "
                + "Please select from 'ttest', 'sam', 'paired-ttest' or 'anova' for anova with follow up tukey or 'wald' for wald-test."
            )

        return res, pvalue_column, tlim_ttest

    def _sam_calculate_fdr_line(self):
        fdr_line = multicova.get_fdr_line(
            t_limit=self.tlim_ttest,
            # TODO: Fix that this is hardcoded (see issues 270 and 273)
            s0=0.05,
            n_x=len(
                list(
                    self.metadata[self.metadata[self.column] == self.group1][
                        Cols.SAMPLE
                    ]
                )
            ),
            n_y=len(
                list(
                    self.metadata[self.metadata[self.column] == self.group2][
                        Cols.SAMPLE
                    ]
                )
            ),
            fc_s=np.arange(
                0,
                np.max(np.abs(self.res.log2fc)),
                np.max(np.abs(self.res.log2fc)) / 200,
            ),
            s_s=np.arange(0.005, 6, 0.0025),
            plot=False,
        )
        return fdr_line

    def _anova(self) -> Tuple[pd.DataFrame, str]:
        print("Calculating ANOVA with follow-up tukey test...")

        result_df = self._statistics.anova(
            column=self.column, protein_ids="all", tukey=True
        )

        group1_samples = self.metadata[self.metadata[self.column] == self.group1][
            Cols.SAMPLE
        ].tolist()

        group2_samples = self.metadata[self.metadata[self.column] == self.group2][
            Cols.SAMPLE
        ].tolist()

        mat_transpose = self.mat.transpose()

        fc = calculate_foldchange(
            mat_transpose,
            group1_samples,
            group2_samples,
            self.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED],
        )
        fc_df = pd.DataFrame({"log2fc": fc, Cols.INDEX: mat_transpose.index})

        # check how column is ordered
        pvalue_column = self.group1 + " vs. " + self.group2 + " Tukey Test"

        if pvalue_column not in result_df.columns:
            pvalue_column = self.group2 + " vs. " + self.group1 + " Tukey Test"

        res = result_df.reset_index().merge(fc_df.reset_index(), on=Cols.INDEX)

        return res, pvalue_column

    def _add_hover_data_columns(self):
        # additional labeling with gene names
        self.hover_data = [
            Cols.INDEX
        ]  # TODO this now shows the internal column name as description

        if Cols.GENE_NAMES in self.rawinput.columns:
            self.res = pd.merge(
                self.res,
                self.rawinput[[Cols.GENE_NAMES, Cols.INDEX]],
                on=Cols.INDEX,
                how="left",
            )
            self.hover_data.append(Cols.GENE_NAMES)

    def _annotate_result_df(self):
        """
        convert pvalue to log10
        add color labels for up and down regulates
        """
        self.res = self.res[(self.res["log2fc"] < 20) & (self.res["log2fc"] > -20)]
        # TODO: this is a bit hacky, but is necessary due to the masked p-values after automatic filtering. Look for a better solution where the p-values are calculated
        self.res["-log10(p-value)"] = [
            -np.log10(el) for el in self.res[self.pvalue_column]
        ]

        self.alpha = -np.log10(self.alpha)
        # add color variable to plot

        if self.method != "sam":
            condition = [
                (self.res["log2fc"] < -self.min_fc)
                & (self.res["-log10(p-value)"] > self.alpha),
                (self.res["log2fc"] > self.min_fc)
                & (self.res["-log10(p-value)"] > self.alpha),
            ]

        else:
            condition = [
                (self.res["log2fc"] < 0) & (self.res["FDR"] == "sig"),
                (self.res["log2fc"] > 0) & (self.res["FDR"] == "sig"),
            ]

        value = ["down", "up"]

        self.res["color"] = np.select(condition, value, default="non_sig")

        if len(self.color_list) > 0:
            self.res["color"] = np.where(
                self.res[Cols.INDEX].isin(self.color_list),
                "color",
                "no_color",
            )

    def _add_labels_plot(self):
        """
        add gene names as hover data if they are given
        """

        self.res["label"] = np.where(
            self.res.color != "non_sig",
            [self.feature_to_repr_map[feature] for feature in self.res[Cols.INDEX]],
            "",
        )
        # replace nas with empty string (can cause error when plotting with gene names)
        self.res["label"] = self.res["label"].fillna("")
        self.res["label"] = [
            ";".join([i for i in j.split(";") if i]) for j in self.res["label"].tolist()
        ]

        if self.labels:
            res = self.res[self.res["label"] != ""]
            for x, y, label_column in res[
                ["log2fc", "-log10(p-value)", "label"]
            ].itertuples(index=False):
                self.plot.add_annotation(
                    x=x, y=y, text=label_column, showarrow=False, yshift=10
                )

    def _draw_lines_plot(self):
        """
        draw lines using fold change and p-value cut-off, fast method
        """

        self.plot.add_hline(
            y=self.alpha, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )
        self.plot.add_vline(
            x=self.min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )
        self.plot.add_vline(
            x=-self.min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )

    def _draw_fdr_line(self):
        """
        Draw fdr line if SAM was applied
        """
        fdr_line = self._sam_calculate_fdr_line()

        for mask in [(fdr_line.fc_s > 0), (fdr_line.fc_s < 0)]:
            self.plot.add_trace(
                go.Scatter(
                    x=fdr_line[mask].fc_s,
                    y=-np.log10(fdr_line[mask].pvals),
                    line_color="black",
                    line_shape="spline",
                    showlegend=False,
                )
            )

    def _color_data_points(self):
        # update coloring
        if len(self.color_list) == 0:
            color_dict = {"non_sig": "#404040", "up": "#B65EAF", "down": "#009599"}

        else:
            color_dict = {"no_color": "#404040", "color": "#B65EAF"}

        self.plot = self._update_colors_plotly(self.plot, color_dict=color_dict)

    def _plot(self):
        self.plot = px.scatter(
            self.res,
            x="log2fc",
            y="-log10(p-value)",
            color="color",
            hover_data=self.hover_data,
            template="simple_white+alphastats_colors",
        )

        # update coloring
        self._color_data_points()

        self._add_labels_plot()

        if self.draw_line:
            if self.method == "sam":
                self._draw_fdr_line()
            else:
                self._draw_lines_plot()

        self.plot.update_layout(showlegend=False)
        self.plot.update_layout(width=600, height=700)

        # save plotting data in figure object
        self.plot = PlotlyObject(self.plot)
        self._update_figure_attributes(
            self.plot,
            plotting_data=self.res,
            preprocessing_info=self.preprocessing_info,
            method=self.method,
        )
