from functools import lru_cache
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go

from alphastats.DataSet_Preprocess import PreprocessingStateKeys
from alphastats.DataSet_Statistics import Statistics
from alphastats.plots.PlotUtils import PlotUtils, plotly_object
from alphastats.statistics.DifferentialExpressionAnalysis import (
    DifferentialExpressionAnalysis,
)
from alphastats.utils import ignore_warning

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
        sample: str,
        index_column: str,
        gene_names: str,
        preprocessing_info: Dict,
        group1: Union[List[str], str],
        group2: Union[List[str], str],
        column: str = None,
        method: str = None,
        labels=None,
        min_fc=None,
        alpha=None,
        draw_line=None,
        plot=True,
        perm=100,
        fdr=0.05,
        color_list=None,
    ):
        if color_list is None:
            color_list = []
        self.mat: pd.DataFrame = mat
        self.rawinput = rawinput
        self.metadata: pd.DataFrame = metadata
        self.sample: str = sample
        self.index_column: str = index_column
        self.gene_names: str = gene_names
        self.preprocessing_info: Dict = preprocessing_info

        self.method = method
        self.labels = labels
        self.min_fc = min_fc
        self.fdr = fdr
        self.alpha = alpha
        self.draw_line = draw_line
        self.hover_data = None
        self.res = None
        self.pvalue_column = None
        self.perm = perm
        self.color_list = color_list

        if isinstance(group1, list) and isinstance(group2, list):
            self.metadata, self.column = self._add_metadata_column(
                metadata, group1, group2
            )
            self.group1, self.group2 = "group1", "group2"
        else:
            self.metadata, self.column = metadata, column
            self.group1, self.group2 = group1, group2

        self._check_input()

        self._statistics = Statistics(
            mat=self.mat,
            metadata=self.metadata,
            sample=self.sample,
            index_column=self.index_column,
            preprocessing_info=self.preprocessing_info,
        )

        if plot:
            self._perform_differential_expression_analysis()
            self._annotate_result_df()
            self._add_hover_data_columns()
            self._plot()

    # TODO this used to change the actual metadata .. was this intended?
    def _add_metadata_column(
        self, metadata: pd.DataFrame, group1_list: list, group2_list: list
    ):
        # create new column in metadata with defined groups

        sample_names = metadata[self.sample].to_list()
        misc_samples = list(set(group1_list + group2_list) - set(sample_names))
        if len(misc_samples) > 0:
            raise ValueError(
                f"Sample names: {misc_samples} are not described in Metadata."
            )

        column = "_comparison_column"
        conditons = [
            metadata[self.sample].isin(group1_list),
            metadata[self.sample].isin(group2_list),
        ]
        choices = ["group1", "group2"]
        metadata[column] = np.select(conditons, choices, default=np.nan)

        return metadata, column

    def _check_input(self):
        """Check if self.column is set correctly."""
        if self.column is None:
            raise ValueError(
                "Column containing group1 and group2 needs to be specified"
            )

    # TODO revisit this
    def _update(self, updated_attributes):
        """
        update attributes using dict
        """
        for key, value in updated_attributes.items():
            setattr(self, key, value)

    @ignore_warning(UserWarning)
    @ignore_warning(RuntimeWarning)
    def _perform_differential_expression_analysis(self):
        """
        wrapper for diff analysis
        """
        if self.method == "wald":
            self._wald()

        elif self.method == "ttest":
            self._ttest()

        elif self.method == "anova":
            self._anova()

        elif self.method == "welch-ttest":
            self._welch_ttest()

        elif self.method == "paired-ttest":
            self._pairedttest()

        elif self.method == "sam":
            self._sam()

        else:
            raise ValueError(
                f"{self.method} is not available."
                + "Please select from 'ttest', 'sam', 'paired-ttest' or 'anova' for anova with follow up tukey or 'wald' for wald-test."
            )

    @lru_cache(maxsize=20)
    def _sam_calculate_fdr_line(self):
        from alphastats.multicova import multicova

        self.fdr_line = multicova.get_fdr_line(
            t_limit=self.tlim_ttest,
            s0=0.05,
            n_x=len(
                list(
                    self.metadata[self.metadata[self.column] == self.group1][
                        self.sample
                    ]
                )
            ),
            n_y=len(
                list(
                    self.metadata[self.metadata[self.column] == self.group2][
                        self.sample
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

    @lru_cache(maxsize=20)
    def _sam(self):  # TODO duplicated? DUP1
        from alphastats.multicova import multicova

        print("Calculating t-test and permutation based FDR (SAM)... ")

        transposed = self.mat.transpose()

        if not self.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED]:
            # needs to be lpog2 transformed for fold change calculations
            transposed = transposed.transform(lambda x: np.log2(x))

        transposed[self.index_column] = transposed.index
        transposed = transposed.reset_index(drop=True)

        res_ttest, tlim_ttest = multicova.perform_ttest_analysis(
            transposed,
            c1=list(
                self.metadata[self.metadata[self.column] == self.group1][self.sample]
            ),
            c2=list(
                self.metadata[self.metadata[self.column] == self.group2][self.sample]
            ),
            s0=0.05,
            n_perm=self.perm,
            fdr=self.fdr,
            id_col=self.index_column,
            parallelize=True,
        )

        fdr_column = "FDR" + str(int(self.fdr * 100)) + "%"
        self.res = res_ttest[
            [
                self.index_column,
                "fc",
                "tval",
                "pval",
                "tval_s0",
                "pval_s0",
                "qval",
            ]
        ]
        self.res["log2fc"] = res_ttest["fc"]
        self.res["FDR"] = res_ttest[fdr_column]
        self.tlim_ttest = tlim_ttest
        self.pvalue_column = "pval"

    @lru_cache(maxsize=20)
    def _wald(self):
        print(
            "Calculating differential expression analysis using wald test. Fitting generalized linear model..."
        )
        self.res = self._statistics.diff_expression_analysis(
            column=self.column,
            group1=self.group1,
            group2=self.group2,
            method=self.method,
        )
        self.pvalue_column = "qval"

    @lru_cache(maxsize=20)
    def _welch_ttest(self):
        print("Calculating Welchs t-test...")

        self.res = self._statistics.diff_expression_analysis(
            column=self.column,
            group1=self.group1,
            group2=self.group2,
            method=self.method,
        )
        self.pvalue_column = "pval"

    @lru_cache(maxsize=20)
    def _ttest(self):
        print("Calculating Students t-test...")

        self.res = self._statistics.diff_expression_analysis(
            column=self.column,
            group1=self.group1,
            group2=self.group2,
            method=self.method,
        )
        self.pvalue_column = "pval"

    @lru_cache(maxsize=20)
    def _pairedttest(self):
        print("Calculating paired t-test...")

        self.res = self._statistics.diff_expression_analysis(
            column=self.column,
            group1=self.group1,
            group2=self.group2,
            method=self.method,
        )
        self.pvalue_column = "pval"

    @lru_cache(maxsize=20)
    def _anova(self):
        print("Calculating ANOVA with follow-up tukey test...")

        result_df = self._statistics.anova(
            column=self.column, protein_ids="all", tukey=True
        )

        group1_samples = self.metadata[self.metadata[self.column] == self.group1][
            self.sample
        ].tolist()

        group2_samples = self.metadata[self.metadata[self.column] == self.group2][
            self.sample
        ].tolist()

        mat_transpose = self.mat.transpose()

        fc = DifferentialExpressionAnalysis.calculate_foldchange(
            mat_transpose,
            group1_samples,
            group2_samples,
            self.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED],
        )
        fc_df = pd.DataFrame({"log2fc": fc, self.index_column: mat_transpose.index})

        # check how column is ordered
        self.pvalue_column = self.group1 + " vs. " + self.group2 + " Tukey Test"

        if self.pvalue_column not in result_df.columns:
            self.pvalue_column = self.group2 + " vs. " + self.group1 + " Tukey Test"

        self.res = result_df.reset_index().merge(
            fc_df.reset_index(), on=self.index_column
        )

    def _add_hover_data_columns(self):
        # additional labeling with gene names
        self.hover_data = [self.index_column]

        if self.gene_names is not None:
            self.res = pd.merge(
                self.res,
                self.rawinput[[self.gene_names, self.index_column]],
                on=self.index_column,
                how="left",
            )
            self.hover_data.append(self.gene_names)

    def _annotate_result_df(self):
        """
        convert pvalue to log10
        add color labels for up and down regulates
        """
        self.res = self.res[(self.res["log2fc"] < 20) & (self.res["log2fc"] > -20)]
        self.res["-log10(p-value)"] = -np.log10(self.res[self.pvalue_column])

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
                self.res[self.index_column].isin(self.color_list),
                "color",
                "no_color",
            )

    def get_colored_labels_df(self):
        """
        get dataframe of upregulated and downregulated genes in form of {gene_name: color},
        """
        if "label" not in self.res.columns:
            if self.gene_names is not None:
                label_column = self.gene_names
            else:
                label_column = self.index_column

            self.res["label"] = np.where(
                self.res.color != "non_sig", self.res[label_column], ""
            )
            # replace nas with empty string (can cause error when plotting with gene names)
            self.res["label"] = self.res["label"].fillna("")
            self.res = self.res[self.res["label"] != ""]
        if "color" not in self.res.columns:
            self._annotate_result_df()

        return self.res

    def _add_labels_plot(self):
        """
        add gene names as hover data if they are given
        """

        if self.gene_names is not None:
            label_column = self.gene_names
        else:
            label_column = self.index_column

        self.res["label"] = np.where(
            self.res.color != "non_sig", self.res[label_column], ""
        )
        # replace nas with empty string (can cause error when plotting with gene names)
        self.res["label"] = self.res["label"].fillna("")
        self.res["label"] = [
            ";".join([i for i in j.split(";") if i]) for j in self.res["label"].tolist()
        ]
        self.res = self.res[self.res["label"] != ""]

        for x, y, label_column in self.res[
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
        self._sam_calculate_fdr_line()

        self.plot.add_trace(
            go.Scatter(
                x=self.fdr_line[self.fdr_line.fc_s > 0].fc_s,
                y=-np.log10(self.fdr_line[self.fdr_line.fc_s > 0].pvals),
                line_color="black",
                line_shape="spline",
                showlegend=False,
            )
        )
        self.plot.add_trace(
            go.Scatter(
                x=self.fdr_line[self.fdr_line.fc_s < 0].fc_s,
                y=-np.log10(self.fdr_line[self.fdr_line.fc_s < 0].pvals),
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

        if self.labels:
            self._add_labels_plot()

        if self.draw_line:
            if self.method == "sam":
                self._draw_fdr_line()
            else:
                self._draw_lines_plot()

        self.plot.update_layout(showlegend=False)
        self.plot.update_layout(width=600, height=700)

        # save plotting data in figure object
        self.plot = plotly_object(self.plot)
        self._update_figure_attributes(
            self.plot,
            plotting_data=self.res,
            preprocessing_info=self.preprocessing_info,
            method=self.method,
        )
