from alphastats.plots.PlotUtils import PlotUtils, plotly_object
from alphastats.utils import ignore_warning, check_for_missing_values

import numpy as np
import pandas as pd
import plotly.express as px

from functools import lru_cache


class VolcanoPlot(PlotUtils):
    def __init__(
        self, dataset, group1, group2, column=None, method=None, labels=None, min_fc=None, alpha=None, draw_line=None, plot=True
    ):  
        self.dataset = dataset
        self.group1 = group1
        self.group2 = group2
        self.column = column
        self.method = method
        self.labels = labels
        self.min_fc = min_fc
        self.alpha = alpha
        self.draw_line = draw_line
        self.hover_data = None
        self.res = None
        self.pvalue_column = None
        self._check_input()
       
        if plot:
            self._perform_differential_expression_analysis()
            self._annotate_result_df()
            self._add_hover_data_columns()
            self._plot()

    def _check_input(self):
        """
        check input and add metadata column if samples are given
        """
        if isinstance(self.group1, list) and isinstance(self.group2, list):
            self.column, self.group1, self.group2 = self.dataset._add_metadata_column(
                self.group1, self.group2
            )

        if self.column is None:
            raise ValueError(
                "Column containing group1 and group2 needs to be specified"
            )

    def _update(self, updated_attributes):
        """
        update attributes using dict
        """
        for key,value in updated_attributes.items():
            setattr(self,key,value)

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

        else:
            raise ValueError(
                f"{self.method} is not available."
                + "Please select from 'ttest' or 'anova' for anova with follow up tukey or 'wald' for wald-test using."
            )

    @lru_cache(maxsize=20)
    def _wald(self):

        print(
            "Calculating differential expression analysis using wald test. Fitting generalized linear model..."
        )
        self.res = self.dataset.diff_expression_analysis(
            column=self.column,
            group1=self.group1,
            group2=self.group2,
            method=self.method,
        )
        self.pvalue_column = "qval"

    @lru_cache(maxsize=20)
    def _ttest(self):

        print("Calculating t-test...")

        self.res = self.dataset.diff_expression_analysis(
            column=self.column,
            group1=self.group1,
            group2=self.group2,
            method=self.method,
        )
        self.pvalue_column = "pval"

    @lru_cache(maxsize=20)
    def _anova(self):

        print("Calculating ANOVA with follow-up tukey test...")

        result_df = self.dataset.anova(
            column=self.column, protein_ids="all", tukey=True
        )

        group1_samples = self.dataset.metadata[
            self.dataset.metadata[self.column] == self.group1
        ][self.dataset.sample].tolist()

        group2_samples = self.dataset.metadata[
            self.dataset.metadata[self.column] == self.group2
        ][self.dataset.sample].tolist()

        mat_transpose = self.dataset.mat.transpose()
        fc = self.dataset._calculate_foldchange(mat_transpose, group1_samples, group2_samples)

        #  check how column is ordered
        self.pvalue_column = self.group1 + " vs. " + self.group2 + " Tukey Test"

        if self.pvalue_column not in result_df.columns:
            self.pvalue_column = self.group2 + " vs. " + self.group1 + " Tukey Test"

        self.res = result_df.reset_index().merge(fc.reset_index(), on=self.dataset.index_column)

    def _add_hover_data_columns(self):
        # additional labeling with gene names
        self.hover_data = [self.dataset.index_column]

        if self.dataset.gene_names is not None:
            self.res = pd.merge(
                self.res,
                self.dataset.rawinput[[self.dataset.gene_names, self.dataset.index_column]],
                on=self.dataset.index_column,
                how="left",
            )
            self.hover_data.append(self.dataset.gene_names)


    def _annotate_result_df(self):
        self.res = self.res[(self.res["log2fc"] < 10) & (self.res["log2fc"] > -10)]
        self.res["-log10(p-value)"] = -np.log10(self.res[self.pvalue_column])
        
        self.alpha = -np.log10(self.alpha)
        # add color variable to plot
        
        condition = [
            (self.res["log2fc"] < -self.min_fc) & (self.res["-log10(p-value)"] > self.alpha),
            (self.res["log2fc"] > self.min_fc) & (self.res["-log10(p-value)"] > self.alpha),
        ]

        value = ["down", "up"]
        self.res["color"] = np.select(condition, value, default="non-significant")   
        

    def _add_labels_plot(self):

        if self.dataset.gene_names is not None:
            label_column = self.dataset.gene_names
        else:
            label_column = self.dataset.index_column

        self.res["label"] = np.where(
            self.res.color != "non-significant", self.res[label_column], ""
        )
        #  replace nas with empty string (can cause error when plotting with gene names)
        self.res["label"] = self.res["label"].fillna("")
        self.res = self.res[self.res["label"] != ""]

        for x, y, label_column in self.res[
            ["log2fc", "-log10(p-value)", label_column]
        ].itertuples(index=False):
            self.plot.add_annotation(
                x=x, y=y, text=label_column, showarrow=False, yshift=10
            )

    def _draw_lines_plot(self):

        self.plot.add_hline(
            y=self.alpha, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )
        self.plot.add_vline(
            x=self.min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )
        self.plot.add_vline(
            x=-self.min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )


    def _plot(self):
        self.plot = px.scatter(
            self.res,
            x="log2fc",
            y="-log10(p-value)",
            color="color",
            hover_data=self.hover_data,
        )

        if self.labels:
            self._add_labels_plot()
        if self.draw_line:
            self._draw_lines_plot()

        # update coloring
        color_dict = {"non-significant": "#404040", "up": "#B65EAF", "down": "#009599"}
        self.plot = self._update_colors_plotly(self.plot, color_dict=color_dict)
        
        self.plot.update_layout(showlegend=False)
        self.plot.update_layout(width=600, height=700)

        #  save plotting data in figure object
        self.plot = plotly_object(self.plot)
        self.plot = self._update_figure_attributes(
            figure_object=self.plot, 
            plotting_data=self.res, 
            preprocessing_info=self.dataset.preprocessing_info, 
            method=self.method
        )
   
