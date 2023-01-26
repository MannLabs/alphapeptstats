from alphastats.DataSet_Plot import plotly_object
from alphastats.plots.PlotUtils import PlotUtils
from alphastats.utils import ignore_warning, check_for_missing_values

import numpy as np
import plotly.express as px

class VolcanoPlot(PlotUtils):
    def __init__(self, dataset, group1, group2, column, method, min_fc, alpha, draw_line):
        self.dataset = dataset
        self.group1 = group1
        self.group2 = group2
        self.column = column
        self.method = method
        self.min_fc = min_fc
        self.alpha = alpha
        self.draw_line = draw_line
        self.res = None
        self.min_fc = None
        self.pvalue_column = None

        self._check_input()
        self._perform_differential_expression_analysis()
        self._plot()
    
    def _check_input(self):
        if isinstance(self.group1, list) and isinstance(self.group2, list):
            self.column, self.group1, self.group2 = self.dataset._add_metadata_column(self.group1, self.group2)

        if self.column is None:
            raise ValueError(
                "Column containing group1 and group2 needs to be specified"
            )
    
    @ignore_warning(UserWarning)
    @ignore_warning(RuntimeWarning)
    def _perform_differential_expression_analysis(self):
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

    def _wald(self):
        
        print(
                "Calculating differential expression analysis using wald test. Fitting generalized linear model..."
            )
        self.res = self.dataset.diff_expression_analysis(
                column=self.column, group1=self.group1, group2=self.group2, method=self.method
            )
        self.pvalue_column = "qval"
    
    def _ttest(self):
        
        print("Calculating t-test...")
        
        self.res = self.dataset.diff_expression_analysis(
                column=self.column, group1=self.group1, group2=self.group2, method=self.method
            )
        self.pvalue_column = "pval"

    def _anova(self):

        print("Calculating ANOVA with follow-up tukey test...")

        result_df = self.dataset.anova(column=self.column, protein_ids="all", tukey=True)
        
        group1_samples = self.dataset.metadata[self.dataset.metadata[self.column] == self.group1][
                self.sample
            ].tolist()

        group2_samples = self.dataset.metadata[self.dataset.metadata[self.column] == self.group2][
                self.sample
            ].tolist()

        mat_transpose = self.dataset.mat.transpose()
        fc = self._calculate_foldchange(
                mat_transpose, group1_samples, group2_samples
            )

            #  check how column is ordered
        pvalue_column = self.group1 + " vs. " + self.group2 + " Tukey Test"

        if pvalue_column not in fc.columns:
            pvalue_column = self.group2 + " vs. " + self.group1 + " Tukey Test"

        self.res = result_df.reset_index().merge(
                fc.reset_index(), on=self.index_column
            )


    def _annotate_result_df(self):
        res = self.res[(self.res["log2fc"] < 10) & (self.res["log2fc"] > -10)]
        res["-log10(p-value)"] = -np.log10(res[self.pvalue_column])
        alpha = -np.log10(alpha)
        # add color variable to plot
        
        condition = [
            (res["log2fc"] < -self.min_fc) & (res["-log10(p-value)"] > alpha),
            (res["log2fc"] > self.min_fc) & (res["-log10(p-value)"] > alpha),
        ]
        
        value = ["down", "up"]
        res["color"] = np.select(condition, value, default="non-significant")
        self.res = res

    def _add_labels_plot(self, result_df, figure_object):

        if self.dataset.gene_names is not None:
            label_column = self.dataset.gene_names
        else:
            label_column = self.dataset.index_column

        result_df["label"] = np.where(
            result_df.color != "non-significant", result_df[label_column], ""
        )
        #  replace nas with empty string (can cause error when plotting with gene names)
        result_df["label"] = result_df["label"].fillna("")
        result_df = result_df[result_df["label"] != ""]

        for x, y, label_column in result_df[
            ["log2fc", "-log10(p-value)", label_column]
        ].itertuples(index=False):
            figure_object.add_annotation(
                x=x, y=y, text=label_column, showarrow=False, yshift=10
            )
        return figure_object

    def _draw_lines_plot(self, volcano_plot):
        
        volcano_plot.add_hline(
            y=self.alpha, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )
        volcano_plot.add_vline(
            x=self.min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )
        volcano_plot.add_vline(
            x=-self.min_fc, line_width=1, line_dash="dash", line_color="#8c8c8c"
        )

        return volcano_plot

    def _plot(self):
        volcano_plot = px.scatter(
            self.res,
            x="log2fc",
            y="-log10(p-value)",
            color="color",
            hover_data=self.hover_data,
        )

        if self.labels:
            volcano_plot = self._add_labels_plot(
                result_df=self.res, figure_object=volcano_plot
            )
        if self.draw_line:
            volcano_plot = self._draw_lines_plot(volcano_plot=volcano_plot)

        # update coloring
        color_dict = {"non-significant": "#404040", "up": "#B65EAF", "down": "#009599"}
        volcano_plot = self._update_colors_plotly(volcano_plot, color_dict=color_dict)
        volcano_plot.update_layout(showlegend=False)
        volcano_plot.update_layout(width=600, height=700)

        #  save plotting data in figure object
        volcano_plot = plotly_object(volcano_plot)
        volcano_plot = self._update_figure_attributes(
            figure_object=volcano_plot, plotting_data=self.re, method=self.method
        )
        self.plot = volcano_plot
        
       
