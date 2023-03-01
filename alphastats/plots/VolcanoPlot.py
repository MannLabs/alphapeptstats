from alphastats.plots.PlotUtils import PlotUtils, plotly_object
from alphastats.utils import ignore_warning, check_for_missing_values

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from functools import lru_cache


class VolcanoPlot(PlotUtils):
    def __init__(
        self, dataset, group1, group2, 
        column=None, method=None, 
        labels=None, min_fc=None, 
        alpha=None, draw_line=None, 
        plot=True, perm=100, fdr=0.05
    ):  
        self.dataset = dataset
        self.group1 = group1
        self.group2 = group2
        self.column = column
        self.method = method
        self.labels = labels
        self.min_fc = min_fc
        self.fdr = fdr
        self.alpha = alpha
        self.draw_line = draw_line
        self.hover_data = None
        self.res = None
        self.pvalue_column = None
        self.perm=perm
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
        
        elif self.method == "sam":
            self._sam()
        
        # elif self.method == "Multi Covariates":
        #    raise NotImplementedError

        else:
            raise ValueError(
                f"{self.method} is not available."
                + "Please select from 'ttest', 'sam' or 'anova' for anova with follow up tukey or 'wald' for wald-test."
            )

    @lru_cache(maxsize=20)
    def _sam_calculate_fdr_line(self):
        from alphastats.multicova import multicova

        self.fdr_line= multicova.get_fdr_line(
                t_limit=self.tlim_ttest,
                s0=0.05,
                n_x=len(list(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1][self.dataset.sample])),
                n_y=len(list(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2][self.dataset.sample])),
                fc_s = np.arange(0,np.max(np.abs(self.res.log2fc)),np.max(np.abs(self.res.log2fc))/200),
                s_s = np.arange(0.005, 6, 0.0025),
                plot=False
            )
    
    @lru_cache(maxsize=20)
    def _sam(self):
        from alphastats.multicova import multicova

        print(
            "Calculating t-test and permuation based FDR (SAM)... "
        )

        transposed = self.dataset.mat.transpose()

        if self.dataset.preprocessing_info["Normalization"] is None:
             # needs to be lpog2 transformed for fold change calculations
            transposed = transposed.transform(lambda x: np.log2(x))

        transposed[self.dataset.index_column] = transposed.index
        transposed = transposed.reset_index(drop=True)

        res_ttest, tlim_ttest = multicova.perform_ttest_analysis(
            transposed,
            c1 =list(self.dataset.metadata[self.dataset.metadata[self.column]==self.group1][self.dataset.sample]),                                      
            c2 =list(self.dataset.metadata[self.dataset.metadata[self.column]==self.group2][self.dataset.sample]), 
            s0=0.05, 
            n_perm=self.perm,
            fdr=self.fdr,
            id_col=self.dataset.index_column,
            parallelize=True
        )
        
        fdr_column = "FDR"  + str(int(self.fdr*100)) + "%"
        self.res = res_ttest[[self.dataset.index_column, 'fc', 'tval', 'pval', 'tval_s0', 'pval_s0', 'qval']]
        self.res["log2fc"] = res_ttest["fc"]
        self.res["FDR"] = res_ttest[fdr_column]
        self.tlim_ttest = tlim_ttest
        self.pvalue_column = "pval"


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
        """
        convert pvalue to log10
        add color labels for up and down regulates
        """
        self.res = self.res[(self.res["log2fc"] < 10) & (self.res["log2fc"] > -10)]
        self.res["-log10(p-value)"] = -np.log10(self.res[self.pvalue_column])
        
        self.alpha = -np.log10(self.alpha)
        # add color variable to plot

        if self.method != "sam":
        
            condition = [
                (self.res["log2fc"] < -self.min_fc) & (self.res["-log10(p-value)"] > self.alpha),
                (self.res["log2fc"] > self.min_fc) & (self.res["-log10(p-value)"] > self.alpha),
            ]

        else:

            condition = [
                (self.res["log2fc"] < 0) & (self.res["FDR"] == "sig"),
                (self.res["log2fc"] > 0) & (self.res["FDR"] == "sig"),
            ]


        value = ["down", "up"]
        self.res["color"] = np.select(condition, value, default="non_sig")   
        

    def _add_labels_plot(self):
        """
        add gene names as hover data if they are given
        """

        if self.dataset.gene_names is not None:
            label_column = self.dataset.gene_names
        else:
            label_column = self.dataset.index_column

        self.res["label"] = np.where(
            self.res.color != "non_sig", self.res[label_column], ""
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
        
        self.plot.add_trace(go.Scatter(
            x=self.fdr_line[self.fdr_line.fc_s > 0].fc_s,
            y=-np.log10(self.fdr_line[self.fdr_line.fc_s > 0].pvals),
            line_color="black",
            line_shape='spline',
            showlegend=False)
        )
        self.plot.add_trace(go.Scatter(
            x=self.fdr_line[self.fdr_line.fc_s < 0].fc_s,
            y=-np.log10(self.fdr_line[self.fdr_line.fc_s < 0].pvals),
            line_color="black",
            line_shape='spline',
            showlegend=False)
        )


    def _plot(self):
        self.plot = px.scatter(
            self.res,
            x="log2fc",
            y="-log10(p-value)",
            color="color",
            hover_data=self.hover_data,
        )
        
        # update coloring
        color_dict = {"non_sig": "#404040", "up": "#B65EAF", "down": "#009599"}
        self.plot = self._update_colors_plotly(self.plot, color_dict=color_dict)

        if self.labels:
            self._add_labels_plot()
        
        if self.draw_line:
            if self.method == "sam":
                self._draw_fdr_line()
            else:
                self._draw_lines_plot()

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
    

   
