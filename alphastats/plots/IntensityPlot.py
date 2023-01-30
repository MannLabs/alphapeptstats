
import logging
import scipy
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from alphastats.plots.PlotUtils import plotly_object, PlotUtils


class IntensityPlot(PlotUtils):
    def __init__(self,
        dataset,
        protein_id,
        group,
        subgroups,
        method,
        add_significance,
        log_scale,
    ) -> None:
        self.dataset = dataset
        self.protein_id = protein_id
        self.group = group
        self.subgroups = subgroups
        self.method = method
        self.add_significance = add_significance
        self.log_scale = log_scale

        self._prepare_data()
        self._plot()
 

    @staticmethod
    def _add_significance(plot):
        # add sginficance pvalue, and stars to pairwise intensity plot
        plot_dict = plot.to_plotly_json()
        data = plot_dict.get("data")

        if len(data) != 2:
            logging.warning(
                "Signficane can only be estimated when there are two groups plotted."
            )
            return plot

        group1, group2 = data[0]["name"], data[1]["name"]
        y_array1, y_array2 = data[0]["y"], data[1]["y"]
        #  do ttest
        pvalue = scipy.stats.ttest_ind(y_array1, y_array2).pvalue

        pvalue_text = "<i>p=" + str(round(pvalue, 4)) + "</i>"

        if pvalue < 0.001:
            significance_level = "***"
            pvalue_text = "<i>p<0.001</i>"
        
        elif pvalue < 0.01:
            significance_level = "**"
       
        elif pvalue < 0.05:
            significance_level = "*"
       
        else:
            significance_level = "-"

        y_max = np.concatenate((y_array1, y_array2)).max()
        # add connecting bar for pvalue
        plot.add_trace(
            go.Scatter(
                x=[group1, group1, group2, group2],
                y=[y_max * 1.1, y_max * 1.15, y_max * 1.15, y_max * 1.1],
                fill=None,
                mode="lines",
                line=dict(color="rgba(0,0,0,1)", width=1),
                showlegend=False,
            )
        )

        # Add p-values
        plot.add_annotation(
            text=pvalue_text,
            name="p-value",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.95,
            showarrow=False,
            font=dict(size=12, color="black"),
        )

        plot.add_annotation(
            text=significance_level,
            name="significance",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.002,
            showarrow=False,
            font=dict(size=12, color="black"),
        )

        plot.update_layout(width=600, height=700)
        return plot
    
    def _prepare_data(self):
        #  TODO use difflib to find similar ProteinId if ProteinGroup is not present
        df = self.dataset.mat[[self.protein_id]].reset_index().rename(columns={"index": self.dataset.sample})
        df = df.merge(self.dataset.metadata, how="inner", on=[self.dataset.sample])

        if self.subgroups is not None:
            df = df[df[self.group].isin(self.subgroups)]

        self.y_label = self.protein_id + " - " + self.dataset.intensity_column.replace("[sample]", "")
        self.prepared_df = df

    def _plot(self):
        if self.method == "violin":
            fig = px.violin(
                self.prepared_df, y=self.protein_id, x=self.group, color=self.group, labels={self.protein_id: self.y_label}
            )

        elif self.method == "box":
            fig = px.box(
                self.prepared_df, y=self.protein_id, x=self.group, color=self.group, labels={self.protein_id: self.y_label}
            )

        elif self.method == "scatter":
            fig = px.scatter(
                 self.prepared_df, y=self.protein_id, x=self.group, color=self.group, labels={self.protein_id: self.y_label}
            )

        else:
            raise ValueError(
                f"{self.method} is not available."
                + "Please select from 'violin' for Violinplot, 'box' for Boxplot and 'scatter' for Scatterplot."
            )

        if self.log_scale:
            fig.update_layout(yaxis=dict(type="log"))

        if self.add_significance:
            fig = self._add_significance(fig)

        fig = plotly_object(fig)
        fig = self._update_figure_attributes(
            figure_object=fig, plotting_data= self.prepared_df, preprocessing_info=self.dataset.preprocessing_info,  method=self.method
        )

        self.plot = fig


