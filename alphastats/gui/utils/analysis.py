"""Module providing frontend widgets for gathering parameters and mapping them to the actual analysis."""

from abc import ABC, abstractmethod
from collections import defaultdict

import streamlit as st

from alphastats.keys import Cols
from alphastats.plots.VolcanoPlot import VolcanoPlot


class Analysis(ABC):
    """Abstract class for analysis widgets."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._parameters = defaultdict(lambda: None)

    @abstractmethod
    def show_widget(self):
        """Show the widget and gather parameters."""
        pass

    @abstractmethod
    def do_analysis(self):
        """Perform the analysis."""
        pass


class GroupCompareAnalysis(Analysis, ABC):
    """Abstract class for group comparison analysis widgets."""

    def show_widget(self):
        """Gather parameters to compare two group."""

        metadata = self._dataset.metadata

        default_option = "<None>"
        grouping_variable = st.selectbox(
            "Grouping variable",
            options=[default_option] + metadata.columns.to_list(),
        )

        if grouping_variable != default_option:
            unique_values = metadata[grouping_variable].unique().tolist()

            group1 = st.selectbox("Group 1", options=unique_values)
            group2 = st.selectbox("Group 2", options=list(reversed(unique_values)))

        else:
            group1 = st.multiselect(
                "Group 1 samples:",
                options=metadata[Cols.SAMPLE].to_list(),
            )

            group2 = st.multiselect(
                "Group 2 samples:",
                options=list(reversed(metadata[Cols.SAMPLE].to_list())),
            )

            intersection_list = list(set(group1).intersection(set(group2)))
            if len(intersection_list) > 0:
                st.warning(
                    "Group 1 and Group 2 contain same samples: "
                    + str(intersection_list)
                )

        if group1 == group2:
            st.error(
                "Group 1 and Group 2 can not be the same please select different group."
            )
            st.stop()

        self._parameters.update({"group1": group1, "group2": group2})


class VolcanoPlotAnalysis(GroupCompareAnalysis):
    """Widget for Volcano Plot analysis."""

    def show_widget(self):
        """Show the widget and gather parameters."""
        super().show_widget()

        parameters = {}
        method = st.selectbox(
            "Differential Analysis using:",
            options=["ttest", "anova", "wald", "sam", "paired-ttest", "welch-ttest"],
        )
        parameters["method"] = method

        parameters["labels"] = st.checkbox("Add labels", value=True)

        parameters["draw_line"] = st.checkbox("Draw lines", value=True)

        parameters["alpha"] = st.number_input(
            label="alpha", min_value=0.001, max_value=0.050, value=0.050
        )

        parameters["min_fc"] = st.select_slider(
            "Foldchange cutoff", range(0, 3), value=1
        )

        if method == "sam":
            parameters["perm"] = st.number_input(
                label="Number of Permutations", min_value=1, max_value=1000, value=10
            )
            parameters["fdr"] = st.number_input(
                label="FDR cut off", min_value=0.005, max_value=0.1, value=0.050
            )

        self._parameters.update(parameters)

    def do_analysis(self):
        """Draw Volcano Plot using the VolcanoPlot class.

        Returns a tuple(figure, analysis_object, parameters) where figure is the plot,
        analysis_object is the underlying object, parameters is a dictionary of the parameters used.
        """
        # TODO currently there's no other way to obtain both the plot and the underlying data
        #  Should be refactored such that the interface provided by DateSet.plot_volcano() is used
        #  One option could be to always return the whole analysis object.

        volcano_plot = VolcanoPlot(
            mat=self._dataset.mat,
            rawinput=self._dataset.rawinput,
            metadata=self._dataset.metadata,
            preprocessing_info=self._dataset.preprocessing_info,
            group1=self._parameters["group1"],
            group2=self._parameters["group2"],
            column=self._parameters["column"],
            method=self._parameters["method"],
            labels=self._parameters["labels"],
            min_fc=self._parameters["min_fc"],
            alpha=self._parameters["alpha"],
            draw_line=self._parameters["draw_line"],
            perm=self._parameters["perm"],
            fdr=self._parameters["fdr"],
            color_list=self._parameters["color_list"],
        )
        return volcano_plot.plot, volcano_plot, self._parameters
