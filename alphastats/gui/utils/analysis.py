"""Module providing frontend widgets for gathering parameters and mapping them to the actual analysis."""

from abc import ABC, abstractmethod
from collections import defaultdict

import streamlit as st

from alphastats.DataSet import DataSet
from alphastats.keys import Cols
from alphastats.plots.VolcanoPlot import VolcanoPlot


class ConstantsClass(type):
    """A metaclass for classes that should only contain string constants."""

    def __setattr__(self, name, value):
        raise TypeError("Constants class cannot be modified")

    def get_values(cls):
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]


class PlottingOptions(metaclass=ConstantsClass):
    """Keys for the plotting options."""

    PCA_PLOT = "PCA Plot"
    UMAP_PLOT = "UMAP Plot"
    TSNE_PLOT = "t-SNE Plot"
    VOLCANO_PLOT = "Volcano Plot"
    SAMPLE_DISTRIBUTION_PLOT = "Sampledistribution Plot"
    INTENSITY_PLOT = "Intensity Plot"
    CLUSTERMAP = "Clustermap"


class StatisticOptions(metaclass=ConstantsClass):
    DIFFERENTIAL_EXPRESSION = "Differential Expression Analysis"
    TUKEY_TEST = "Tukey-Test"


class Analysis(ABC):
    """Abstract class for analysis widgets."""

    _works_with_nans = True

    def __init__(self, dataset):
        self._dataset: DataSet = dataset
        self._parameters = defaultdict(lambda: None)

    def show_widget(self):  # noqa: B027
        """Show the widget and gather parameters."""
        pass

    def do_analysis(self):
        """Perform the analysis.

        Returns a tuple(figure, analysis_object, parameters) where figure is the plot,
        analysis_object is the underlying object, parameters is a dictionary of the parameters used.
        """
        if not self._works_with_nans and self._dataset.mat.isnull().values.any():
            st.error("This analysis does not work with NaN values.")
            st.stop()
        return self._do_analysis()

    @abstractmethod
    def _do_analysis(self):
        pass


class GroupCompareAnalysis(Analysis, ABC):
    """Abstract class for group comparison analysis widgets."""

    def show_widget(self):
        """Gather parameters to compare two group."""

        metadata = self._dataset.metadata

        default_option = "<select>"
        custom_group_option = "Custom group from samples .."

        grouping_variable = st.selectbox(
            "Grouping variable",
            options=[default_option]
            + metadata.columns.to_list()
            + [custom_group_option],
        )

        column = None
        if grouping_variable == default_option:
            st.stop()  # TODO: using stop here is not really great
        elif grouping_variable != custom_group_option:
            unique_values = metadata[grouping_variable].unique().tolist()

            column = grouping_variable
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
                "Group 1 and Group 2 can not be the same. Please select different groups."
            )
            st.stop()

        self._parameters.update({"group1": group1, "group2": group2})
        if column is not None:
            self._parameters["column"] = column


class DimensionReductionAnalysis(Analysis, ABC):
    """Abstract class for dimension reduction analysis widgets."""

    def show_widget(self):
        """Gather parameters for dimension reduction analysis."""

        group = st.selectbox(
            "Color according to",
            options=[None] + self._dataset.metadata.columns.to_list(),
        )

        circle = st.checkbox("circle")

        self._parameters.update({"circle": circle, "group": group})


class AbstractIntensityPlot(Analysis, ABC):
    """Abstract class for intensity plot analysis widgets."""

    def show_widget(self):
        """Gather parameters for intensity plot analysis."""

        group = st.selectbox(
            "Color according to",
            options=[None] + self._dataset.metadata.columns.to_list(),
        )
        method = st.selectbox(
            "Plot layout",
            options=["violin", "box", "scatter"],
        )

        self._parameters.update({"group": group, "method": method})


class IntensityPlot(AbstractIntensityPlot, ABC):
    """Abstract class for intensity plot analysis widgets."""

    def show_widget(self):
        """Gather parameters for intensity plot analysis."""
        super().show_widget()

        protein_id = st.selectbox(
            "ProteinID/ProteinGroup",
            options=self._dataset.mat.columns.to_list(),
        )

        self._parameters.update({"protein_id": protein_id})

    def _do_analysis(self):
        """Draw Intensity Plot using the IntensityPlot class."""
        intensity_plot = self._dataset.plot_intensity(
            protein_id=self._parameters["protein_id"],
            method=self._parameters["method"],
            group=self._parameters["group"],
        )
        return intensity_plot, None, self._parameters


class SampleDistributionPlot(AbstractIntensityPlot, ABC):
    """Abstract class for sampledistribution_plot analysis widgets."""

    def _do_analysis(self):
        """Draw Intensity Plot using the IntensityPlot class."""
        intensity_plot = self._dataset.plot_sampledistribution(
            method=self._parameters["method"],
            color=self._parameters["group"],  # no typo
        )
        return intensity_plot, None, self._parameters


class PCAPlotAnalysis(DimensionReductionAnalysis):
    """Widget for PCA Plot analysis."""

    def _do_analysis(self):
        """Draw PCA Plot using the PCAPlot class."""

        pca_plot = self._dataset.plot_pca(
            group=self._parameters["group"],
            circle=self._parameters["circle"],
        )
        return pca_plot, None, self._parameters


class UMAPPlotAnalysis(DimensionReductionAnalysis):
    """Widget for UMAP Plot analysis."""

    def _do_analysis(self):
        """Draw PCA Plot using the PCAPlot class."""
        umap_plot = self._dataset.plot_umap(
            group=self._parameters["group"],
            circle=self._parameters["circle"],
        )
        return umap_plot, None, self._parameters


class TSNEPlotAnalysis(DimensionReductionAnalysis):
    """Widget for t-SNE Plot analysis."""

    def show_widget(self):
        """Show the widget and gather parameters."""
        super().show_widget()

        n_iter = st.select_slider(
            "Maximum number of iterations for the optimization",
            range(250, 2001),
            value=1000,
        )
        perplexity = st.select_slider("Perplexity", range(5, 51), value=30)

        self._parameters.update(
            {
                "n_iter": n_iter,
                "perplexity": perplexity,
            }
        )

    def _do_analysis(self):
        """Draw t-SNE Plot using the TSNEPlot class."""
        tsne_plot = self._dataset.plot_tsne(
            group=self._parameters["group"],
            circle=self._parameters["circle"],
            perplexity=self._parameters["perplexity"],
            n_iter=self._parameters["n_iter"],
        )
        return tsne_plot, None, self._parameters


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

    def _do_analysis(self):
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


class ClustermapAnalysis(Analysis):
    """Widget for Clustermap analysis."""

    _works_with_nans = False

    def _do_analysis(self):
        """Draw Clustermap using the Clustermap class."""
        clustermap = self._dataset.plot_clustermap()
        return clustermap, None, self._parameters


class DifferentialExpressionAnalysis(GroupCompareAnalysis):
    """Widget for differential expression analysis."""

    def show_widget(self):
        """Show the widget and gather parameters."""

        method = st.selectbox(
            "Differential Analysis using:",
            options=["ttest", "wald"],
        )

        if method == "wald":
            self._works_with_nans = False

        super().show_widget()

        self._parameters.update({"method": method})

    def _do_analysis(self):
        """Perform T-test analysis."""
        diff_exp_analysis = self._dataset.diff_expression_analysis(
            method=self._parameters["method"],
            group1=self._parameters["group1"],
            group2=self._parameters["group2"],
            column=self._parameters["column"],
        )
        return diff_exp_analysis, None, self._parameters


class TukeyTestAnalysis(Analysis):
    """Widget for Tukey-Test analysis."""

    def show_widget(self):
        """Show the widget and gather parameters."""

        protein_id = st.selectbox(
            "ProteinID/ProteinGroup",
            options=self._dataset.mat.columns.to_list(),
        )
        group = st.selectbox(
            "A metadata variable to calculate a pairwise tukey test",
            options=self._dataset.metadata.columns.to_list(),
        )
        self._parameters.update({"protein_id": protein_id, "group": group})

    def _do_analysis(self):
        """Perform Tukey-test analysis."""
        tukey_test_analysis = self._dataset.tukey_test(
            protein_id=self._parameters["protein_id"],
            group=self._parameters["group"],
        )
        return tukey_test_analysis, None, self._parameters
