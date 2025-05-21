"""Module providing frontend widgets for gathering parameters and mapping them to the actual analysis."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from alphastats.dataset.dataset import DataSet
from alphastats.dataset.keys import Cols, ConstantsClass
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.gui.utils.result import (
    DifferentialExpressionTwoGroupsResult,
    ResultComponent,
)
from alphastats.gui.utils.ui_helper import AnalysisParameters
from alphastats.plots.plot_utils import PlotlyObject
from alphastats.plots.volcano_plot import VolcanoPlot
from alphastats.tl.differential_expression_analysis import (
    DeaTestTypes,
    DifferentialExpressionAnalysisTTest,
)


class PlottingOptions(metaclass=ConstantsClass):
    """Keys for the plotting options, the order determines order in UI."""

    VOLCANO_PLOT = "Volcano Plot"
    PCA_PLOT = "PCA Plot"
    UMAP_PLOT = "UMAP Plot"
    TSNE_PLOT = "t-SNE Plot"
    SAMPLE_DISTRIBUTION_PLOT = "Sampledistribution Plot"
    INTENSITY_PLOT = "Intensity Plot"
    CLUSTERMAP = "Clustermap"
    DENDROGRAM = "Dendrogram"


class StatisticOptions(metaclass=ConstantsClass):
    """Keys for the statistical options, the order determines order in UI."""

    DIFFERENTIAL_EXPRESSION = "Differential Expression Analysis"
    TUKEY_TEST = "Tukey-Test"
    ANOVA = "ANOVA"
    ANCOVA = "ANCOVA"


class NewAnalysisOptions(metaclass=ConstantsClass):
    """Keys for the new analysis options, the order determines order in UI."""

    DIFFERENTIAL_EXPRESSION_TWO_GROUPS = "Differential Expression Analysis (Two Groups)"


class AnalysisComponent(ABC):
    """Abstract class for analysis widgets."""

    _works_with_nans = True

    def __init__(self, dataset):
        self._dataset: DataSet = dataset
        # Note: parameters that are accessed but are not present are added to the dict with value "None"
        self._parameters = defaultdict(lambda: None)

    def show_widget(self):  # noqa: B027
        """Show the widget and gather parameters."""
        pass

    def do_analysis(
        self,
    ) -> Tuple[
        Union[PlotlyObject, pd.DataFrame], Optional[VolcanoPlot], Dict[str, Any]
    ]:
        """Perform the analysis after some upfront method-dependent checks (e.g. or NaNs).

        Returns a tuple(analysis, analysis_object, parameters) where 'analysis' is the plot or dataframe,
        'analysis_object' is the underlying object, 'parameters' is a dictionary of the parameters used.
        """
        try:
            self._nan_check()
            self._pre_analysis_check()
        except ValueError as e:
            st.error(str(e))
            st.stop()

        analysis, analysis_object = self._do_analysis()
        return analysis, analysis_object, dict(self._parameters)

    @abstractmethod
    def _do_analysis(
        self,
    ) -> Tuple[
        Union[PlotlyObject, pd.DataFrame, ResultComponent], Optional[VolcanoPlot]
    ]:
        pass

    def _nan_check(self) -> None:  # noqa: B027
        """Raise ValueError for methods that do not tolerate NaNs if there are any."""
        if not self._works_with_nans and self._dataset.mat.isnan().values.any():
            raise ValueError("This analysis does not work with NaN values.")
        # TODO: raises attribute error for isnan during wald analysis

    def _pre_analysis_check(self) -> None:  # noqa: B027
        """Perform pre-analysis check, raise ValueError on fail."""
        pass


class AbstractGroupCompareAnalysis(AnalysisComponent, ABC):
    """Abstract class for group comparison analysis widgets."""

    def show_widget(self):
        """Gather parameters to compare two group."""

        metadata = self._dataset.metadata

        default_option = "<select>"
        metadata_groups = metadata.columns.to_list()
        custom_group_option = "Custom groups from samples .."

        options = [default_option] + metadata_groups + [custom_group_option]
        grouping_variable = st.selectbox(
            "Grouping variable",
            options=options,
            index=options.index(
                st.session_state.get(
                    AnalysisParameters.TWOGROUP_COLUMN,
                    default_option if len(metadata_groups) == 0 else metadata_groups[0],
                )
            ),
            key=AnalysisParameters.TWOGROUP_COLUMN,
        )

        column = None
        if grouping_variable == default_option:
            group1 = st.selectbox("Group 1", options=[])
            group2 = st.selectbox("Group 2", options=[])

        elif grouping_variable != custom_group_option:
            unique_values = metadata[grouping_variable].unique().tolist()

            column = grouping_variable
            group1 = st.selectbox(
                "Group 1",
                options=unique_values,
                key=AnalysisParameters.TWOGROUP_GROUP1,
            )
            group2 = st.selectbox(
                "Group 2",
                options=list(reversed(unique_values)),
                key=AnalysisParameters.TWOGROUP_GROUP2,
            )

        else:
            group1 = st.multiselect(
                "Group 1 samples:",
                options=metadata[Cols.SAMPLE].to_list(),
                key=AnalysisParameters.TWOGROUP_GROUP1 + "multi",
            )

            group2 = st.multiselect(
                "Group 2 samples:",
                options=list(reversed(metadata[Cols.SAMPLE].to_list())),
                key=AnalysisParameters.TWOGROUP_GROUP2 + "multi",
            )

            intersection_list = list(set(group1).intersection(set(group2)))
            if len(intersection_list) > 0:
                st.warning(
                    "Group 1 and Group 2 contain same samples: "
                    + str(intersection_list)
                )

        self._parameters.update(
            {
                AnalysisParameters.TWOGROUP_GROUP1: group1,
                AnalysisParameters.TWOGROUP_GROUP2: group2,
            }
        )
        if column is not None:
            self._parameters[AnalysisParameters.TWOGROUP_COLUMN] = column

    def _pre_analysis_check(self):
        """Raise if selected groups are different."""
        if (
            self._parameters[AnalysisParameters.TWOGROUP_GROUP1]
            == self._parameters[AnalysisParameters.TWOGROUP_GROUP2]
        ):
            raise (
                ValueError(
                    "Group 1 and Group 2 can not be the same. Please select different groups."
                )
            )


class AbstractDimensionReductionAnalysis(AnalysisComponent, ABC):
    """Abstract class for dimension reduction analysis widgets."""

    def show_widget(self):
        """Gather parameters for dimension reduction analysis."""

        group = st.selectbox(
            "Color according to",
            options=[None] + self._dataset.metadata.columns.to_list(),
        )

        circle = st.checkbox("circle")

        self._parameters.update({"circle": circle, "group": group})


class AbstractIntensityPlot(AnalysisComponent, ABC):
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

        protein_id_or_gene_name = st.selectbox(
            "Gene or protein identifier to plot",
            options=list(self._dataset._gene_to_features_map.keys())
            + list(self._dataset._protein_to_features_map.keys()),
        )

        self._parameters.update(
            {"protein_id": protein_id_or_gene_name},
        )

    def _do_analysis(self):
        """Draw Intensity Plot using the IntensityPlot class."""
        intensity_plot = self._dataset.plot_intensity(
            feature=self._parameters["protein_id"],
            method=self._parameters["method"],
            group=self._parameters["group"],
        )
        return intensity_plot, None


class SampleDistributionPlot(AbstractIntensityPlot, ABC):
    """Abstract class for sampledistribution_plot analysis widgets."""

    def _do_analysis(self):
        """Draw Intensity Plot using the IntensityPlot class."""
        intensity_plot = self._dataset.plot_sampledistribution(
            method=self._parameters["method"],
            color=self._parameters["group"],  # no typo
        )
        return intensity_plot, None


class PCAPlotAnalysis(AbstractDimensionReductionAnalysis):
    """Widget for PCA Plot analysis."""

    def _do_analysis(self):
        """Draw PCA Plot using the PCAPlot class."""

        pca_plot = self._dataset.plot_pca(
            group=self._parameters["group"],
            circle=self._parameters["circle"],
        )
        return pca_plot, None


class UMAPPlotAnalysis(AbstractDimensionReductionAnalysis):
    """Widget for UMAP Plot analysis."""

    def _do_analysis(self):
        """Draw PCA Plot using the PCAPlot class."""
        umap_plot = self._dataset.plot_umap(
            group=self._parameters["group"],
            circle=self._parameters["circle"],
        )
        return umap_plot, None


class TSNEPlotAnalysis(AbstractDimensionReductionAnalysis):
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
        return tsne_plot, None


class VolcanoPlotAnalysis(AbstractGroupCompareAnalysis):
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

        # TODO: The sam fdr cutoff should be mutually exclusive with alpha
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
        # TODO: This is the place, where the new workflow of run/fetch DEA, filter significance, create plot should live. 1. self._dataset.get_dea(**parameters1), 2. dea.get_signficance(result, parameters2), 3. plot_volcano(result, significance, parameters3)
        # Note that currently, values that are not set by they UI would still be passed as None to the VolcanoPlot class,
        # thus overwriting the default values set therein.
        # If we introduce optional parameters in the UI, either use `inspect` to get the defaults from the class,
        # or refactor it so that all default values are `None` and the class sets the defaults programmatically.
        volcano_plot = VolcanoPlot(
            mat=self._dataset.mat,
            rawinput=self._dataset.rawinput,
            metadata=self._dataset.metadata,
            preprocessing_info=self._dataset.preprocessing_info,
            feature_to_repr_map=self._dataset._feature_to_repr_map,
            group1=self._parameters[AnalysisParameters.TWOGROUP_GROUP1],
            group2=self._parameters[AnalysisParameters.TWOGROUP_GROUP2],
            column=self._parameters[AnalysisParameters.TWOGROUP_COLUMN],
            method=self._parameters[AnalysisParameters.DEA_TWOGROUPS_METHOD],
            labels=self._parameters["labels"],
            min_fc=self._parameters["min_fc"],
            alpha=self._parameters["alpha"],
            draw_line=self._parameters["draw_line"],
            perm=self._parameters["perm"],
            fdr=self._parameters["fdr"],
            color_list=self._parameters["color_list"],
        )
        # TODO currently there's no other way to obtain both the plot and the underlying data
        #  Should be refactored such that the interface provided by DateSet.plot_volcano() is used
        #  One option could be to always return the whole analysis object.

        return volcano_plot.plot, volcano_plot


class ClustermapAnalysis(AnalysisComponent):
    """Widget for Clustermap analysis."""

    _works_with_nans = False

    def _do_analysis(self):
        """Draw Clustermap using the Clustermap class."""
        clustermap = self._dataset.plot_clustermap()
        return clustermap, None


class DendrogramAnalysis(AnalysisComponent):
    """Widget for Dendrogram analysis."""

    _works_with_nans = False

    def _do_analysis(self):
        """Draw Clustermap using the Clustermap class."""
        dendrogram = self._dataset.plot_dendrogram()
        return dendrogram, None


class DifferentialExpressionAnalysis(AbstractGroupCompareAnalysis):
    """Widget for differential expression analysis."""

    # TODO: This functionality will disappear and become a part of the VolcanoPlot class. This will produce a widget to select whether the result should be displayed as table or as plot.

    def show_widget(self):
        """Show the widget and gather parameters."""

        method = st.selectbox(
            "Differential Analysis using:",
            options=["ttest", "wald"],
        )

        if method == "wald":
            self._works_with_nans = False

        super().show_widget()

        self._parameters.update({AnalysisParameters.DEA_TWOGROUPS_METHOD: method})

    def _do_analysis(self):
        """Perform T-test analysis."""
        diff_exp_analysis = self._dataset.diff_expression_analysis(
            method=self._parameters[AnalysisParameters.DEA_TWOGROUPS_METHOD],
            group1=self._parameters[AnalysisParameters.TWOGROUP_GROUP1],
            group2=self._parameters[AnalysisParameters.TWOGROUP_GROUP2],
            column=self._parameters[AnalysisParameters.TWOGROUP_COLUMN],
        )
        return diff_exp_analysis, None


class TukeyTestAnalysis(AnalysisComponent):
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
        return tukey_test_analysis, None


class AnovaAnalysis(AbstractGroupCompareAnalysis):
    """Widget for ANOVA analysis."""

    def show_widget(self):
        """Show the widget and gather parameters."""

        column = st.selectbox(
            "A variable from the metadata to calculate ANOVA",
            options=self._dataset.metadata.columns.to_list(),
        )
        protein_ids = st.selectbox(
            "All ProteinIDs/or specific ProteinID to perform ANOVA",
            options=["all"] + self._dataset.mat.columns.to_list(),
        )

        tukey = st.checkbox("Follow-up Tukey")

        self._parameters.update(
            {"column": column, "protein_ids": protein_ids, "tukey": tukey}
        )

    def _do_analysis(self):
        """Perform Anova analysis."""
        anova_analysis = self._dataset.anova(
            column=self._parameters["column"],
            protein_ids=self._parameters["protein_ids"],
            tukey=self._parameters["tukey"],
        )
        return anova_analysis, None


class AncovaAnalysis(AnalysisComponent):
    """Widget for Ancova analysis."""

    def show_widget(self):
        """Show the widget and gather parameters."""

        protein_id = st.selectbox(
            "ProteinID/ProteinGroup",
            options=self._dataset.mat.columns.to_list(),
        )
        covar = st.selectbox(
            "Name(s) of column(s) in metadata with the covariate.",
            options=self._dataset.metadata.columns.to_list(),
        )  # TODO: why plural if only one can be selected?
        between = st.selectbox(
            "Name of the column in the metadata with the between factor.",
            options=self._dataset.metadata.columns.to_list(),
        )

        self._parameters.update(
            {"protein_id": protein_id, "covar": covar, "between": between}
        )

    def _do_analysis(self):
        """Perform ANCOVA analysis."""
        ancova_analysis = self._dataset.ancova(
            protein_id=self._parameters["protein_id"],
            covar=self._parameters["covar"],
            between=self._parameters["between"],
        )
        return ancova_analysis, None


class DifferentialExpressionTwoGroupsAnalysis(AbstractGroupCompareAnalysis):
    """Widget for Differential expression analysis between two groups."""

    def show_widget(self):
        """Show the widget and gather parameters."""
        super().show_widget()

        parameters = {}
        method = st.selectbox(
            "Differential Analysis using:",
            options=["independent t-test", "paired t-test"],
            key=AnalysisParameters.DEA_TWOGROUPS_METHOD,
        )
        parameters[AnalysisParameters.DEA_TWOGROUPS_METHOD] = method

        fdr_method = st.selectbox(
            "FDR method",
            options=["fdr_bh", "bonferroni"],
            index=0,
            format_func=lambda x: {
                "fdr_bh": "Benjamini-Hochberg",
                "bonferroni": "Bonferroni",
            }[x],
            key=AnalysisParameters.DEA_TWOGROUPS_FDR_METHOD,
        )
        parameters[AnalysisParameters.DEA_TWOGROUPS_FDR_METHOD] = fdr_method

        self._parameters.update(parameters)

    def _do_analysis(self) -> Tuple[ResultComponent, None]:
        """Run the differential expression analysis between two groups and return the corresponding results object."""

        test_type = {
            "independent t-test": DeaTestTypes.INDEPENDENT,
            "paired t-test": DeaTestTypes.PAIRED,
        }[self._parameters[AnalysisParameters.DEA_TWOGROUPS_METHOD]]

        dea = DifferentialExpressionAnalysisTTest(
            self._dataset.mat,
            is_log2_transformed=self._dataset.preprocessing_info[
                PreprocessingStateKeys.LOG2_TRANSFORMED
            ],
        )
        dea_result = dea.perform(
            test_type=test_type,
            group1=self._parameters[AnalysisParameters.TWOGROUP_GROUP1],
            group2=self._parameters[AnalysisParameters.TWOGROUP_GROUP2],
            grouping_column=self._parameters[AnalysisParameters.TWOGROUP_COLUMN],
            metadata=self._dataset.metadata,
            fdr_method=self._parameters[AnalysisParameters.DEA_TWOGROUPS_FDR_METHOD],
        )

        return DifferentialExpressionTwoGroupsResult(
            dea_result,
            preprocessing=self._dataset.preprocessing_info,
            method=self._parameters,
            feature_to_repr_map=self._dataset._feature_to_repr_map,
            is_plottable=True,
        ), None  # None is for backwards compatibility


# TODO: Merge functionality from old DEA and Volcano to new DEA (other tests, multicova) and delete the old classes afterwards.

ANALYSIS_OPTIONS = {
    # PlottingOptions.VOLCANO_PLOT: VolcanoPlotAnalysis,
    PlottingOptions.PCA_PLOT: PCAPlotAnalysis,
    PlottingOptions.UMAP_PLOT: UMAPPlotAnalysis,
    PlottingOptions.TSNE_PLOT: TSNEPlotAnalysis,
    PlottingOptions.SAMPLE_DISTRIBUTION_PLOT: SampleDistributionPlot,
    PlottingOptions.INTENSITY_PLOT: IntensityPlot,
    PlottingOptions.CLUSTERMAP: ClustermapAnalysis,
    PlottingOptions.DENDROGRAM: DendrogramAnalysis,
    # StatisticOptions.DIFFERENTIAL_EXPRESSION: DifferentialExpressionAnalysis,
    StatisticOptions.TUKEY_TEST: TukeyTestAnalysis,
    StatisticOptions.ANOVA: AnovaAnalysis,
    StatisticOptions.ANCOVA: AncovaAnalysis,
    NewAnalysisOptions.DIFFERENTIAL_EXPRESSION_TWO_GROUPS: DifferentialExpressionTwoGroupsAnalysis,
}
