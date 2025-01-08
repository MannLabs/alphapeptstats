"""Module providing frontend widgets for gathering parameters and mapping them to the actual analysis."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Literal, Optional, Tuple, Union

import pandas as pd
import streamlit as st
from plotly.graph_objs._figure import Figure

from alphastats.dataset.dataset import DataSet
from alphastats.dataset.keys import Cols, ConstantsClass
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.gui.utils.ui_helper import StateKeys
from alphastats.pl.volcano import _plot_volcano, prepare_result_df
from alphastats.plots.plot_utils import PlotlyObject
from alphastats.plots.volcano_plot import VolcanoPlot
from alphastats.tl.differential_expression_analysis import (
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


# TODO rename to AnalysisComponent
class AbstractAnalysis(ABC):
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
    ) -> Tuple[Union[PlotlyObject, pd.DataFrame], Optional[VolcanoPlot]]:
        pass

    def _nan_check(self) -> None:  # noqa: B027
        """Raise ValueError for methods that do not tolerate NaNs if there are any."""
        if not self._works_with_nans and self._dataset.mat.isnan().values.any():
            raise ValueError("This analysis does not work with NaN values.")
        # TODO: raises attribute error for isnan during wald analysis

    def _pre_analysis_check(self) -> None:  # noqa: B027
        """Perform pre-analysis check, raise ValueError on fail."""
        pass


class AbstractGroupCompareAnalysis(AbstractAnalysis, ABC):
    """Abstract class for group comparison analysis widgets."""

    def show_widget(self):
        """Gather parameters to compare two group."""

        metadata = self._dataset.metadata

        default_option = "<select>"
        custom_group_option = "Custom groups from samples .."

        grouping_variable = st.selectbox(
            "Grouping variable",
            options=[default_option]
            + metadata.columns.to_list()
            + [custom_group_option],
        )

        column = None
        if grouping_variable == default_option:
            group1 = st.selectbox("Group 1", options=[])
            group2 = st.selectbox("Group 2", options=[])

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

        self._parameters.update({"group1": group1, "group2": group2})
        if column is not None:
            self._parameters["column"] = column

    def _pre_analysis_check(self):
        """Raise if selected groups are different."""
        if self._parameters["group1"] == self._parameters["group2"]:
            raise (
                ValueError(
                    "Group 1 and Group 2 can not be the same. Please select different groups."
                )
            )


class AbstractDimensionReductionAnalysis(AbstractAnalysis, ABC):
    """Abstract class for dimension reduction analysis widgets."""

    def show_widget(self):
        """Gather parameters for dimension reduction analysis."""

        group = st.selectbox(
            "Color according to",
            options=[None] + self._dataset.metadata.columns.to_list(),
        )

        circle = st.checkbox("circle")

        self._parameters.update({"circle": circle, "group": group})


class AbstractIntensityPlot(AbstractAnalysis, ABC):
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
            {
                "protein_id": self._dataset._gene_to_features_map[
                    protein_id_or_gene_name
                ]
                if protein_id_or_gene_name in self._dataset._gene_to_features_map
                else self._dataset._protein_to_features_map[protein_id_or_gene_name]
            }
        )

    def _do_analysis(self):
        """Draw Intensity Plot using the IntensityPlot class."""
        intensity_plot = self._dataset.plot_intensity(
            protein_id=self._parameters["protein_id"],
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
        # TODO currently there's no other way to obtain both the plot and the underlying data
        #  Should be refactored such that the interface provided by DateSet.plot_volcano() is used
        #  One option could be to always return the whole analysis object.

        return volcano_plot.plot, volcano_plot


class ClustermapAnalysis(AbstractAnalysis):
    """Widget for Clustermap analysis."""

    _works_with_nans = False

    def _do_analysis(self):
        """Draw Clustermap using the Clustermap class."""
        clustermap = self._dataset.plot_clustermap()
        return clustermap, None


class DendrogramAnalysis(AbstractAnalysis):
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

        self._parameters.update({"method": method})

    def _do_analysis(self):
        """Perform T-test analysis."""
        diff_exp_analysis = self._dataset.diff_expression_analysis(
            method=self._parameters["method"],
            group1=self._parameters["group1"],
            group2=self._parameters["group2"],
            column=self._parameters["column"],
        )
        return diff_exp_analysis, None


class TukeyTestAnalysis(AbstractAnalysis):
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


class AncovaAnalysis(AbstractAnalysis):
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
    """Widget for Volcano Plot analysis."""

    def show_widget(self):
        """Show the widget and gather parameters."""
        super().show_widget()

        parameters = {}
        method = st.selectbox(
            "Differential Analysis using:",
            options=["ttest"],
        )
        parameters["method"] = method

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
        if self._parameters["method"] == "ttest":
            dea = DifferentialExpressionAnalysisTTest(
                self._dataset.mat,
                self._dataset.preprocessing_info[
                    PreprocessingStateKeys.LOG2_TRANSFORMED
                ],
            )
            dea_result = dea.perform(
                test_type="independent",
                group1=self._parameters["group1"],
                group2=self._parameters["group2"],
                grouping_column=self._parameters["column"],
                metadata=self._dataset.metadata,
                fdr_method="fdr_bh",
            )

        return DifferentialExpressionTwoGroupsResult(
            dea_result,
            preprocessing=self._dataset.preprocessing_info,
            method=self._parameters,
        ), None


ANALYSIS_OPTIONS = {
    PlottingOptions.VOLCANO_PLOT: VolcanoPlotAnalysis,
    PlottingOptions.PCA_PLOT: PCAPlotAnalysis,
    PlottingOptions.UMAP_PLOT: UMAPPlotAnalysis,
    PlottingOptions.TSNE_PLOT: TSNEPlotAnalysis,
    PlottingOptions.SAMPLE_DISTRIBUTION_PLOT: SampleDistributionPlot,
    PlottingOptions.INTENSITY_PLOT: IntensityPlot,
    PlottingOptions.CLUSTERMAP: ClustermapAnalysis,
    PlottingOptions.DENDROGRAM: DendrogramAnalysis,
    StatisticOptions.DIFFERENTIAL_EXPRESSION: DifferentialExpressionAnalysis,
    StatisticOptions.TUKEY_TEST: TukeyTestAnalysis,
    StatisticOptions.ANOVA: AnovaAnalysis,
    StatisticOptions.ANCOVA: AncovaAnalysis,
    NewAnalysisOptions.DIFFERENTIAL_EXPRESSION_TWO_GROUPS: DifferentialExpressionTwoGroupsAnalysis,
}


class ResultObject(ABC):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        plottable: bool,
        preprocessing: Dict,
        method: Dict,
    ):
        if dataframe is None:
            raise ValueError("Either dataframe or plot must be provided")

        self.dataframe = dataframe
        self.annotated_dataframe = dataframe
        self.plottable = plottable
        self.plot = None
        self.preprocessing = preprocessing
        self.method = method
        self.data_annotation_options = {}
        self.display_options = {}
        self.display_selection = "Plot" if self.plottable else "Dataframe"

    def _apply_data_annotation_options(self):
        """Function to get all significance options for the analysis object"""
        self.data_annotation_options = self._get_data_annotation_options()
        self.annotated_dataframe = self._update_data_annotation(
            **self.data_annotation_options
        )

    @abstractmethod
    def _get_data_annotation_options(self) -> Dict:
        pass

    @abstractmethod
    def _update_data_annotation(self, **kwargs) -> pd.DataFrame:
        pass

    def _apply_display_options(self):
        """Funciton to get all display options for the analysis object"""
        if self.plottable is False:
            display_selection_options = ["Raw dataframe", "Annotated dataframe"]
        else:
            display_selection_options = ["Plot", "Raw dataframe", "Annotated dataframe"]
        display_selection = st.radio(
            "Select display",
            display_selection_options,
            index=display_selection_options.index(
                st.session_state.get("TMP_display_selection", self.display_selection)
            ),
            key="TMP_display_selection",
        )
        self.display_selection = display_selection
        if display_selection == "Plot":
            self.display_options = self._get_plot_options()
            self.plot = self._update_plot(**self.display_options)

    @abstractmethod
    def _get_plot_options(self) -> Dict:
        """Function to get all plotting related options for the analysis object"""
        pass

    @abstractmethod
    def _update_plot(self, **kwargs) -> Figure:
        """Function to update the display of the analysis object"""
        pass

    def _display_object(self):
        """Function to display the analysis object"""
        if self.display_selection == "Plot":
            st.plotly_chart(self.plot.update())
        elif self.display_selection == "Raw dataframe":
            st.dataframe(self.dataframe)
        elif self.display_selection == "Annotated dataframe":
            st.dataframe(self.annotated_dataframe)

    def display_object(
        self,
        display_column: st.container,
        data_annotation_editable: bool = False,
        display_editable: bool = False,
        widget_column: Optional[st.container] = None,
    ):
        """Function to display the object.
        The function will display the object in the display column and the options in the widget column.
        The boolean flags are intended for controlling behaviour in different sections of the application.

        Args:
            display_column (st.container): The container to display the object.
            data_annotation_editable (bool, optional): If the data_annotation options are editable. Defaults to False.
            display_editable (bool, optional): If the display options are editable. Defaults to False.
            widget_column (Optional[st.container], optional): The container to display the widgets. Defaults to None.

        Raises:
            ValueError: If the widget column container is not provided.
        """
        if data_annotation_editable or display_editable:
            if widget_column is None:
                raise ValueError("Widget column container must be provided")
            with widget_column:
                if data_annotation_editable:
                    self._apply_data_annotation_options()
                if display_editable:
                    self._apply_display_options()
                else:
                    self.plot = self._update_plot(**self.display_options)
        with display_column:
            self._display_object()

    def get_standard_layout_options(self):
        return {
            "height": st.number_input(
                "Height",
                200,
                1000,
                st.session_state.get(
                    "TMP_height", self.display_options.get("height", 500)
                ),
                10,
                key="TMP_height",
            ),
            "width": st.number_input(
                "Width",
                200,
                1000,
                st.session_state.get(
                    "TMP_width", self.display_options.get("width", 500)
                ),
                10,
                key="TMP_width",
            ),
            "showlegend": st.checkbox(
                "Show legend",
                st.session_state.get(
                    "TMP_showlegend", self.display_options.get("showlegend", 500)
                ),
                key="TMP_showlegend",
            ),
        }


class DifferentialExpressionTwoGroupsResult(ResultObject):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessing: Dict,
        method: Dict,
    ):
        super().__init__(
            dataframe, plottable=True, preprocessing=preprocessing, method=method
        )
        self.log2name = ""

    def _get_data_annotation_options(self) -> Dict:
        return {
            "qvalue_cutoff": st.number_input(
                "Q-value cutoff",
                0.0,
                1.0,
                st.session_state.get(
                    "TMP_qvalue_cutoff",
                    self.data_annotation_options.get("qvalue_cutoff", 0.05),
                ),
                0.01,
                format="%.2f",
                key="TMP_qvalue_cutoff",
            ),
            "log2fc_cutoff": st.number_input(
                "Log2FC cutoff",
                0.0,
                10.0,
                st.session_state.get(
                    "TMP_log2fc_cutoff",
                    self.data_annotation_options.get("log2fc_cutoff", 1.0),
                ),
                0.1,
                format="%.1f",
                key="TMP_log2fc_cutoff",
            ),
            "flip_xaxis": st.checkbox(
                "Flip groups",
                st.session_state.get(
                    "TMP_flip_xaxis",
                    self.data_annotation_options.get("flip_xaxis", False),
                ),
                key="TMP_flip_xaxis",
            ),
        }

    def _update_data_annotation(
        self,
        qvalue_cutoff: float,
        log2fc_cutoff: float,
        flip_xaxis: bool,
    ) -> pd.DataFrame:
        formatted_df, log2name = prepare_result_df(
            statistics_results=self.dataframe,
            feature_to_repr_map=st.session_state[
                StateKeys.DATASET
            ]._feature_to_repr_map,
            group1=self.method["group1"],
            group2=self.method["group2"],
            qvalue_cutoff=qvalue_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            flip_xaxis=flip_xaxis,
        )
        self.log2name = log2name
        return formatted_df

    def _get_plot_options(self) -> Dict:
        with st.expander("Display options"):
            renderer_options = ["webgl", "svg"]
            return {
                **{
                    "drawlines": st.checkbox(
                        "Draw significance and fold change lines",
                        st.session_state.get(
                            "TMP_drawlines", self.display_options.get("drawlines", True)
                        ),
                        key="TMP_drawlines",
                    ),
                    "label_significant": st.checkbox(
                        "Label significant points",
                        st.session_state.get(
                            "TMP_label_significant",
                            self.display_options.get("label_significant", True),
                        ),
                        key="TMP_label_significant",
                    ),
                    "renderer": st.radio(
                        "Renderer",
                        renderer_options,
                        index=renderer_options.index(
                            st.session_state.get(
                                "TMP_renderer",
                                self.display_options.get("renderer", "webgl"),
                            )
                        ),
                        key="TMP_renderer",
                    ),
                },
                **self.get_standard_layout_options(),
            }

    def _update_plot(
        self,
        drawlines: bool,
        label_significant: bool,
        renderer: Literal["webgl", "svg"],
        **kwargs,
    ) -> Figure:
        return _plot_volcano(
            df_plot=self.annotated_dataframe,
            log2name=self.log2name,
            group1=self.method["group1"],
            group2=self.method["group2"],
            qvalue_cutoff=self.data_annotation_options["qvalue_cutoff"],
            log2fc_cutoff=self.data_annotation_options["log2fc_cutoff"],
            flip_xaxis=self.data_annotation_options["flip_xaxis"],
            drawlines=drawlines,
            label_significant=label_significant,
            renderer=renderer,
            **kwargs,
        )
