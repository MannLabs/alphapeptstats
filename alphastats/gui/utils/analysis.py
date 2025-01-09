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


class RESULT_PARAMETERS(metaclass=ConstantsClass):
    WIDTH = "width"
    HEIGHT = "height"
    SHOWLEGEND = "showlegend"
    QVALUE_CUTOFF = "qvalue_cutoff"
    LOG2FC_CUTOFF = "log2fc_cutoff"
    FLIP_XAXIS = "flip_xaxis"
    DRAWLINES = "drawlines"
    LABEL_SIGNIFICANT = "label_significant"
    RENDERER = "renderer"


class ANALYSIS_PARAMETERS(metaclass=ConstantsClass):
    TWOGROUP_GROUP1 = "group1"
    TWOGROUP_GROUP2 = "group2"
    DEA_TWOGROUPS_METHOD = "method"
    DEA_TWOGROUPS_FDR_METHOD = "fdr_method"
    TWOGROUP_COLUMN = "column"


class ResultObject(ABC):  # move to new file
    """Base class for providing the UI for inspecting and parameterizing the analysis based of statistical results.

    The intended use is that in a first step data can be annotated e.g. based on significance cutoffs and then plotted, e.g. applying cosmetic preferences like lines and colors.

    The display function can parameterized to restrict editing options to 'freeze' the result.
    """

    class DISPLAY_SELECTION(metaclass=ConstantsClass):
        """Keys for the display options, the order determines order in UI."""

        PLOT = "Plot"
        RAW_DATAFRAME = "Raw Dataframe"
        ANNOTATED_DATAFRAME = "Annotated Dataframe"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        is_plottable: bool,
        preprocessing: Dict,
        method: Dict,
    ) -> None:
        """Initialize the ResultObject with a dataframe.

        Args:
            dataframe (pd.DataFrame): The dataframe to be used for the analysis.
            is_plottable (bool): If the analysis is plottable.
            preprocessing (Dict): The preprocessing information.
            method (Dict): The method used for the analysis.

        Raises:
            ValueError: If no dataframe is provided.
        """
        self.dataframe = dataframe
        self.annotated_dataframe = dataframe
        self._is_plottable = is_plottable
        self.preprocessing = preprocessing
        self.method = method

        self.plot: Optional[Figure] = None
        self._data_annotation_options = {}
        self._display_options = {}
        self._display_selection = (
            self.DISPLAY_SELECTION.PLOT
            if self._is_plottable
            else self.DISPLAY_SELECTION.RAW_DATAFRAME
        )

    def _apply_data_annotation_options(self, name: str = "") -> None:
        """Function to get and apply all options for data annotation.
        This sets the data_annotation_options and updates the annotated_dataframe."""
        self._data_annotation_options = self._get_data_annotation_options(name=name)
        self.annotated_dataframe = self._update_data_annotation(
            **self._data_annotation_options
        )

    @abstractmethod
    def _get_data_annotation_options(self, name: str = "") -> Dict:
        """Implementations of this functions should generate a streamlit interface and return a dictionary of parameters that can be passed to _update_data_annotation as kwargs."""
        pass

    @abstractmethod
    def _update_data_annotation(self, **kwargs) -> pd.DataFrame:
        """Implementations of this function should create the dataframe that can then directly be used by the _update_plot method to update the plot."""
        pass

    def _apply_display_options(self, name: str = ""):
        """Function to get and apply all display options.

        This sets the display_options and parameterizes and updates the plot if plot is the selection made.
        """
        if self._is_plottable is False:
            display_selection_options = [
                self.DISPLAY_SELECTION.RAW_DATAFRAME,
                self.DISPLAY_SELECTION.ANNOTATED_DATAFRAME,
            ]  # string constants
        else:
            display_selection_options = [
                self.DISPLAY_SELECTION.RAW_DATAFRAME,
                self.DISPLAY_SELECTION.ANNOTATED_DATAFRAME,
                self.DISPLAY_SELECTION.PLOT,
            ]  # string constants

        display_selection = st.radio(
            "Select display",
            display_selection_options,
            index=display_selection_options.index(
                st.session_state.get(
                    f"{name}_display_selection", self._display_selection
                )
            ),
            key=f"{name}_display_selection",
        )

        self._display_selection = display_selection
        if display_selection == self.DISPLAY_SELECTION.PLOT:
            self._display_options = self._get_plot_options(name=name)
            self.plot = self._update_plot(**self._display_options)

    @abstractmethod
    def _get_plot_options(self, name: str = "") -> Dict:
        """Implementations of this functions should generate a streamlit interface and return a dictionary of parameters that can be passed to _update_plot as kwargs."""
        pass

    @abstractmethod
    def _update_plot(self, **kwargs) -> Figure:
        """Implementations of this function should use the annotated_dataframe attribute and kwargs to create the plot that can then directly be displayed by the _display_object method."""
        pass

    def _display_object(self) -> None:
        """Function to display the result object"""
        if self._display_selection == self.DISPLAY_SELECTION.PLOT:
            st.plotly_chart(self.plot.update())
        elif self._display_selection == self.DISPLAY_SELECTION.RAW_DATAFRAME:
            st.dataframe(self.dataframe)
        elif self._display_selection == self.DISPLAY_SELECTION.ANNOTATED_DATAFRAME:
            st.dataframe(self.annotated_dataframe)

    def display_object(
        self,
        st_display_column: st.delta_generator.DeltaGenerator,
        data_annotation_editable: bool = False,
        display_editable: bool = False,
        st_widget_column: Optional[st.delta_generator.DeltaGenerator] = None,
        name: str = "",
    ):
        """Function to display the object.
        The function will display the object in the display column and the options in the widget column.
        The boolean flags are intended for controlling behaviour in different sections of the application.

        Args:
            display_column (st.container): The container to display the object.
            data_annotation_editable (bool, optional): If the data_annotation options are editable. Defaults to False.
            display_editable (bool, optional): If the display options are editable. Defaults to False.
            widget_column (Optional[st.container], optional): The container to display the widgets. Defaults to None.
            name: ...

        Raises:
            ValueError: If the widget column container is not provided.
        """
        if data_annotation_editable or display_editable:
            if st_widget_column is None:
                raise ValueError("Widget column container must be provided")
            with st_widget_column:
                if data_annotation_editable:
                    self._apply_data_annotation_options(name=name)
                    if not display_editable and self._is_plottable:
                        self.plot = self._update_plot(**self._display_options)
                if display_editable:
                    self._apply_display_options(name=name)
        with st_display_column:
            self._display_object()

    def _get_standard_layout_options(self, name: str = "") -> Dict:
        """Function to get the standard layout options for the plot.

        This can be used by the _get_plot_options method to get the standard layout options for the plot and then passed as kwargs to the _update_plot > plotting function > Figure.update."""
        return {
            RESULT_PARAMETERS.HEIGHT: st.number_input(
                "Height",
                200,
                1000,
                self._initialize_widget(
                    RESULT_PARAMETERS.HEIGHT, name, self._display_options, 500
                ),
                10,
                key=self._create_temporary_sessionstate_key(
                    RESULT_PARAMETERS.HEIGHT, name
                ),
            ),
            RESULT_PARAMETERS.WIDTH: st.number_input(
                "Width",
                200,
                1000,
                self._initialize_widget(
                    RESULT_PARAMETERS.WIDTH, name, self._display_options, 500
                ),
                10,
                key=self._create_temporary_sessionstate_key(
                    RESULT_PARAMETERS.WIDTH, name
                ),
            ),
            RESULT_PARAMETERS.SHOWLEGEND: st.checkbox(
                "Show legend",
                self._initialize_widget(
                    RESULT_PARAMETERS.SHOWLEGEND, name, self._display_options, False
                ),
                key=self._create_temporary_sessionstate_key(
                    RESULT_PARAMETERS.SHOWLEGEND, name
                ),
            ),
        }

    def _create_temporary_sessionstate_key(
        self,
        parameter_name: str,
        result_id: str,
    ) -> str:
        """Function to create the temporary session state key for a widget.

        Parameters
        ----------
        parameter_name : str
            name for the parameter
        result_id : str
            unique identifier for this instance"""
        return f"tmp_key_{parameter_name}_{result_id}"

    def _initialize_widget(
        self,
        parameter_name: str,
        result_id: str,
        parameter_dictionary: Dict,
        default_value: Any,
    ) -> Any:
        """The behaviour we want to achieve for widgets created in the context of results is the following.

        - be initialized from the parameters stored in the dictionary, hence persistent
        - be changeable by only one click
        - be agnostic to other widgets for the same parameter on the page

        If we just initialize from parameter_dictionary and then set parameter_dictionary in the next line every widget interaction has to happen twice. This is because the first interaction triggers a rerun, without setting the dictionary value first. During this rerun the value is set, but only after the widget is drawn with the old value as default (despite showing the correct value). Only the next rerun sets the widget default to the value from the dictionary. This leads to really odd behaviour. The only way to ensure, that the widget stays in sync is to use the session state, as this get updated before the rerun. To cover the case where the widget is created for the first time on a page we default that to the dictionary. The reason we use temprorary keys that use a unique id, is to avoid session state clutter and be able to display two widget for the same parameter name on the same page (results page).

        Parameters
        ----------
        parameter_name : str
            name for the parameter
        result_id : str
            unique identifier for this instance
        parameter_dictionary : dict
            dictionary to fetch the value from
        default_value : Any
            value to set as default if neither session_state, nor parameter_dictionary have a value stored
        """
        return st.session_state.get(
            self._create_temporary_sessionstate_key(parameter_name, result_id),
            parameter_dictionary.get(
                parameter_name,
                default_value,
            ),
        )


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
    ) -> Tuple[Union[PlotlyObject, pd.DataFrame, ResultObject], Optional[VolcanoPlot]]:
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
        metadata_groups = metadata.columns.to_list()
        custom_group_option = "Custom groups from samples .."

        options = [default_option] + metadata_groups + [custom_group_option]
        grouping_variable = st.selectbox(
            "Grouping variable",
            options=options,
            index=options.index(
                st.session_state.get(
                    ANALYSIS_PARAMETERS.TWOGROUP_COLUMN,
                    default_option if len(metadata_groups) == 0 else metadata_groups[0],
                )
            ),
            key=ANALYSIS_PARAMETERS.TWOGROUP_COLUMN,
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
                key=ANALYSIS_PARAMETERS.TWOGROUP_GROUP1,
            )
            group2 = st.selectbox(
                "Group 2",
                options=list(reversed(unique_values)),
                key=ANALYSIS_PARAMETERS.TWOGROUP_GROUP2,
            )

        else:
            group1 = st.multiselect(
                "Group 1 samples:",
                options=metadata[Cols.SAMPLE].to_list(),
                key=ANALYSIS_PARAMETERS.TWOGROUP_GROUP1 + "multi",
            )

            group2 = st.multiselect(
                "Group 2 samples:",
                options=list(reversed(metadata[Cols.SAMPLE].to_list())),
                key=ANALYSIS_PARAMETERS.TWOGROUP_GROUP2 + "multi",
            )

            intersection_list = list(set(group1).intersection(set(group2)))
            if len(intersection_list) > 0:
                st.warning(
                    "Group 1 and Group 2 contain same samples: "
                    + str(intersection_list)
                )

        self._parameters.update(
            {
                ANALYSIS_PARAMETERS.TWOGROUP_GROUP1: group1,
                ANALYSIS_PARAMETERS.TWOGROUP_GROUP2: group2,
            }
        )
        if column is not None:
            self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_COLUMN] = column

    def _pre_analysis_check(self):
        """Raise if selected groups are different."""
        if (
            self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP1]
            == self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP2]
        ):
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
            group1=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP1],
            group2=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP2],
            column=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_COLUMN],
            method=self._parameters[ANALYSIS_PARAMETERS.DEA_TWOGROUPS_METHOD],
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

        self._parameters.update({ANALYSIS_PARAMETERS.DEA_TWOGROUPS_METHOD: method})

    def _do_analysis(self):
        """Perform T-test analysis."""
        diff_exp_analysis = self._dataset.diff_expression_analysis(
            method=self._parameters[ANALYSIS_PARAMETERS.DEA_TWOGROUPS_METHOD],
            group1=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP1],
            group2=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP2],
            column=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_COLUMN],
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
    """Widget for Differential expression analysis between two groups."""

    def show_widget(self):
        """Show the widget and gather parameters."""
        super().show_widget()

        parameters = {}
        method = st.selectbox(
            "Differential Analysis using:",
            options=["independent t-test", "paired t-test"],
            key=ANALYSIS_PARAMETERS.DEA_TWOGROUPS_METHOD,
        )
        parameters[ANALYSIS_PARAMETERS.DEA_TWOGROUPS_METHOD] = method

        fdr_method = st.selectbox(
            "FDR method",
            options=["fdr_bh", "bonferroni"],
            index=0,
            format_func=lambda x: {
                "fdr_bh": "Benjamini-Hochberg",
                "bonferroni": "Bonferroni",
            }[x],
            key=ANALYSIS_PARAMETERS.DEA_TWOGROUPS_FDR_METHOD,
        )
        parameters[ANALYSIS_PARAMETERS.DEA_TWOGROUPS_FDR_METHOD] = fdr_method

        self._parameters.update(parameters)

    def _do_analysis(self) -> Tuple[ResultObject, None]:
        """Run the differential expression analysis between two groups and return the corresponding results object."""

        test_type = {
            "independent t-test": DeaTestTypes.INDEPENDENT,
            "paired t-test": DeaTestTypes.PAIRED,
        }[self._parameters[ANALYSIS_PARAMETERS.DEA_TWOGROUPS_METHOD]]

        dea = DifferentialExpressionAnalysisTTest(
            self._dataset.mat,
            self._dataset.preprocessing_info[PreprocessingStateKeys.LOG2_TRANSFORMED],
        )
        dea_result = dea.perform(
            test_type=test_type,
            group1=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP1],
            group2=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_GROUP2],
            grouping_column=self._parameters[ANALYSIS_PARAMETERS.TWOGROUP_COLUMN],
            metadata=self._dataset.metadata,
            fdr_method=self._parameters[ANALYSIS_PARAMETERS.DEA_TWOGROUPS_FDR_METHOD],
        )

        return DifferentialExpressionTwoGroupsResult(
            dea_result,
            preprocessing=self._dataset.preprocessing_info,
            method=self._parameters,
            is_plottable=True,
        ), None  # None is for backwards compatibility


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


class DifferentialExpressionTwoGroupsResult(ResultObject):
    """Implementation of the ResultObject for the differential expression analysis between two groups."""

    def _get_data_annotation_options(self, name: str = "") -> Dict:
        """Function to get the data annotation options for the differential expression analysis between two groups.

        Parameters fetched are: qvalue_cutoff, log2fc_cutoff, flip_xaxis.
        """
        return {
            RESULT_PARAMETERS.QVALUE_CUTOFF: st.number_input(
                "Q-value cutoff",
                0.0,
                1.0,
                self._initialize_widget(
                    RESULT_PARAMETERS.QVALUE_CUTOFF,
                    name,
                    self._data_annotation_options,
                    0.05,
                ),
                0.01,
                format="%.2f",
                key=self._create_temporary_sessionstate_key(
                    RESULT_PARAMETERS.QVALUE_CUTOFF, name
                ),
            ),
            RESULT_PARAMETERS.LOG2FC_CUTOFF: st.number_input(
                "Log2FC cutoff",
                0.0,
                10.0,
                self._initialize_widget(
                    RESULT_PARAMETERS.LOG2FC_CUTOFF,
                    name,
                    self._data_annotation_options,
                    1.0,
                ),
                0.1,
                format="%.1f",
                key=self._create_temporary_sessionstate_key(
                    RESULT_PARAMETERS.LOG2FC_CUTOFF, name
                ),
            ),
            RESULT_PARAMETERS.FLIP_XAXIS: st.checkbox(
                "Flip groups",
                self._initialize_widget(
                    RESULT_PARAMETERS.FLIP_XAXIS,
                    name,
                    self._data_annotation_options,
                    False,
                ),
                key=self._create_temporary_sessionstate_key(
                    RESULT_PARAMETERS.FLIP_XAXIS, name
                ),
            ),
        }

    def _update_data_annotation(
        self,
        qvalue_cutoff: float,
        log2fc_cutoff: float,
        flip_xaxis: bool,
    ) -> pd.DataFrame:
        """Function to update the data annotation for the differential expression analysis between two groups.

        Parameters
        ----------
        qvalue_cutoff : float
                The q-value cutoff for the differential expression analysis.
        log2fc_cutoff : float
            The log2 fold change cutoff for the differential expression analysis.
        flip_xaxis : bool
            Whether to flip the x-axis. This determines the new column name for the fold change column, stored in log2name.
        """
        formatted_df = prepare_result_df(
            statistics_results_df=self.dataframe,
            feature_to_repr_map=st.session_state[
                StateKeys.DATASET
            ]._feature_to_repr_map,
            group1=self.method[ANALYSIS_PARAMETERS.TWOGROUP_GROUP1],
            group2=self.method[ANALYSIS_PARAMETERS.TWOGROUP_GROUP2],
            qvalue_cutoff=qvalue_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            flip_xaxis=flip_xaxis,
        )
        return formatted_df

    def _get_plot_options(self, name: str = "") -> Dict:
        """Function to get the plot options for the differential expression analysis between two groups.

        Parameters fetched are: drawlines, label_significant, renderer. Additionally the standard layout options are fetched.
        Note: renderer is important to make sure that pdf and svg downloads of the plot are actually vetorized and not a png set into a vectorized frame."""
        with st.expander("Display options"):
            renderer_options = ["webgl", "svg"]
            return {
                **{
                    RESULT_PARAMETERS.DRAWLINES: st.checkbox(
                        "Draw significance and fold change lines",
                        self._initialize_widget(
                            RESULT_PARAMETERS.DRAWLINES,
                            name,
                            self._display_options,
                            True,
                        ),
                        key=self._create_temporary_sessionstate_key(
                            RESULT_PARAMETERS.DRAWLINES, name
                        ),
                    ),
                    RESULT_PARAMETERS.LABEL_SIGNIFICANT: st.checkbox(
                        "Label significant points",
                        self._initialize_widget(
                            RESULT_PARAMETERS.LABEL_SIGNIFICANT,
                            name,
                            self._display_options,
                            True,
                        ),
                        key=self._create_temporary_sessionstate_key(
                            RESULT_PARAMETERS.LABEL_SIGNIFICANT, name
                        ),
                    ),
                    RESULT_PARAMETERS.RENDERER: st.radio(
                        "Renderer (Choose svg before download to maintain quality.)",
                        renderer_options,
                        index=renderer_options.index(
                            self._initialize_widget(
                                RESULT_PARAMETERS.RENDERER,
                                name,
                                self._display_options,
                                "webgl",
                            )
                        ),
                        key=self._create_temporary_sessionstate_key(
                            RESULT_PARAMETERS.RENDERER, name
                        ),
                    ),
                },
                **self._get_standard_layout_options(name=name),
            }

    def _update_plot(
        self,
        drawlines: bool,
        label_significant: bool,
        renderer: Literal["webgl", "svg"],
        **layout_options,
    ) -> Figure:
        """Function to update the plot for the differential expression analysis between two groups.

        It additionally uses the data_annotation_options to get the qvalue_cutoff, log2fc_cutoff and flip_xaxis parameters and the method to get the group1 and group2 parameters.

        Parameters
        ----------
        drawlines : bool
            Whether to draw the significance and fold change lines.
        label_significant : bool
            Whether to label significant points.
        renderer : Whether to use the webgl (better for web display) or svg (required for proper svg download) rendering engine."""
        return _plot_volcano(
            df_plot=self.annotated_dataframe,
            group1=self.method[ANALYSIS_PARAMETERS.TWOGROUP_GROUP1],
            group2=self.method[ANALYSIS_PARAMETERS.TWOGROUP_GROUP2],
            qvalue_cutoff=self._data_annotation_options[
                RESULT_PARAMETERS.QVALUE_CUTOFF
            ],
            log2fc_cutoff=self._data_annotation_options[
                RESULT_PARAMETERS.LOG2FC_CUTOFF
            ],
            flip_xaxis=self._data_annotation_options[RESULT_PARAMETERS.FLIP_XAXIS],
            drawlines=drawlines,
            label_significant=label_significant,
            renderer=renderer,
            **layout_options,
        )
