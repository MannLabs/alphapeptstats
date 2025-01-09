from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

import pandas as pd
import streamlit as st
from plotly.graph_objects import Figure

from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.ui_helper import (
    ANALYSIS_PARAMETERS,
    RESULT_PARAMETERS,
    StateKeys,
)
from alphastats.pl.volcano import _plot_volcano, prepare_result_df


class ResultComponent(ABC):  # move to new file
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


class DifferentialExpressionTwoGroupsResult(ResultComponent):
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
