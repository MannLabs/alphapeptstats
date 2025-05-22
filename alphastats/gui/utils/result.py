"""Module for the ResultComponent class and its implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import streamlit as st

from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.ui_helper import (
    AnalysisParameters,
    ResultParameters,
)
from alphastats.pl.volcano import _plot_volcano, prepare_result_df

if TYPE_CHECKING:
    import pandas as pd
    from plotly.graph_objects import Figure


class ResultComponent(ABC):  # move to new file
    """Base class for providing the UI for inspecting and parameterizing the analysis based of statistical results.

    The intended use is that in a first step data can be annotated e.g. based on significance cutoffs and then plotted, e.g. applying cosmetic preferences like lines and colors.

    The display function can parameterized to restrict editing options to 'freeze' the result.
    """

    class DisplaySelection(metaclass=ConstantsClass):
        """Keys for the display options, the order determines order in UI."""

        PLOT = "Plot"
        RAW_DATAFRAME = "Raw Dataframe"
        ANNOTATED_DATAFRAME = "Annotated Dataframe"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessing: dict,
        method: dict,
        feature_to_repr_map: dict,
        *,
        is_plottable: bool,
    ) -> None:
        """Initialize the ResultObject with a dataframe.

        Args:
        ----
            dataframe (pd.DataFrame): The dataframe to be used for the analysis.
            is_plottable (bool): If the analysis is plottable.
            preprocessing (Dict): The preprocessing information.
            method (Dict): The method used for the analysis.
            feature_to_repr_map (Dict): The feature to representation mapping.

        Raises:
        ------
            ValueError: If no dataframe is provided.

        """
        self.dataframe = dataframe
        self.annotated_dataframe = dataframe
        self._is_plottable = is_plottable
        self.preprocessing = preprocessing
        self.method = method
        self.feature_to_repr_map = feature_to_repr_map

        self.plot: Figure | None = None
        self._data_annotation_options = {}
        self._display_options = {}
        self._display_selection = (
            self.DisplaySelection.PLOT
            if self._is_plottable
            else self.DisplaySelection.RAW_DATAFRAME
        )

        self._key = str(datetime.now())  # noqa: DTZ005

    def _apply_data_annotation_options(self, name: str = "") -> None:
        """Get and apply all options for data annotation.

        This sets the data_annotation_options and updates the annotated_dataframe.
        """
        self._data_annotation_options = self._get_data_annotation_options(name=name)
        self.annotated_dataframe = self._annotate_data(**self._data_annotation_options)

    @abstractmethod
    def _get_data_annotation_options(self, name: str = "") -> dict:
        """Implementations of this functions should generate a streamlit interface and return a dictionary of parameters that can be passed to _update_data_annotation as kwargs."""

    @abstractmethod
    def _annotate_data(self, **kwargs) -> pd.DataFrame:
        """Implementations of this function should create the dataframe that can then directly be used by the _update_plot method to update the plot."""

    def _apply_display_options(self, name: str = "") -> None:
        """Get and apply all display options.

        This sets the display_options and parameterizes and updates the plot if plot is the selection made.
        """
        if self._is_plottable is False:
            display_selection_options = [
                self.DisplaySelection.RAW_DATAFRAME,
                self.DisplaySelection.ANNOTATED_DATAFRAME,
            ]
        else:
            display_selection_options = [
                self.DisplaySelection.RAW_DATAFRAME,
                self.DisplaySelection.ANNOTATED_DATAFRAME,
                self.DisplaySelection.PLOT,
            ]

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
        if display_selection == self.DisplaySelection.PLOT:
            self._display_options = self._get_plot_options(name=name)
            self.plot = self._create_plot(**self._display_options)

    @abstractmethod
    def _get_plot_options(self, name: str = "") -> dict:
        """Implementations of this functions should generate a streamlit interface and return a dictionary of parameters that can be passed to _update_plot as kwargs."""

    @abstractmethod
    def _create_plot(self, **kwargs) -> Figure:
        """Implementations of this function should use the annotated_dataframe attribute and kwargs to create the plot that can then directly be displayed by the _display_object method."""

    def _display_object(self) -> None:
        """Function to display the result object."""
        if self._display_selection == self.DisplaySelection.PLOT:
            st.plotly_chart(self.plot.update(), key=f"{self._key}_plot")
        elif self._display_selection == self.DisplaySelection.RAW_DATAFRAME:
            st.dataframe(self.dataframe, key=f"{self._key}_df")
        elif self._display_selection == self.DisplaySelection.ANNOTATED_DATAFRAME:
            st.dataframe(self.annotated_dataframe, key=f"{self._key}_ann_df")

    def display_object(
        self,
        st_display_column: st.delta_generator.DeltaGenerator,
        st_widget_column: st.delta_generator.DeltaGenerator | None = None,
        name: str = "",
        *,
        data_annotation_editable: bool = False,
        display_editable: bool = False,
    ) -> None:
        """Display the object, optionally with widgets.

        The function will display the object in the display column and the options in the widget column.
        The boolean flags are intended for controlling behaviour in different sections of the application.

        Parameters
        ----------
        st_display_column : st.delta_generator.DeltaGenerator
            The container to display the object.
        st_widget_column : st.delta_generator.DeltaGenerator | None
            The container to display the widgets. Defaults to None.
        name : str
            The name of the object. Defaults to "".
        data_annotation_editable : bool
            Whether the data_annotation options are editable. Defaults to False.
        display_editable : bool
            Whether the display options are editable. Defaults to False.

        Raises
        ------
            ValueError: If the widget column container is not provided.

        """
        if data_annotation_editable or display_editable:
            if st_widget_column is None:
                raise ValueError("Widget column container must be provided")
            with st_widget_column:
                if data_annotation_editable:
                    self._apply_data_annotation_options(name=name)
                    if not display_editable and self._is_plottable:
                        self.plot = self._create_plot(**self._display_options)
                if display_editable:
                    self._apply_display_options(name=name)
        with st_display_column:
            self._display_object()

    def _get_standard_layout_options(self, name: str = "") -> dict:
        """Get the standard layout options for the plot.

        This can be used by the _get_plot_options method to get the standard layout options for the plot and then passed as kwargs to the _update_plot > plotting function > Figure.update.
        """
        return {
            ResultParameters.HEIGHT: st.number_input(
                "Height",
                min_value=200,
                max_value=1000,
                value=self._initialize_widget(
                    ResultParameters.HEIGHT, name, self._display_options, 500
                ),
                step=10,
                key=self._create_temporary_session_state_key(
                    ResultParameters.HEIGHT, name
                ),
            ),
            ResultParameters.WIDTH: st.number_input(
                "Width",
                min_value=200,
                max_value=1000,
                value=self._initialize_widget(
                    ResultParameters.WIDTH, name, self._display_options, 500
                ),
                step=10,
                key=self._create_temporary_session_state_key(
                    ResultParameters.WIDTH, name
                ),
            ),
            ResultParameters.SHOWLEGEND: st.checkbox(
                "Show legend",
                value=self._initialize_widget(
                    ResultParameters.SHOWLEGEND,
                    name,
                    self._display_options,
                    default=False,
                ),
                key=self._create_temporary_session_state_key(
                    ResultParameters.SHOWLEGEND, name
                ),
            ),
        }

    def _create_temporary_session_state_key(
        self,
        parameter_name: str,
        result_id: str,
    ) -> str:
        """Create the temporary session state key for a widget.

        Parameters
        ----------
        parameter_name : str
            name for the parameter
        result_id : str
            unique identifier for this instance

        """
        return f"tmp_key_{parameter_name}_{result_id}"

    def _initialize_widget(
        self,
        parameter_name: str,
        result_id: str,
        parameter_dictionary: dict,
        default: bool | float | str | list,
    ) -> bool | float | str | list:
        """Return the value for a widget based on session_state, stored values or a default.

        The behaviour we want to achieve for widgets created in the context of results is the following.

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
        default : bool | float | str | list
            value to set as default if neither session_state, nor parameter_dictionary have a value stored

        """
        return st.session_state.get(
            self._create_temporary_session_state_key(parameter_name, result_id),
            parameter_dictionary.get(
                parameter_name,
                default,
            ),
        )


class DifferentialExpressionTwoGroupsResult(ResultComponent):
    """Implementation of the ResultObject for the differential expression analysis between two groups."""

    def _get_data_annotation_options(self, name: str = "") -> dict:
        """Function to get the data annotation options for the differential expression analysis between two groups.

        Parameters fetched are: qvalue_cutoff, log2fc_cutoff, flip_xaxis.
        """
        return {
            ResultParameters.QVALUE_CUTOFF: st.number_input(
                "Q-value cutoff",
                min_value=0.0,
                max_value=1.0,
                value=self._initialize_widget(
                    ResultParameters.QVALUE_CUTOFF,
                    name,
                    self._data_annotation_options,
                    0.05,
                ),
                step=0.01,
                format="%.2f",
                key=self._create_temporary_session_state_key(
                    ResultParameters.QVALUE_CUTOFF, name
                ),
            ),
            ResultParameters.LOG2FC_CUTOFF: st.number_input(
                "Log2FC cutoff",
                min_value=0.0,
                max_value=10.0,
                value=self._initialize_widget(
                    ResultParameters.LOG2FC_CUTOFF,
                    name,
                    self._data_annotation_options,
                    1.0,
                ),
                step=0.1,
                format="%.1f",
                key=self._create_temporary_session_state_key(
                    ResultParameters.LOG2FC_CUTOFF, name
                ),
            ),
            ResultParameters.FLIP_XAXIS: st.checkbox(
                "Flip groups",
                value=self._initialize_widget(
                    ResultParameters.FLIP_XAXIS,
                    name,
                    self._data_annotation_options,
                    default=False,
                ),
                key=self._create_temporary_session_state_key(
                    ResultParameters.FLIP_XAXIS, name
                ),
            ),
        }

    def _annotate_data(
        self,
        qvalue_cutoff: float,
        log2fc_cutoff: float,
        *,
        flip_xaxis: bool,
    ) -> pd.DataFrame:
        """Update the data annotation for the differential expression analysis between two groups.

        Parameters
        ----------
        qvalue_cutoff : float
                The q-value cutoff for the differential expression analysis.
        log2fc_cutoff : float
            The log2 fold change cutoff for the differential expression analysis.
        flip_xaxis : bool
            Whether to flip the x-axis. This determines the new column name for the fold change column, stored in log2name.

        """
        return prepare_result_df(
            statistics_results_df=self.dataframe,
            feature_to_repr_map=self.feature_to_repr_map,
            group1=self.method[AnalysisParameters.TWOGROUP_GROUP1],
            group2=self.method[AnalysisParameters.TWOGROUP_GROUP2],
            qvalue_cutoff=qvalue_cutoff,
            log2fc_cutoff=log2fc_cutoff,
            flip_xaxis=flip_xaxis,
        )

    def _get_plot_options(self, name: str = "") -> dict:
        """Get the plot options for the differential expression analysis between two groups.

        Parameters fetched are: draw_lines, label_significant, renderer. Additionally the standard layout options are fetched.
        Note: renderer is important to make sure that pdf and svg downloads of the plot are actually vetorized and not a png set into a vectorized frame.
        """
        with st.expander("Display options"):
            renderer_options = ["webgl", "svg"]
            return {
                ResultParameters.DRAW_LINES: st.checkbox(
                    "Draw significance and fold change lines",
                    value=self._initialize_widget(
                        ResultParameters.DRAW_LINES,
                        name,
                        self._display_options,
                        default=True,
                    ),
                    key=self._create_temporary_session_state_key(
                        ResultParameters.DRAW_LINES, name
                    ),
                ),
                ResultParameters.LABEL_SIGNIFICANT: st.checkbox(
                    "Label significant points",
                    value=self._initialize_widget(
                        ResultParameters.LABEL_SIGNIFICANT,
                        name,
                        self._display_options,
                        default=True,
                    ),
                    key=self._create_temporary_session_state_key(
                        ResultParameters.LABEL_SIGNIFICANT, name
                    ),
                ),
                ResultParameters.RENDERER: st.radio(
                    "Renderer (Choose svg before download to maintain quality.)",
                    options=renderer_options,
                    index=renderer_options.index(
                        self._initialize_widget(
                            ResultParameters.RENDERER,
                            name,
                            self._display_options,
                            "webgl",
                        )
                    ),
                    key=self._create_temporary_session_state_key(
                        ResultParameters.RENDERER, name
                    ),
                ),
                **self._get_standard_layout_options(name=name),
            }

    def _create_plot(
        self,
        renderer: Literal["webgl", "svg"],
        *,
        draw_lines: bool,
        label_significant: bool,
        **layout_options,
    ) -> Figure:
        """Update the plot for the differential expression analysis between two groups.

        It additionally uses the data_annotation_options to get the qvalue_cutoff, log2fc_cutoff and flip_xaxis parameters and the method to get the group1 and group2 parameters.

        Parameters
        ----------
        renderer : str
            Whether to use the webgl (better for web display) or svg (required for proper svg download) rendering engine.
        draw_lines : bool
            Whether to draw the significance and fold change lines.
        label_significant : bool
            Whether to label significant points.
        layout_options : dict
            The layout options for the plot, directly passed to Figure.update_layout.

        """
        return _plot_volcano(
            df_plot=self.annotated_dataframe,
            group1=self.method[AnalysisParameters.TWOGROUP_GROUP1],
            group2=self.method[AnalysisParameters.TWOGROUP_GROUP2],
            qvalue_cutoff=self._data_annotation_options[ResultParameters.QVALUE_CUTOFF],
            log2fc_cutoff=self._data_annotation_options[ResultParameters.LOG2FC_CUTOFF],
            flip_xaxis=self._data_annotation_options[ResultParameters.FLIP_XAXIS],
            draw_lines=draw_lines,
            label_significant=label_significant,
            renderer=renderer,
            **layout_options,
        )
