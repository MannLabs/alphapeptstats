from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
    def _update_plot(self, **kwargs) -> go.Figure:
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
