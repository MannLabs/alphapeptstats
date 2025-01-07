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
        self.dataframe = self._update_data_annotation(**self.data_annotation_options)

    @abstractmethod
    def _get_data_annotation_options(self) -> Dict:
        pass

    @abstractmethod
    def _update_data_annotation(self, **kwargs) -> pd.DataFrame:
        pass

    def _apply_display_options(self):
        """Funciton to get all display options for the analysis object"""
        if self.plottable is False:
            return
        display_selection = st.radio("Select display", ["Plot", "Dataframe"])
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
            st.plotly_chart(self.plot.update_layout(plot_bgcolor="white"))
        elif self.display_selection == "Dataframe":
            st.dataframe(self.dataframe)

    def display_object(
        self,
        display_column: st.container,
        significance_editable: bool = False,
        display_editable: bool = False,
        widget_column: Optional[st.container] = None,
    ):
        """Function to display the object"""
        if significance_editable or display_editable:
            if widget_column is None:
                raise ValueError("Widget column container must be provided")
            with widget_column:
                if significance_editable:
                    self._apply_data_annotation_options()
                if display_editable:
                    self._apply_display_options()
        with display_column:
            self._display_object()

    @staticmethod
    def get_standard_layout_options():
        return {
            "height": st.number_input("Height", 200, 1000, 500, 10),
            "width": st.number_input("Width", 200, 1000, 500, 10),
            "showlegend": st.checkbox("Show legend", True),
        }
