from multiprocessing.util import ForkAwareThreadLock
from operator import sub
from random import sample
import streamlit as st
import os
import pandas as pd
import datetime
import yaml
from typing import Union, Tuple
import logging
from .import_data import ImportData
from .preprocessing import Preprocessing
from .plotting import Plotting


class user_interface(ImportData, Preprocessing, Plotting):
    def __init__(self):
        self.dataset = None
        self.software = None
        self.metadata_columns = None
        self.plotting_options = None
        self.loader = None

    def overview(self):
        if self.dataset is None:
            st.write("Load data first.")
            return
        self.dataset.preprocess_print_info()

    def get_column_names_metadata(self):
        self.metadata_columns = self.dataset.metadata.columns.to_list()

    def get_unique_values_from_column(self, column):
        unique_values = self.dataset.metadata[column].unique()
        return unique_values