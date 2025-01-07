import pandas as pd
import streamlit as st

from alphastats.dataset.keys import Cols
from alphastats.gui.utils.ui_helper import StateKeys
from alphastats.pl.volcano import DifferentialExpressionTwoGroupsResult
from alphastats.tl.differential_expression_analysis import DeaColumns

show_figures = False

valid_statistics_results = pd.DataFrame(
    {
        DeaColumns.LOG2FC: [0.5, 1.0, 1.5, -2.0, 3.0, 0],
        DeaColumns.QVALUE: [0.1, 0.05, 0.06, 0.001, 0.002, 0.02],
        DeaColumns.PVALUE: [0.1, 0.05, 0.06, 0.001, 0.002, 0.02],
    },
    index=["f1", "f2", "f3", "f4", "f5", "f6"],
)
valid_statistics_results.index.name = Cols.INDEX
valid_feature_to_repr_map = {
    "f1": "gene1",
    "f2": "feature2;lorem ipsum",
    "f3": "gene3",
    "f4": "gene4",
    "f5": "gene5",
    "f6": "gene6",
}
valid_method = {"group1": "mutant", "group2": "ctrl"}


class DummyDataset:
    def __init__(self):
        self.feature_to_repr_map = valid_feature_to_repr_map


def test_DifferentialExpressionTwoGroupsResult():
    c1, c2 = st.columns((1, 2))
    st.session_state[StateKeys.DATASET] = DummyDataset()
    result = DifferentialExpressionTwoGroupsResult(
        dataframe=valid_statistics_results,
        preprocessing={},
        method=valid_method,
    )

    result.display_object(c2, True, True, c1)
