from streamlit.testing.v1 import AppTest


def MinimalDifferentialExpressionTwoGroupsResult(
    display_editable=True, data_annotation_editable=True
) -> None:
    import pandas as pd
    import streamlit as st

    from alphastats.dataset.keys import Cols
    from alphastats.gui.utils.result import DifferentialExpressionTwoGroupsResult
    from alphastats.tl.differential_expression_analysis import DeaColumns

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

    c1, c2 = st.columns((1, 2))
    if "a" not in st.session_state:
        st.session_state["a"] = DifferentialExpressionTwoGroupsResult(
            dataframe=valid_statistics_results,
            preprocessing={},
            method=valid_method,
            is_plottable=True,
            feature_to_repr_map=valid_feature_to_repr_map,
        )
    (
        st.session_state["a"].display_object(
            c2,
            c1,
            name="",
            data_annotation_editable=data_annotation_editable,
            display_editable=display_editable,
        ),
    )


def test_DifferentialExpressionTwoGroupsResult():
    at = AppTest.from_function(
        MinimalDifferentialExpressionTwoGroupsResult,
        default_timeout=200,
        kwargs={"data_annotation_editable": True, "display_editable": True},
    )
    at.run()

    assert at.session_state["a"].plot is not None
