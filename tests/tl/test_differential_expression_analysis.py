from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphastats.dataset.keys import Cols
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.tl.differential_expression_analysis import (
    DeaColumns,
    DeaParameters,
    DifferentialExpressionAnalysis,
    DifferentialExpressionAnalysisTTest,
    DifferentialExpressionAnalysisTwoGroups,
)


class TestableDifferentialExpressionAnalysis(DifferentialExpressionAnalysis):
    @staticmethod
    def _allowed_parameters():
        return [DeaParameters.METADATA]

    def _extend_validation(self, metadata: pd.DataFrame, **kwargs):
        if metadata.empty:
            raise ValueError("Non-empty dataframe must be provided")

    def _run_statistical_test(self, **kwargs):
        return self._statistical_test_fun(
            self.mat, metadata=kwargs[DeaParameters.METADATA]
        )

    @staticmethod
    def _statistical_test_fun(mat, metadata):
        return pd.DataFrame(
            [[0.055, 2, 0.009]],
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
            index=["gene1"],
        )


def test_dea_no_abstractmethods():
    with pytest.raises(TypeError):
        DifferentialExpressionAnalysis(None)


def test_dea_perform_success():
    mat = pd.DataFrame(np.zeros((3, 3)))
    dea = TestableDifferentialExpressionAnalysis(mat)
    dea.perform(metadata=pd.DataFrame())
    assert dea.result.equals(
        pd.DataFrame(
            [[0.055, 2, 0.009]],
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
            index=["gene1"],
        )
    )


def test_dea_no_metadata():
    mat = pd.DataFrame(np.zeros((3, 3)))
    dea = TestableDifferentialExpressionAnalysis(mat)
    with pytest.raises(TypeError, match=r"'metadata"):
        dea.perform()


def test_dea_additional_arguments():
    mat = pd.DataFrame(np.zeros((3, 3)))
    dea = TestableDifferentialExpressionAnalysis(mat)
    with pytest.raises(TypeError):
        dea.perform(metadata=pd.DataFrame(np.zeros(3, 3)), additional_argument=1)


def test_dea_metadata_validation():
    mat = pd.DataFrame(np.zeros((3, 3)))
    dea = TestableDifferentialExpressionAnalysis(mat)
    with pytest.raises(ValueError, match="Non-empty dataframe must be provided"):
        dea.perform(metadata=None)


def test_dea_output_validation_static():
    result = pd.DataFrame(
        [[0.055, 2, 0.009]],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
        index=["gene1"],
    )
    DifferentialExpressionAnalysis._validate_output(result)


def test_dea_output_validation_missing_column():
    mat = pd.DataFrame(np.zeros((3, 3)))
    dea = TestableDifferentialExpressionAnalysis(mat)
    dea._statistical_test_fun = lambda x, metadata: pd.DataFrame(
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC], index=["gene1"]
    )
    with pytest.raises(KeyError):
        dea.perform(metadata=pd.DataFrame())


def test_dea_output_validation_none():
    mat = pd.DataFrame(np.zeros((3, 3)))
    dea = TestableDifferentialExpressionAnalysis(mat)
    dea._statistical_test_fun = lambda x, metadata: None
    with pytest.raises(ValueError):
        dea.perform(metadata=pd.DataFrame())


def test_dea_output_validation_pass_additional_columns():
    mat = pd.DataFrame(np.zeros((3, 3)))
    dea = TestableDifferentialExpressionAnalysis(mat)
    dea._statistical_test_fun = lambda x, metadata: pd.DataFrame(
        [[0.055, 2, 0.009, 0.1]],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE, "additional"],
        index=["gene1"],
    )
    dea.perform(metadata=pd.DataFrame())


def test_dea_get_significance_static():
    result = pd.DataFrame(
        [[0.055, 2, 0.009], [0.1, 3, 0.1]],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
        index=["gene1", "gene2"],
    )
    significant = DifferentialExpressionAnalysis.get_significance(result, 0.01)
    assert significant.equals(
        pd.DataFrame(
            [[True], [False]],
            columns=[DeaColumns.SIGNIFICANT],
            index=["gene1", "gene2"],
        )
    )


def test_dea_get_dict_key():
    assert (
        DifferentialExpressionAnalysis.get_dict_key(
            {DeaParameters.METADATA: pd.DataFrame()}
        )
        == """{'metadata': Empty DataFrame
Columns: []
Index: []}"""
    )


class TestableDifferentialExpressionAnalysisTwoGroups(
    DifferentialExpressionAnalysisTwoGroups
):
    @staticmethod
    def _allowed_parameters():
        return [
            DeaParameters.METADATA,
            DeaParameters.GROUP1,
            DeaParameters.GROUP2,
            DeaParameters.GROUPING_COLUMN,
        ]

    def _run_statistical_test(self, **kwargs):
        group1, group2 = self._get_group_members(
            group1=kwargs[DeaParameters.GROUP1],
            group2=kwargs[DeaParameters.GROUP2],
            metadata=kwargs[DeaParameters.METADATA],
            grouping_column=kwargs[DeaParameters.GROUPING_COLUMN],
        )
        return self._statistical_test_fun(self.mat, group1=group1, group2=group2)

    @staticmethod
    def _statistical_test_fun(mat, group1, group2):
        return pd.DataFrame(
            [[0.055, 2, 0.009]],
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
            index=["gene1"],
        )

    valid_parameter_input = {
        DeaParameters.METADATA: pd.DataFrame(
            [
                ["sample1", "group1"],
                ["sample2", "group1"],
                ["sample3", "group2"],
                ["sample4", "group2"],
            ],
            columns=[Cols.SAMPLE, "grouping_column"],
        ),
        DeaParameters.GROUP1: "group1",
        DeaParameters.GROUP2: "group2",
        DeaParameters.GROUPING_COLUMN: "grouping_column",
    }

    valid_data_input = pd.DataFrame(
        [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
        columns=["sample1", "sample2", "sample3", "sample4"],
        index=["gene1", "gene2", "gene3"],
    ).T


def test_dea_two_groups_no_abstractmethods():
    with pytest.raises(TypeError):
        DifferentialExpressionAnalysisTwoGroups(None)


def test_dea_two_groups_perform_success():
    mat = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(mat)
    dea.perform(**TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input)
    assert dea.result.equals(
        pd.DataFrame(
            [[0.055, 2, 0.009]],
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
            index=["gene1"],
        )
    )


def test_dea_two_groups_validation_missing_sample():
    mat = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input.drop(
        index="sample1"
    )
    dea = TestableDifferentialExpressionAnalysisTwoGroups(mat)
    with pytest.raises(KeyError):
        dea._validate_input(
            **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input
        )


@patch(
    "alphastats.tl.differential_expression_analysis.DifferentialExpressionAnalysisTwoGroups._get_group_members"
)
def test_dea_two_groups_validation_calls_get_groups(mock_get_group_members):
    mat = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(mat)
    mock_get_group_members.return_value = ["sample1", "sample2"], ["sample3", "sample4"]
    dea._validate_input(
        **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input
    )
    mock_get_group_members.assert_called_once()


def test_dea_two_groups_get_group_members_from_metadata():
    group1, group2 = DifferentialExpressionAnalysisTwoGroups._get_group_members(
        **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input
    )
    assert group1 == ["sample1", "sample2"]
    assert group2 == ["sample3", "sample4"]


def test_dea_two_groups_get_group_members_from_lists():
    group1, group2 = DifferentialExpressionAnalysisTwoGroups._get_group_members(
        group1=["sample1", "sample2"], group2=["sample3", "sample4"]
    )
    assert group1 == ["sample1", "sample2"]
    assert group2 == ["sample3", "sample4"]


def test_dea_two_groups_get_group_members_both_grouping_methods():
    """Test custom error if both grouping methods are parameterized."""
    with pytest.raises(
        TypeError,
        match=r"If grouping_column is provided.*",
    ):
        DifferentialExpressionAnalysisTwoGroups._get_group_members(
            **{
                DeaParameters.GROUPING_COLUMN: "grouping_column",
                DeaParameters.GROUP1: ["sample1", "sample2"],
                DeaParameters.GROUP2: ["sample3", "sample4"],
            }
        )


def test_dea_two_groups_missing_get_group_members_grouping_column():
    """Test custom error if expected kwarg grouping column is missing."""
    with pytest.raises(TypeError, match=r"If grouping_column is not provided.*"):
        DifferentialExpressionAnalysisTwoGroups._get_group_members(
            **{
                DeaParameters.GROUP1: "group1",
                DeaParameters.GROUP2: "group2",
                DeaParameters.METADATA: pd.DataFrame([[1, 2]]),
            }
        )


def test_dea_two_groups_get_group_members_missing_metadata():
    """Test custom error if expected kwarg metadata is missing."""
    with pytest.raises(TypeError, match=r"'metadata'"):
        DifferentialExpressionAnalysisTwoGroups._get_group_members(
            **{
                DeaParameters.GROUPING_COLUMN: "grouping_column",
                DeaParameters.GROUP1: "group1",
                DeaParameters.GROUP2: "group2",
            }
        )


def test_dea_ttest_perform_runs():
    mat = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(mat)
    dea.perform(
        **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input,
        **{
            DeaParameters.TEST_FUN: "independent",
            DeaParameters.FDR_METHOD: "fdr_bh",
            PreprocessingStateKeys.LOG2_TRANSFORMED: True,
        },
    )
    assert dea.result.shape == (3, 3)


@patch("pandas.DataFrame.transform")
def test_dea_ttest_runs_log(mock_transform):
    mat = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(mat)
    mock_transform.return_value = mat.T
    result = dea._run_statistical_test(
        **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input,
        **{
            DeaParameters.TEST_FUN: "independent",
            DeaParameters.FDR_METHOD: "fdr_bh",
            PreprocessingStateKeys.LOG2_TRANSFORMED: False,
        },
    )
    assert result.shape == (3, 3)
    mock_transform.assert_called_once_with(np.log2)


def test_dea_ttest_validation_wrong_stats_method():
    mat = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(mat)
    with pytest.raises(
        ValueError,
        match="test_fun must be either 'independent' for scipy.stats.ttest_ind or 'paired' for scipy.stats.ttest_rel.",
    ):
        dea._validate_input(
            **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input,
            **{
                DeaParameters.TEST_FUN: "not defined",
                DeaParameters.FDR_METHOD: "fdr_bh",
                PreprocessingStateKeys.LOG2_TRANSFORMED: True,
            },
        )


def test_dea_ttest_validation_wrong_fdr_method():
    mat = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(mat)
    with pytest.raises(
        ValueError, match="fdr_method must be one of 'fdr_bh', 'bonferroni'."
    ):
        dea._validate_input(
            **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input,
            **{
                DeaParameters.TEST_FUN: "independent",
                DeaParameters.FDR_METHOD: "unknown",
                PreprocessingStateKeys.LOG2_TRANSFORMED: True,
            },
        )
