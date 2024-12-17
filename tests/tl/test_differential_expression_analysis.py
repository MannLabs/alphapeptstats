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

    def _extend_validation(self, parameters):
        if parameters[DeaParameters.METADATA] is None:
            raise ValueError("Metadata must be provided")

    def _run_statistical_test(self, **kwargs):
        return self._statistical_test_fun(
            self.input_data, metadata=kwargs[DeaParameters.METADATA]
        )

    @staticmethod
    def _statistical_test_fun(input_data, metadata):
        return pd.DataFrame(
            [[0.055, 2, 0.009]],
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
            index=["gene1"],
        )


def test_dea_no_abstractmethods():
    with pytest.raises(TypeError):
        DifferentialExpressionAnalysis(None)


def test_dea_perform_success():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    dea.perform(metadata=pd.DataFrame())
    assert dea.result.equals(
        pd.DataFrame(
            [[0.055, 2, 0.009]],
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
            index=["gene1"],
        )
    )


def test_dea_data_none():
    input_data = None
    dea = TestableDifferentialExpressionAnalysis(input_data)
    with pytest.raises(ValueError, match="No input data was provided."):
        dea.perform()


def test_dea_parameters_none():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    with pytest.raises(ValueError, match="No parameters were provided."):
        dea._validate_input(None)


def test_dea_no_metadata():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    with pytest.raises(KeyError):
        dea.perform()


def test_dea_additional_arguments():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    with pytest.raises(ValueError):
        dea.perform(metadata=pd.DataFrame(), additional_argument=1)


def test_dea_metadata_validation():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    with pytest.raises(ValueError, match="Metadata must be provided"):
        dea.perform(metadata=None)


def test_dea_output_validation_static():
    result = pd.DataFrame(
        [[0.055, 2, 0.009]],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
        index=["gene1"],
    )
    DifferentialExpressionAnalysis._validate_output(result)


def test_dea_output_validation_missing_column():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    dea._statistical_test_fun = lambda x, metadata: pd.DataFrame(
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC], index=["gene1"]
    )
    with pytest.raises(KeyError):
        dea.perform(metadata=pd.DataFrame())


def test_dea_output_validation_none():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    dea._statistical_test_fun = lambda x, metadata: None
    with pytest.raises(ValueError):
        dea.perform(metadata=pd.DataFrame())


def test_dea_output_validation_pass_additional_columns():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    dea._statistical_test_fun = lambda x, metadata: pd.DataFrame(
        [[0.055, 2, 0.009, 0.1]],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE, "additional"],
        index=["gene1"],
    )
    dea.perform(metadata=pd.DataFrame())


def test_dea_get_significance_from_instance():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    dea._statistical_test_fun = lambda x, metadata: pd.DataFrame(
        [[0.055, 2, 0.009], [0.1, 3, 0.1]],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
        index=["gene1", "gene2"],
    )
    dea.perform(metadata=pd.DataFrame())
    significant = dea.get_significance(dea.result, 0.01)
    assert significant.equals(
        pd.DataFrame(
            [[True], [False]],
            columns=[DeaColumns.SIGNIFICANT],
            index=["gene1", "gene2"],
        )
    )


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


def test_dea_get_dict_key_from_instance():
    input_data = pd.DataFrame()
    dea = TestableDifferentialExpressionAnalysis(input_data)
    assert (
        dea.get_dict_key({DeaParameters.METADATA: pd.DataFrame()})
        == """{'metadata': Empty DataFrame
Columns: []
Index: []}"""
    )


def test_dea_get_dict_key_static():
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
        group1, group2 = self._get_group_members(kwargs)
        return self._statistical_test_fun(self.input_data, group1=group1, group2=group2)

    @staticmethod
    def _statistical_test_fun(input_data, group1, group2):
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
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    dea.perform(**TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input)
    assert dea.result.equals(
        pd.DataFrame(
            [[0.055, 2, 0.009]],
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
            index=["gene1"],
        )
    )


def test_dea_two_groups_validate_missing_sample_metadata():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input.drop(
        index="sample1"
    )
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    with pytest.raises(KeyError):
        dea.perform(
            **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input
        )


def test_dea_two_groups_validate_missing_sample_direct_grouping():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input.drop(
        index="sample1"
    )
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    with pytest.raises(KeyError):
        dea.perform(
            **{"group1": ["sample1", "sample2"], "group2": ["sample3", "sample4"]}
        )


def test_dea_two_groups_missing_group1():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    with pytest.raises(KeyError):
        dea.perform(**{DeaParameters.GROUP2: ["sample3", "sample4"]})


def test_dea_two_groups_missing_group2():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    with pytest.raises(KeyError):
        dea.perform(**{DeaParameters.GROUP1: ["sample1", "sample2"]})


def test_dea_two_groups_both_grouping_methods():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    with pytest.raises(
        ValueError,
        match="Please provide either a list of columns OR the grouping column, not both.",
    ):
        dea.perform(
            **{
                DeaParameters.GROUPING_COLUMN: "grouping_column",
                DeaParameters.GROUP1: ["sample1", "sample2"],
                DeaParameters.GROUP2: ["sample3", "sample4"],
            }
        )


def test_dea_two_groups_missing_grouping_column():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    with pytest.raises(
        ValueError, match=f"Parameter {DeaParameters.GROUPING_COLUMN} is missing."
    ):
        dea.perform(
            **{
                DeaParameters.GROUP1: "group1",
                DeaParameters.GROUP2: "group2",
                DeaParameters.METADATA: pd.DataFrame([[1, 2]]),
            }
        )


def test_dea_two_groups_missing_metadata():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = TestableDifferentialExpressionAnalysisTwoGroups(input_data)
    with pytest.raises(
        ValueError, match=f"Parameter {DeaParameters.METADATA} is missing."
    ):
        dea.perform(
            **{
                DeaParameters.GROUPING_COLUMN: "grouping_column",
                DeaParameters.GROUP1: "group1",
                DeaParameters.GROUP2: "group2",
            }
        )


def test_dea_two_groups_get_group_members_metadata():
    group1, group2 = DifferentialExpressionAnalysisTwoGroups._get_group_members(
        TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input
    )
    assert group1 == ["sample1", "sample2"]
    assert group2 == ["sample3", "sample4"]


def test_dea_two_groups_get_group_members_direct_grouping():
    group1, group2 = DifferentialExpressionAnalysisTwoGroups._get_group_members(
        {"group1": ["sample1", "sample2"], "group2": ["sample3", "sample4"]}
    )
    assert group1 == ["sample1", "sample2"]
    assert group2 == ["sample3", "sample4"]


def test_dea_ttest_perform_runs():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(input_data)
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
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(input_data)
    mock_transform.return_value = input_data.T
    dea.perform(
        **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input,
        **{
            DeaParameters.TEST_FUN: "independent",
            DeaParameters.FDR_METHOD: "fdr_bh",
            PreprocessingStateKeys.LOG2_TRANSFORMED: False,
        },
    )
    assert dea.result.shape == (3, 3)
    mock_transform.assert_called_once_with(np.log2)


def test_dea_ttest_validate_wrong_stats_method():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(input_data)
    with pytest.raises(
        ValueError,
        match="test_fun must be either 'independent' for scipy.stats.ttest_ind or 'paired' for scipy.stats.ttest_rel.",
    ):
        dea.perform(
            **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input,
            **{
                DeaParameters.TEST_FUN: "not defined",
                DeaParameters.FDR_METHOD: "fdr_bh",
                PreprocessingStateKeys.LOG2_TRANSFORMED: True,
            },
        )


def test_dea_ttest_validate_wrong_fdr_method():
    input_data = TestableDifferentialExpressionAnalysisTwoGroups.valid_data_input
    dea = DifferentialExpressionAnalysisTTest(input_data)
    with pytest.raises(
        ValueError, match="fdr_method must be one of 'fdr_bh', 'bonferroni'."
    ):
        dea.perform(
            **TestableDifferentialExpressionAnalysisTwoGroups.valid_parameter_input,
            **{
                DeaParameters.TEST_FUN: "independent",
                DeaParameters.FDR_METHOD: "unknown",
                PreprocessingStateKeys.LOG2_TRANSFORMED: True,
            },
        )
