from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphastats.dataset.keys import Cols
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.tl.differential_expression_analysis import (
    DeaColumns,
    DeaParameters,
    DeaTestTypes,
    DifferentialExpressionAnalysis,
    DifferentialExpressionAnalysisTTest,
    DifferentialExpressionAnalysisTwoGroups,
)

valid_dea_output = pd.DataFrame(
    [[0.055, 2, 0.009], [0.1, 0.1, 0.1], [0.1, 0.1, np.nan]],
    columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
    index=["gene1", "gene2", "gene3"],
)


class TestableDifferentialExpressionAnalysis(DifferentialExpressionAnalysis):
    """Testable implementation of DifferentialExpressionAnalysis."""

    @staticmethod
    def _allowed_parameters():
        return [DeaParameters.METADATA]

    def _extend_validation(self, metadata: pd.DataFrame, **kwargs):
        if metadata.empty:
            raise ValueError("Non-empty dataframe must be provided")

    def _perform(self, **kwargs):
        return self._run_statistical_test(
            self.mat, metadata=kwargs[DeaParameters.METADATA]
        )

    @staticmethod
    def _run_statistical_test(mat, metadata):
        return valid_dea_output


class Test_DifferentialExpressionAnalysis:
    """Tests for instances of DifferentialExpressionAnalysis."""

    def setup_method(self):
        """Setup testable DifferentialExpressionAnalysis instance."""
        mat = pd.DataFrame(np.zeros((3, 3)))
        self.dea = TestableDifferentialExpressionAnalysis(mat)

    def teardown_method(self):
        """Teardown testable DifferentialExpressionAnalysis instance."""
        del self.dea

    @patch(
        "test_differential_expression_analysis.TestableDifferentialExpressionAnalysis._run_statistical_test"
    )
    def test_dea_perform_calls_test(self, mock_run_statistical_test):
        """Perform method calls test method."""
        mock_run_statistical_test.return_value = valid_dea_output
        self.dea.perform(metadata=pd.DataFrame(np.zeros((3, 3))))
        assert mock_run_statistical_test.called_once()

    def test_dea_no_metadata(self):
        """Type error raised if metadata is not provided."""
        with pytest.raises(TypeError, match=r"'metadata'"):
            self.dea.perform()

    def test_dea_additional_arguments(self):
        """Type error raised if additional arguments are provided."""
        with pytest.raises(TypeError):
            self.dea.perform(
                metadata=pd.DataFrame(np.zeros(3, 3)), additional_argument=1
            )

    def test_dea_metadata_validation(self):
        """Validation error raised if metadata is empty."""
        with pytest.raises(ValueError, match="Non-empty dataframe must be provided"):
            self.dea.perform(metadata=pd.DataFrame())

    def test_dea_output_validation_missing_column(self):
        """Key error raised if output is missing required columns."""
        self.dea._run_statistical_test = lambda x, metadata: pd.DataFrame(
            columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC], index=["gene1"]
        )
        with pytest.raises(KeyError, match=rf"'{DeaColumns.QVALUE}'"):
            self.dea.perform(metadata=pd.DataFrame(np.zeros((3, 3))))

    @patch(
        "test_differential_expression_analysis.TestableDifferentialExpressionAnalysis._run_statistical_test"
    )
    def test_dea_output_validation_none(self, mock_run_statistical_test):
        """Value error raised if output is empty."""
        mock_run_statistical_test.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match=r"empty"):
            self.dea.perform(metadata=pd.DataFrame(np.zeros((3, 3))))

    @patch(
        "test_differential_expression_analysis.TestableDifferentialExpressionAnalysis._run_statistical_test"
    )
    def test_dea_output_validation_pass_additional_columns(
        self, mock_run_statistical_test
    ):
        """Additional columns are allowed in output."""
        additional_column_output = valid_dea_output.copy()
        additional_column_output["additional"] = 1
        mock_run_statistical_test.return_value = additional_column_output
        self.dea.perform(metadata=pd.DataFrame(np.zeros((3, 3))))


def test_dea_get_significance():
    """Test get_significance method."""
    result = valid_dea_output
    significant = DifferentialExpressionAnalysis.get_significance(result, 0.01)
    assert significant.equals(
        pd.DataFrame(
            [[True], [False], [False]],
            columns=[DeaColumns.SIGNIFICANT],
            index=["gene1", "gene2", "gene3"],
        )
    )


def test_dea_get_dict_key():
    """Test get_dict_key method."""
    assert (
        DifferentialExpressionAnalysis.get_dict_key(
            {DeaParameters.METADATA: pd.DataFrame()}
        )
        == """{'metadata': Empty DataFrame
Columns: []
Index: []}"""
    )


valid_parameter_input_two_groups = {
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

valid_data_input_two_groups = pd.DataFrame(
    [
        [1, 1.1, 3, 4],
        [2, 3, 4, 7],
        [3, 4, 4, 6],
        [0, 0, 0, 0],
        [np.nan, np.nan, np.nan, np.nan],
    ],
    columns=["sample1", "sample2", "sample3", "sample4"],
    index=["gene1", "gene2", "gene3", "zerogene", "nangene"],
).T


class TestableDifferentialExpressionAnalysisTwoGroups(
    DifferentialExpressionAnalysisTwoGroups
):
    """Testable implementation of DifferentialExpressionAnalysisTwoGroups."""

    @staticmethod
    def _allowed_parameters():
        return [
            DeaParameters.METADATA,
            DeaParameters.GROUP1,
            DeaParameters.GROUP2,
            DeaParameters.GROUPING_COLUMN,
        ]

    def _perform(self, **kwargs):
        group1, group2 = self._get_group_members(
            group1=kwargs[DeaParameters.GROUP1],
            group2=kwargs[DeaParameters.GROUP2],
            metadata=kwargs[DeaParameters.METADATA],
            grouping_column=kwargs[DeaParameters.GROUPING_COLUMN],
        )
        return self._run_statistical_test(self.mat, group1=group1, group2=group2)

    @staticmethod
    def _run_statistical_test(mat, group1, group2):
        return valid_dea_output


@patch(
    "test_differential_expression_analysis.TestableDifferentialExpressionAnalysisTwoGroups._run_statistical_test"
)
def test_dea_two_groups_perform_success(mock_run_statistical_test):
    """Test successful execution of DifferentialExpressionAnalysisTwoGroups."""
    mock_run_statistical_test.return_value = valid_dea_output
    dea = TestableDifferentialExpressionAnalysisTwoGroups(valid_data_input_two_groups)
    dea.perform(**valid_parameter_input_two_groups)
    assert mock_run_statistical_test.called_once()


def test_dea_two_groups_validation_missing_sample():
    """Test KeyError if sample is missing from data input."""
    dea = TestableDifferentialExpressionAnalysisTwoGroups(
        valid_data_input_two_groups.drop(index="sample1")
    )
    with pytest.raises(KeyError, match="sample1"):
        dea._validate_input(**valid_parameter_input_two_groups)


@patch(
    "alphastats.tl.differential_expression_analysis.DifferentialExpressionAnalysisTwoGroups._get_group_members"
)
def test_dea_two_groups_validation_calls_get_groups(mock_get_group_members):
    """Test that get_group_members is called during validation."""
    dea = TestableDifferentialExpressionAnalysisTwoGroups(valid_data_input_two_groups)
    mock_get_group_members.return_value = ["sample1", "sample2"], ["sample3", "sample4"]
    dea._validate_input(**valid_parameter_input_two_groups)
    mock_get_group_members.assert_called_once()


def test_dea_two_groups_get_group_members_from_metadata():
    """ "Test successful extraction of group members from metadata."""
    group1, group2 = DifferentialExpressionAnalysisTwoGroups._get_group_members(
        **valid_parameter_input_two_groups
    )
    assert group1 == ["sample1", "sample2"]
    assert group2 == ["sample3", "sample4"]


def test_dea_two_groups_get_group_members_from_lists():
    """Test successful passing of group member lists."""
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


def test_dea_two_groups_get_group_members_grouping_column():
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
    """Test successful execution of DifferentialExpressionAnalysisTTest."""
    expected_result = pd.DataFrame(
        [
            [0.03958486373817782, -2.45, 0.11875459121453347],
            [0.19821627426272678, -3.0, 0.2973244113940902],
            [0.3117527983883147, -1.5, 0.3117527983883147],
            [np.nan, 0.0, np.nan],
        ],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
        index=["gene1", "gene2", "gene3", "zerogene"],
    )
    dea = DifferentialExpressionAnalysisTTest(valid_data_input_two_groups)
    dea.perform(
        **valid_parameter_input_two_groups,
        **{
            DeaParameters.TEST_TYPE: DeaTestTypes.INDEPENDENT,
            DeaParameters.FDR_METHOD: "fdr_bh",
            PreprocessingStateKeys.LOG2_TRANSFORMED: True,
        },
    )
    pd.testing.assert_frame_equal(dea.result, expected_result)


def test_dea_ttest_perform_runs_log():
    """Test that log2 transformation is applied."""
    expected_result = pd.DataFrame(
        [
            [0.01570651089572518, -1.7237294884856105, 0.047119532687175544],
            [0.1556022268654943, -1.1111962106682238, 0.23340334029824145],
            [0.29794294575639113, -0.5, 0.29794294575639113],
        ],
        columns=[DeaColumns.PVALUE, DeaColumns.LOG2FC, DeaColumns.QVALUE],
        index=["gene1", "gene2", "gene3"],
    )
    dea = DifferentialExpressionAnalysisTTest(valid_data_input_two_groups)
    result = dea._perform(
        **valid_parameter_input_two_groups,
        **{
            DeaParameters.TEST_TYPE: DeaTestTypes.INDEPENDENT,
            DeaParameters.FDR_METHOD: "fdr_bh",
            PreprocessingStateKeys.LOG2_TRANSFORMED: False,
        },
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_dea_ttest_validation_wrong_stats_method():
    """Test ValueError if stats method is not recognized."""
    dea = DifferentialExpressionAnalysisTTest(valid_data_input_two_groups)
    with pytest.raises(
        ValueError,
        match="test_type must be either 'independent' for scipy.stats.ttest_ind or 'paired' for scipy.stats.ttest_rel.",
    ):
        dea._validate_input(
            **valid_parameter_input_two_groups,
            **{
                DeaParameters.TEST_TYPE: "not defined",
                DeaParameters.FDR_METHOD: "fdr_bh",
                PreprocessingStateKeys.LOG2_TRANSFORMED: True,
            },
        )


def test_dea_ttest_validation_wrong_fdr_method():
    """Test ValueError if fdr method is not recognized."""
    dea = DifferentialExpressionAnalysisTTest(valid_data_input_two_groups)
    with pytest.raises(
        ValueError, match="fdr_method must be one of 'fdr_bh', 'bonferroni'."
    ):
        dea._validate_input(
            **valid_parameter_input_two_groups,
            **{
                DeaParameters.TEST_TYPE: DeaTestTypes.INDEPENDENT,
                DeaParameters.FDR_METHOD: "unknown",
                PreprocessingStateKeys.LOG2_TRANSFORMED: True,
            },
        )
