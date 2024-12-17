import pandas as pd
import pytest

from alphastats.tl.differential_expression_analysis import (
    DeaColumns,
    DeaParameters,
    DifferentialExpressionAnalysis,
)


class TestableDifferentialExpressionAnalysis(DifferentialExpressionAnalysis):
    def allowed_parameters(self):
        return [DeaParameters.METADATA]

    def _extend_validation(self, input_data, parameters):
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
    with pytest.raises(ValueError):
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
        TestableDifferentialExpressionAnalysis.get_dict_key(
            {DeaParameters.METADATA: pd.DataFrame()}
        )
        == """{'metadata': Empty DataFrame
Columns: []
Index: []}"""
    )
