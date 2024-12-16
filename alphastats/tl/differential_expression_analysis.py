from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import scipy

from alphastats.dataset.keys import Cols, ConstantsClass
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.statistics.statistic_utils import calculate_foldchange


class DEA_COLUMNS(ConstantsClass):
    PVALUE = "p-value"
    QVALUE = "q-value"
    LOG2FC = "log2(fold change)"
    SIGNIFICANT = "significant"


class DEA_PARAMETERS(ConstantsClass):
    GROUP1 = "group1"
    GROUP2 = "group2"
    TEST_FUN = "test_fun"
    FDR_METHOD = "fdr_method"
    METADATA = "metadata"
    GROUPING_COLUMN = "grouping_column"


class DifferentialExpressionAnalysis(ABC):
    input_data: pd.DataFrame = None
    result: pd.DataFrame = None
    parameters: dict = None

    def __init__(self, input_data: pd.DataFrame, parameters: dict) -> None:
        """Constructor for the DifferentialExpressionAnalysis class. Validates input and parameters.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        parameters (dict): The parameters for the analysis.
        """
        self._validate_input(input_data, parameters)
        self.parameters = parameters.copy()
        self.input_data = input_data.copy()

    @abstractmethod
    def _validate_input(input_data: pd.DataFrame, parameters: dict) -> None:
        """Abstract method to validate the input and parameters. This should raise an exception if the input or parameters are invalid

        This function here checks for all parameters required for analysis regardless of the specific method, namely log2_transformed and metadata.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        parameters (dict): The parameters for the analysis.
        """
        if input_data is None:
            raise Exception("No input data was provided.")
        if parameters is None:
            raise Exception("No parameters were provided.")
        else:
            expected_parameters = [
                PreprocessingStateKeys.LOG2_TRANSFORMED,
                DEA_PARAMETERS.METADATA,
            ]
            for parameter in expected_parameters:
                if parameter not in parameters:
                    raise Exception(f"Parameter {parameter} is missing.")

    def perform(self) -> Tuple[str, pd.DataFrame]:
        """Performs the differential expression analysis. Returns the result and stores it in the result attribute after validating its format.

        Returns:
        dict_key (str): A unique key based on the parameters that can be used for the result in a dictionary.
        result (pd.DataFrame): The result of the analysis.
        """
        result = self._run_test()
        self._validate_output(result)
        self.result = result
        dict_key = self.get_dict_key(self.parameters)
        return dict_key, result

    def _validate_output(result: pd.DataFrame) -> None:
        """Validates the output of the analysis. This raises an exception if the output is invalid.

        The output should be a DataFrame with the columns for the p-value, q-value and log2fold-change.

        Parameters:
        result (pd.DataFrame): The result of the analysis."""
        if result is None:
            raise Exception("No result was generated.")

        expected_columns = [DEA_COLUMNS.PVALUE, DEA_COLUMNS.QVALUE, DEA_COLUMNS.LOG2FC]

        for column in expected_columns:
            if column not in result.columns:
                raise Exception(f"Column {column} is missing from the result.")

    def get_dict_key(parameters: dict) -> str:
        """Generates a unique key based on the parameters for the result dictionary."""
        return str(parameters)

    @abstractmethod
    def _run_test(self):
        """Abstract methodwrapper to run the test. This should only rely on input_data and parameters and return the result. Output needs to conform with _validate_output"""
        pass

    @abstractmethod
    def _test_fun(**kwargs):
        """Abstract method to run the test. This should be called by _run_test, explicitly define which parameters are required and should return the result. Output needs to conform with _validate_output"""
        pass

    def get_significance(result: pd.DataFrame, qvalue_cutoff: float) -> pd.DataFrame:
        """Returns a DataFrame with the significant genes based on the q-value cutoff.

        Parameters:
        result (pd.DataFrame): The result of the analysis.
        qvalue_cutoff (float): The q-value cutoff for significance.

        Returns:
        pd.DataFrame: A DataFrame with a single binary column.
        """
        significance = pd.DataFrame(index=result.index)
        significance[DEA_COLUMNS.SIGNIFICANT] = (
            result[DEA_COLUMNS.QVALUE] < qvalue_cutoff
        )
        return significance


class DifferentialExpressionAnalysis_TwoGroups(DifferentialExpressionAnalysis):
    def _validate_input(input_data: pd.DataFrame, parameters: dict):
        """Validates the input and parameters for the two-group differential expression analysis.

        This function checks for the required parameters for the two-group analysis, namely group1 and group2. If these are strings it additionally requires a grouping column, if these are lists it requires the samples to be present in the input data.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        parameters (dict): The parameters for the analysis.
        """
        super()._validate_input(input_data, parameters)
        expected_parameters = [DEA_PARAMETERS.GROUP1, DEA_PARAMETERS.GROUP2]
        for parameter in expected_parameters:
            if parameter not in parameters:
                raise Exception(f"Parameter {parameter} is missing.")
        if isinstance(parameters["group1"], list):
            if DEA_PARAMETERS.GROUPING_COLUMN in parameters:
                raise Exception(
                    "Please provide either a list of columns OR the grouping column, not both."
                )
            for index in parameters["group1"]:
                if index not in input_data.index:
                    raise Exception(f"Sample {index} is missing from the input data.")
            for index in parameters["group2"]:
                if index not in input_data.index:
                    raise Exception(f"Sample {index} is missing from the input data.")
        else:
            if DEA_PARAMETERS.GROUPING_COLUMN not in parameters:
                raise Exception(
                    f"Parameter {DEA_PARAMETERS.GROUPING_COLUMN} is missing."
                )

    def _get_group_columns(self) -> Tuple[list, list]:
        """Returns the group columns based on the parameters."""
        if DEA_PARAMETERS.GROUPING_COLUMN in self.parameters:
            metadata = self.parameters[DEA_PARAMETERS.METADATA]
            group1 = list(
                metadata[
                    metadata[self.parameters[DEA_PARAMETERS.GROUPING_COLUMN]]
                    == self.parameters[DEA_PARAMETERS.GROUP1],
                    [Cols.SAMPLE],
                ]
            )
            group2 = list(
                metadata[
                    metadata[self.parameters[DEA_PARAMETERS.GROUPING_COLUMN]]
                    == self.parameters[DEA_PARAMETERS.GROUP2],
                    [Cols.SAMPLE],
                ]
            )
        else:
            group1 = self.parameters[DEA_PARAMETERS.GROUP1]
            group2 = self.parameters[DEA_PARAMETERS.GROUP2]
        return group1, group2


class DifferentialExpressionAnalysis_TTest(DifferentialExpressionAnalysis_TwoGroups):
    def _validate_input(input_data: pd.DataFrame, parameters: dict):
        """Validates the input and parameters for the t-test differential expression analysis.

        This function checks for the required parameters for the t-test analysis, namely test_fun and fdr_method. The test_fun must be either scipy.stats.ttest_ind or scipy.stats.ttest_rel and the fdr_method must be one of 'bh' or 'by'.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        parameters (dict): The parameters for the analysis.
        """
        super()._validate_input(input_data, parameters)
        if "test_fun" not in parameters:
            raise Exception("No test_fun was provided.")
        else:
            if parameters["test_fun"] not in [
                scipy.stats.ttest_ind,
                scipy.stats.ttest_rel,
            ]:
                raise ValueError(
                    "test_fun must be either scipy.stats.ttest_ind or scipy.stats.ttest_rel for t-test."
                )
        if "fdr_method" not in parameters:
            raise Exception("No fdr_method was provided.")
        else:
            if parameters["fdr_method"] not in ["bh", "by"]:
                raise ValueError("fdr_method must be one of 'bh', 'by'.")

    def _run_test(self) -> pd.DataFrame:
        """Runs the t-test analysis and returns the result.
        Wrapper to method with actual method parameters and implementation.

        Returns:
        pd.DataFrame: The result of the analysis.
        """
        group1, group2 = self._get_group_columns()
        result = self._test_fun(
            input_data=self.input_data,
            group1=group1,
            group2=group2,
            is_log2_transformed=self.parameters[
                PreprocessingStateKeys.LOG2_TRANSFORMED
            ],
            test_fun=self.parameters[DEA_PARAMETERS.TEST_FUN],
            fdr_method=self.parameters[DEA_PARAMETERS.FDR_METHOD],
        )
        return result

    def _test_fun(
        input_data: pd.DataFrame,
        group1: list,
        group2: list,
        is_log2_transformed: bool,
        test_fun: callable,
        fdr_method: str,
    ) -> pd.DataFrame:
        """Runs the t-test analysis and returns the result.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        group1 (list): The samples for group 1.
        group2 (list): The samples for group 2.
        is_log2_transformed (bool): Whether the data is log2 transformed.
        test_fun (callable): The test function to use, scipy.stats.ttest_ind or scipy.stats.ttest_rel.
        fdr_method (str): The FDR method to use, 'bh' or 'by'.

        Returns:
        pd.DataFrame: The result of the analysis.
        """
        mat_transpose = input_data.loc[group1 + group2, :].transpose()

        if not is_log2_transformed:
            mat_transpose = mat_transpose.transform(lambda x: np.log2(x))
            mat_transpose = mat_transpose.replace([np.inf, -np.inf], np.nan)

        mat_transpose = mat_transpose.dropna(how="all")

        # TODO: return not only the p-value, but also the t-statistic
        p_values = mat_transpose.apply(
            lambda row: test_fun(
                row[group1].values.flatten(),
                row[group2].values.flatten(),
                nan_policy="omit",
            )[1],
            axis=1,
        )

        result = pd.DataFrame(input_data.index)
        result[DEA_COLUMNS.PVALUE] = p_values.values
        result[DEA_COLUMNS.LOG2FC] = calculate_foldchange(
            mat_transpose=mat_transpose,
            group1_samples=group1,
            group2_samples=group2,
            is_log2_transformed=True,
        )

        result[DEA_COLUMNS.QVALUE] = scipy.stats.false_discovery_control(
            p_values.values, method=fdr_method
        )
        return result
