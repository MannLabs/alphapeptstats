from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.multitest import multipletests

from alphastats.dataset.keys import Cols, ConstantsClass
from alphastats.dataset.preprocessing import PreprocessingStateKeys
from alphastats.statistics.statistic_utils import calculate_foldchange


class DeaColumns(ConstantsClass):
    PVALUE = "p-value"
    QVALUE = "q-value"
    LOG2FC = "log2(fold change)"
    SIGNIFICANT = "significant"


class DeaParameters(ConstantsClass):
    METADATA = "metadata"

    GROUP1 = "group1"
    GROUP2 = "group2"
    GROUPING_COLUMN = "grouping_column"

    TEST_FUN = "test_fun"
    FDR_METHOD = "fdr_method"


class DifferentialExpressionAnalysis(ABC):
    """This class implements the basic methods required for differential expression analysis.

    The purpose of this class is to provide a common interface for differential expression analysis. It should be subclassed for specific methods, such as t-tests or ANOVA. The class provides methods for input validation, output validation, and running the analysis. It also provides a method for getting the significance of the results based on a q-value cutoff.

    attributes:
    - input_data (pd.DataFrame): The input data for the analysis.
    - result (pd.DataFrame): The result of the analysis.

    abstract methods:
    - _allowed_parameters: that returns a list of allowed parameters for the analysis, static method
    - _extend_validation: extends the validation of parameters for the specific method
    - _run_statistical_test: wrapper runs the statistical test from kwargs
    - _statistical_test_fun: that runs the statistical test, static method. The wrapper and the actual method are separated to allow for easier testing and to ensure that all parameters are defined with types and defaults.

    public methods:
    - perform: performs the analysis and stores the result. This fixes the worflow of input validation, running the test and validating the output.
    - get_dict_key: generates a unique key for the result dictionary, static method
    - get_significance: returns the significant features based on the q-value cutoff, static method

    Intended usage:
    class DifferentialExpressionAnalysisTwoGroups(DifferentialExpressionAnalysis):
        implement shared methods for two-group analysis
    class DifferentialExpressionAnalysisTTest(DifferentialExpressionAnalysisTwoGroups):
        implement t-test specific methods
    dea = DifferentialExpressionAnalysisTTest(DataSet.mat)
    settings = {'group1': ['A', 'B'], 'group2': ['C', 'D'], 'test_fun': 'independent', 'fdr_method': 'fdr_bh'}
    result = dea.perform(**settings) # run once
    cached_results[dea.get_dict_key(settings)] = result
    significance = dea.get_significance(cached_results[dea.get_dict_key(settings)], 0.05) # run multiple times
    volcano_plot(cached_results[dea.get_dict_key(settings)], significance) # visualize
    """

    def __init__(self, input_data: pd.DataFrame) -> None:
        """Constructor for the DifferentialExpressionAnalysis class. Validates input and parameters.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis. This should be a DataFrame with the samples as rows and the features as columns.
        """
        self.input_data = input_data
        self.result: pd.DataFrame = None

    def _validate_input(self, parameters: dict) -> None:
        """Abstract method to validate the input and parameters. This should raise an exception if the input or parameters are invalid

        This function here checks for all parameters required for analysis regardless of the specific method, namely log2_transformed and metadata.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        parameters (dict): The parameters for the analysis.
        """
        if self.input_data is None:
            raise ValueError("No input data was provided.")
        if parameters is None:
            raise ValueError("No parameters were provided.")

        try:
            self._extend_validation(**parameters)
        except TypeError as err:
            raise TypeError(
                f"{str(err)}. Accepted keyword arguments to perform are {', '.join(self._allowed_parameters())}."
            ) from err

        for parameter in parameters:
            if parameter not in self._allowed_parameters():
                raise TypeError(
                    f"Parameter {parameter} should not be provided for this analysis. Accepted keyword arguments to perform are {', '.join(self._allowed_parameters())}."
                )

    @staticmethod
    @abstractmethod
    def _allowed_parameters() -> List[str]:
        """Method returning a list of allowed parameters for the analysis to avoid calling tests with additional parameters."""
        return []

    @abstractmethod
    def _extend_validation(self, **kwargs) -> None:
        pass

    def perform(self, **kwargs) -> Tuple[str, pd.DataFrame]:
        """Performs the differential expression analysis. Returns the result and stores it in the result attribute after validating its format.

        Returns:
        dict_key (str): A unique key based on the parameters that can be used for the result in a dictionary.
        result (pd.DataFrame): The result of the analysis.
        """
        self._validate_input(kwargs)
        result = self._run_statistical_test(**kwargs)
        self._validate_output(result)
        self.result = result
        return result

    @staticmethod
    def _validate_output(result: pd.DataFrame) -> None:
        """Validates the output of the analysis. This raises an exception if the output is invalid.

        The output should be a DataFrame with the columns for the p-value, q-value and log2fold-change.

        Parameters:
        result (pd.DataFrame): The result of the analysis.
        """
        if result is None:
            raise ValueError("No result was generated.")

        expected_columns = [DeaColumns.PVALUE, DeaColumns.QVALUE, DeaColumns.LOG2FC]

        for column in expected_columns:
            if column not in result.columns:
                raise KeyError(f"Column {column} is missing from the result.")

    @staticmethod
    def get_dict_key(parameters: dict) -> str:
        """Generates a unique key based on the parameters for the result dictionary."""
        return str(parameters)

    @abstractmethod
    def _run_statistical_test(self, **kwargs) -> pd.DataFrame:
        """Abstract methodwrapper to run the test. This should only rely on input_data and parameters and return the result. Output needs to conform with _validate_output

        Parameters:
        **kwargs: The parameters for the analysis. The keys need to be defined within the allowed_parameters method.

        Returns:
        pd.DataFrame: The result of the analysis.
        """
        pass

    @staticmethod
    @abstractmethod
    def _statistical_test_fun(input_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Static abstract method to run the statistical test. This must be run by _run_statistical_test and return the result of the analysis.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        **kwargs: The parameters for the analysis. The keys need to be defined within the allowed_parameters method.

        Returns:
        pd.DataFrame: The result of the analysis.
        """
        pass

    @staticmethod
    def get_significance(result: pd.DataFrame, qvalue_cutoff: float) -> pd.DataFrame:
        """Returns a DataFrame with the significant genes based on the q-value cutoff.

        Parameters:
        result (pd.DataFrame): The result of the analysis.
        qvalue_cutoff (float): The q-value cutoff for significance.

        Returns:
        pd.DataFrame: A DataFrame with a single binary column.
        """
        significance = pd.DataFrame(index=result.index)
        significance[DeaColumns.SIGNIFICANT] = result[DeaColumns.QVALUE] < qvalue_cutoff
        return significance


class DifferentialExpressionAnalysisTwoGroups(DifferentialExpressionAnalysis):
    """This class implements methods required specifically for two-group differential expression analysis."""

    def _extend_validation(
        self,
        group1: Union[List, str],
        group2: Union[List, str],
        grouping_column: Union[str, None] = None,
        metadata: Union[pd.DataFrame, None] = None,
        **kwargs,
    ):
        """Validates the input and parameters for the two-group differential expression analysis.

        This function checks for the required parameters for the two-group analysis, namely group1 and group2. If these are strings it additionally requires a grouping column, if these are lists it requires the samples to be present in the input data.

        Parameters:
        parameters (dict): The parameters for the analysis.
        """
        if isinstance(group1, str):
            if grouping_column is None:
                raise TypeError(
                    f"Parameter '{DeaParameters.GROUPING_COLUMN}' is missing."
                )
            if metadata is None:
                raise TypeError(f"Parameter '{DeaParameters.METADATA}' is missing.")
            group1_samples, group2_samples = self._get_group_members(
                group1=group1,
                group2=group2,
                grouping_column=grouping_column,
                metadata=metadata,
            )
        else:
            if grouping_column is not None:
                raise TypeError(
                    "Please provide either a list of columns OR the grouping column, not both."
                )
            group1_samples = group1
            group2_samples = group2
        for index in group1_samples + group2_samples:
            if index not in self.input_data.index:
                raise KeyError(f"Sample {index} is missing from the input data.")

    @staticmethod
    def _get_group_members(
        group1: Union[List, str],
        group2: Union[List, str],
        grouping_column: Union[str, None] = None,
        metadata: Union[pd.DataFrame, None] = None,
    ) -> Tuple[list, list]:
        """Returns the group columns based on the lists or retrieves it form the metadata."""
        if (
            grouping_column is None
            and isinstance(group1, list)
            and isinstance(group2, list)
        ):
            group1_samples = group1
            group2_samples = group2
        else:
            grouping_values = metadata[grouping_column]
            sample_values = metadata[Cols.SAMPLE]
            group1_samples = [
                sample
                for sample, group in zip(sample_values, grouping_values)
                if group == group1
            ]
            group2_samples = [
                sample
                for sample, group in zip(sample_values, grouping_values)
                if group == group2
            ]
        return group1_samples, group2_samples


class DifferentialExpressionAnalysisTTest(DifferentialExpressionAnalysisTwoGroups):
    """This class implements the t-test differential expression analysis."""

    @staticmethod
    def _allowed_parameters() -> List[str]:
        return [
            DeaParameters.TEST_FUN,
            DeaParameters.FDR_METHOD,
            DeaParameters.GROUP1,
            DeaParameters.GROUP2,
            DeaParameters.GROUPING_COLUMN,
            DeaParameters.METADATA,
            PreprocessingStateKeys.LOG2_TRANSFORMED,
        ]

    def _extend_validation(
        self,
        test_fun: str,
        fdr_method: str,
        **kwargs,
    ):
        """Validates the input and parameters for the t-test differential expression analysis.

        This function checks for the required parameters for the t-test analysis, namely test_fun and fdr_method. The test_fun must be either scipy.stats.ttest_ind or scipy.stats.ttest_rel and the fdr_method must be one of 'bh' or 'by'.

        Parameters:
        parameters (dict): The parameters for the analysis.
        """
        super()._extend_validation(**kwargs)
        if test_fun not in [
            "independent",
            "paired",
        ]:
            raise ValueError(
                "test_fun must be either 'independent' for scipy.stats.ttest_ind or 'paired' for scipy.stats.ttest_rel."
            )
        if fdr_method not in ["fdr_bh", "bonferroni"]:
            raise ValueError("fdr_method must be one of 'fdr_bh', 'bonferroni'.")

    def _run_statistical_test(self, **kwargs) -> pd.DataFrame:
        """Runs the t-test analysis and returns the result.
        Wrapper to method with actual method parameters and implementation.

        Returns:
        pd.DataFrame: The result of the analysis.
        """
        group1_samples, group2_samples = self._get_group_members(
            group1=kwargs[DeaParameters.GROUP1],
            group2=kwargs[DeaParameters.GROUP2],
            grouping_column=kwargs.get(DeaParameters.GROUPING_COLUMN, None),
            metadata=kwargs.get(DeaParameters.METADATA, None),
        )
        result = self._statistical_test_fun(
            input_data=self.input_data,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            is_log2_transformed=kwargs[PreprocessingStateKeys.LOG2_TRANSFORMED],
            test_fun=kwargs[DeaParameters.TEST_FUN],
            fdr_method=kwargs[DeaParameters.FDR_METHOD],
        )
        return result

    @staticmethod
    def _statistical_test_fun(
        input_data: pd.DataFrame,
        group1_samples: list,
        group2_samples: list,
        is_log2_transformed: bool,
        test_fun: str,
        fdr_method: str,
    ) -> pd.DataFrame:
        """Runs the t-test analysis and returns the result.

        Parameters:
        input_data (pd.DataFrame): The input data for the analysis.
        group1_samples (list): The samples for group 1.
        group2_samples (list): The samples for group 2.
        is_log2_transformed (bool): Whether the data is log2 transformed.
        test_fun (str): The test function to use, independent for scipy.stats.ttest_ind or paired for scipy.stats.ttest_rel.
        fdr_method (str): The FDR method to use, 'fdr_bh' or 'bonferroni'.

        Returns:
        pd.DataFrame: The result of the analysis.
        """
        mat_transpose = input_data.loc[group1_samples + group2_samples, :].transpose()

        test_fun = {
            "independent": scipy.stats.ttest_ind,
            "paired": scipy.stats.ttest_rel,
        }[test_fun]

        if not is_log2_transformed:
            mat_transpose = mat_transpose.transform(np.log2)
            mat_transpose = mat_transpose.replace([np.inf, -np.inf], np.nan)

        mat_transpose = mat_transpose.dropna(how="all")

        # TODO: return not only the p-value, but also the t-statistic
        p_values = mat_transpose.apply(
            lambda row: test_fun(
                row[group1_samples].values.flatten(),
                row[group2_samples].values.flatten(),
                nan_policy="omit",
            )[1],
            axis=1,
        )

        result = pd.DataFrame(index=mat_transpose.index)
        result[DeaColumns.PVALUE] = p_values.values
        result[DeaColumns.LOG2FC] = calculate_foldchange(
            mat_transpose=mat_transpose,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            is_log2_transformed=True,
        )

        result[DeaColumns.QVALUE] = multipletests(p_values.values, method=fdr_method)[1]
        return result
