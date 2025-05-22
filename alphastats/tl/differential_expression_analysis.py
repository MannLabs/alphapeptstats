"""Base class for differential expression analysis as well as implementations of specific tests."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Literal

import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.multitest import multipletests

from alphastats.dataset.keys import Cols, ConstantsClass
from alphastats.statistics.statistic_utils import calculate_foldchange


class DeaColumns(ConstantsClass):
    """Constants for the columns in the result of the differential expression analysis."""

    PVALUE = "p-value"
    QVALUE = "q-value"
    LOG2FC = "log2(fold change)"  # group1-group2
    SIGNIFICANTQ = "significant q-value"


class DeaTestTypes(ConstantsClass):
    """Constants for the test types in the differential expression analysis."""

    INDEPENDENT = "independent"
    PAIRED = "paired"


def _validate_perform(func: Callable) -> Callable:
    """Decorator to wrap the perform method in input and output validation."""
    func._validate = True  # noqa: SLF001

    @wraps(func)  # This is needed to keep the function signature
    def wrapper(self: DifferentialExpressionAnalysis, **kwargs) -> pd.DataFrame:
        self._validate_input(**kwargs)
        result = func(self, **kwargs)
        self._validate_output(result)
        return result

    return wrapper


class DifferentialExpressionAnalysis(ABC):
    """Implements the basic methods required for differential expression analysis.

    The purpose of this class is to provide a common interface for differential expression analysis.
    It should be subclassed for specific methods, such as t-tests or ANOVA. The class provides methods
    for input validation, output validation, and running the analysis. It also provides a method for
    getting the significance of the results based on a q-value cutoff.

    Attributes
    ----------
    mat : pd.DataFrame
        The input data for the analysis.
    is_log2_transformed : bool
        Whether the data is log2 transformed.

    Methods
    -------
    get_significance(result: pd.DataFrame, qvalue_cutoff: float) -> pd.DataFrame
        Returns the significant features based on the q-value cutoff.

    Abstract Methods
    ----------------
    perform(**kwargs) -> pd.DataFrame
        Function that runs any preparatory steps and the statistical test. This is required to be decorated with @_validate_perform to fix the workflow of input validation, running the test, and validating the output.
    _extend_validation(**kwargs)
        Extends the validation of parameters for the specific method.
    _run_statistical_test(mat: pd.DataFrame, **kwargs) -> pd.DataFrame
        Runs the statistical test. The wrapper and the actual method are separated to allow for easier
        testing and to ensure that all parameters are defined with types and defaults.

    Intended Usage
    --------------
        Implement shared methods for two-group analysis.
        Implement t-test specific methods.
    result = dea.perform(**settings)  # run once
    significance = dea.get_significance(cached_result, 0.05)  # run multiple times
    volcano_plot(cached_result, significance)  # visualize

    """

    def __init__(self, mat: pd.DataFrame, *, is_log2_transformed: bool) -> None:
        """Constructor for the DifferentialExpressionAnalysis class. sets up mat and results.

        Parameters
        ----------
        mat : pd.DataFrame
            The input data for the analysis. This should be a DataFrame with the samples as rows and the features as columns.
        is_log2_transformed : bool
            Whether the data is log2 transformed.

        """
        if mat.empty:
            raise ValueError(
                "Input matrix to differential expression analysis is empty."
            )
        self.mat = mat
        self.is_log2_transformed = is_log2_transformed

    def _validate_input(self, **kwargs) -> None:
        """Abstract method to validate the input and parameters.

        This should raise an exception if the input or parameters are invalid.
        This function here checks for all parameters required for analysis regardless of the specific method, namely log2_transformed and metadata.

        Parameters
        ----------
        **kwargs : dict
            The parameters for the analysis. The keys need to be defined within the allowed_parameters method.

        """
        self._extend_validation(**kwargs)

    @abstractmethod
    def _extend_validation(self, **kwargs) -> None:
        """Abstract method to extend the validation of parameters for the specific method. This should raise an exception if the input or parameters are invalid.

        It should have all parameters required for the method as keyword arguments and validate that parameters are compatible with mat.
        """

    @staticmethod
    def _validate_output(result: pd.DataFrame) -> None:
        """Validates the output of the analysis. This raises an exception if the output is invalid.

        The output should be a DataFrame with the columns for the p-value, q-value and log2fold-change.

        Parameters
        ----------
        result : pd.DataFrame
            The result of the analysis.

        """
        if result.empty:
            raise ValueError("The result dataframe is empty.")

        expected_columns = [DeaColumns.PVALUE, DeaColumns.QVALUE, DeaColumns.LOG2FC]

        for column in expected_columns:
            if column not in result.columns:
                raise KeyError(f"Column '{column}' is missing from the result.")

    @property
    @abstractmethod
    def perform(self, **kwargs) -> pd.DataFrame:  # noqa: PLR0206
        """Abstract methodwrapper to run the test.

        This should only rely on mat and parameters and return the result. Output needs to conform with _validate_output.

        Parameters
        ----------
        **kwargs: dict
            The parameters for the analysis. The keys need to be defined within the allowed_parameters method.

        Returns
        -------
        pd.DataFrame: The result of the analysis.

        """

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """If a subclass implements perform this checks if the implementation is decorated with @validate."""
        super().__init_subclass__(**kwargs)
        if "perform" in cls.__dict__ and not getattr(cls.perform, "_validate", False):
            raise TypeError(
                f"perform in {cls.__name__} must use @_validate_perform decorator"
            )

    @staticmethod
    @abstractmethod
    def _run_statistical_test(mat: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Static abstract method to run the statistical test. This must be run by _run_statistical_test and return the result of the analysis.

        Parameters
        ----------
        mat : pd.DataFrame
            The input data for the analysis.
        **kwargs : dict
            The parameters for the analysis. The keys need to be defined within the allowed_parameters method.

        Returns
        -------
        pd.DataFrame: The result of the analysis.

        """

    @staticmethod
    def get_significance_qvalue(
        result: pd.DataFrame, qvalue_cutoff: float
    ) -> pd.DataFrame:
        """Returns a DataFrame with the significant genes based on the q-value cutoff.

        Parameters
        ----------
        result : pd.DataFrame
            The result of the analysis.
        qvalue_cutoff : float
            The q-value cutoff for significance.

        Returns
        -------
        pd.DataFrame: A DataFrame with a single binary column.

        """
        significance = pd.DataFrame(index=result.index)
        significance[DeaColumns.SIGNIFICANTQ] = (
            result[DeaColumns.QVALUE] <= qvalue_cutoff
        )
        return significance


class DifferentialExpressionAnalysisTwoGroups(DifferentialExpressionAnalysis, ABC):
    """Implementation of methods required specifically for two-group differential expression analysis."""

    def _extend_validation(
        self,
        group1: list | str,
        group2: list | str,
        grouping_column: str | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> None:
        """Validates the input and parameters for the two-group differential expression analysis.

        This function checks for the required parameters for the two-group analysis, namely group1 and group2 are valid contained in the mat.

        Parameters
        ----------
        group1 : list | str
            The first group.
        group2 : list | str
            The second group.
        grouping_column : str | None
            The column in the metadata to group by.
        metadata : pd.DataFrame | None
            The metadata DataFrame.

        """
        group1_samples, group2_samples = self._get_group_members(
            group1=group1,
            group2=group2,
            grouping_column=grouping_column,
            metadata=metadata,
        )
        for index in group1_samples + group2_samples:
            if index not in self.mat.index:
                raise KeyError(f"Sample {index} is missing from the input data.")

    @staticmethod
    def _get_group_members(
        group1: list | str,
        group2: list | str,
        grouping_column: str | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> tuple(list, list):
        """Returns the group columns based on the lists or retrieves it from the metadata.

        Parameters
        ----------
        group1 : list | str
            The first group.
        group2 : list | str
            The second group.
        grouping_column : str | None
            The column in the metadata to group by.
        metadata : pd.DataFrame | None
            The metadata DataFrame.

        Returns
        -------
        Tuple[list, list]: The samples for group 1 and group 2.

        """
        if grouping_column is None:
            if not isinstance(group1, list) or not isinstance(group2, list):
                raise TypeError(
                    "If grouping_column is not provided, group1 and group2 must be lists of sample names. Alternatively 'grouping_column' is missing."
                )
            group1_samples = group1
            group2_samples = group2
        else:
            if not isinstance(group1, str) or not isinstance(group2, str):
                raise TypeError(
                    "If grouping_column is provided, group1 and group2 must be strings matching elements in the column."
                )
            if not isinstance(metadata, pd.DataFrame):
                raise TypeError(
                    "If grouping_column is provided, 'metadata' must be provided."
                )
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
        if len(group1_samples) == 0:
            raise ValueError("No samples found for group 1.")
        if len(group2_samples) == 0:
            raise ValueError("No samples found for group 2.")
        return group1_samples, group2_samples


class DifferentialExpressionAnalysisTTest(DifferentialExpressionAnalysisTwoGroups):
    """Implementation of the t-test differential expression analysis.

    Examples
    --------
    >>> mat = pd.DataFrame({
    ...     'sample1': [1, 2, 3],
    ...     'sample2': [4, 5, 6],
    ...     'sample3': [7, 8, 9],
    ...     'sample4': [10, 11, 12]
    ... }, index=['gene1', 'gene2', 'gene3']).T
    >>> dea = DifferentialExpressionAnalysisTTest(mat)
    >>> result = dea.perform(
    ...     group1_samples = ['sample1', 'sample2'],
    ...     group2_samples = ['sample3', 'sample4'],
    ...     is_log2_transformed = True,
    ...     test_type = "independent",
    ...     fdr_method = "fdr_bh",
    ... )
    >>> result.columns.tolist()
    ['p-value', 'log2(fold change)', 'q-value']
    >>> result.index.tolist()
    ['gene1', 'gene2', 'gene3']

    """

    def _extend_validation(
        self,
        test_type: str,
        fdr_method: str,
        **kwargs,
    ) -> None:
        """Validates the input and parameters for the t-test differential expression analysis.

        Parameters
        ----------
        test_type : str
            The test function to use, independent for scipy.stats.ttest_ind or paired for scipy.stats.ttest_rel.
        fdr_method : str
            The FDR method to use, 'fdr_bh' or 'bonferroni'.
        **kwargs : dict
            Additional arguments passed to perform.

        """
        super()._extend_validation(**kwargs)
        if test_type not in [
            DeaTestTypes.INDEPENDENT,
            DeaTestTypes.PAIRED,
        ]:
            raise ValueError(
                f"test_type must be either '{DeaTestTypes.INDEPENDENT}' for scipy.stats.ttest_ind or '{DeaTestTypes.PAIRED}' for scipy.stats.ttest_rel."
            )
        if fdr_method not in ["fdr_bh", "bonferroni"]:
            raise ValueError("fdr_method must be one of 'fdr_bh', 'bonferroni'.")

    @_validate_perform
    def perform(  # noqa: PLR0913
        self,
        test_type: Literal[DeaTestTypes.INDEPENDENT, DeaTestTypes.PAIRED],
        fdr_method: Literal["fdr_bh", "bonferroni"],
        group1: list | str,
        group2: list | str,
        grouping_column: str | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Runs the t-test analysis and returns the result.

        Wrapper to statistical method with actual method parameters and implementation.

        Returns
        -------
        pd.DataFrame: The result of the analysis.

        """
        group1_samples, group2_samples = self._get_group_members(
            group1=group1,
            group2=group2,
            grouping_column=grouping_column,
            metadata=metadata,
        )

        return self._run_statistical_test(
            mat=self.mat,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            is_log2_transformed=self.is_log2_transformed,
            test_type=test_type,
            fdr_method=fdr_method,
        )

    @staticmethod
    def _run_statistical_test(  # noqa: PLR0913
        mat: pd.DataFrame,
        group1_samples: list,
        group2_samples: list,
        test_type: Literal[DeaTestTypes.INDEPENDENT, DeaTestTypes.PAIRED],
        fdr_method: Literal["fdr_bh", "bonferroni"],
        *,
        is_log2_transformed: bool,
    ) -> pd.DataFrame:
        """Runs the t-test analysis and returns the result.

        Parameters
        ----------
        mat : pd.DataFrame)
            The input data for the analysis.
        group1_samples : list)
            The samples for group 1.
        group2_samples : list
            The samples for group 2.
        test_type : str
            The test function to use, independent for scipy.stats.ttest_ind or paired for scipy.stats.ttest_rel.
        fdr_method : str
            The FDR method to use, 'fdr_bh' or 'bonferroni'.
        is_log2_transformed : bool
            Whether the data is log2 transformed. If not, this well be done before the analysis.

        Returns
        -------
        pd.DataFrame: The result of the analysis.

        """
        mat_transpose = mat.loc[group1_samples + group2_samples, :].transpose()

        test_fun = {
            DeaTestTypes.INDEPENDENT: scipy.stats.ttest_ind,
            DeaTestTypes.PAIRED: scipy.stats.ttest_rel,
        }[test_type]

        if not is_log2_transformed:
            mat_nans = mat_transpose.isna().sum().sum()
            mat_transpose = mat_transpose.transform(np.log2)
            mat_transpose = mat_transpose.replace([np.inf, -np.inf], np.nan)
            new_mat_nans = mat_transpose.isna().sum().sum()
            warnings.warn(
                f"Automatic log2 transformation was performed prior to ttest analysis. {new_mat_nans-mat_nans!s} values were replaced with NaN in the process.",
                UserWarning,
            )

        mat_len = mat_transpose.shape[0]
        mat_transpose = mat_transpose.dropna(how="all")
        if mat_len > (filtered_mat_len := mat_transpose.shape[0]):
            warnings.warn(
                f"{mat_len-filtered_mat_len!s} proteins contain only NaN values and are removed prior to ttest analysis.",
                UserWarning,
            )

        # TODO: return not only the p-value, but also the t-statistic
        # TODO: Make sure this apply is efficient and address noqa: PD011
        p_values = mat_transpose.apply(
            lambda row: test_fun(
                row[group1_samples].values.flatten(),  # noqa: PD011
                row[group2_samples].values.flatten(),  # noqa: PD011
                nan_policy="omit",
            )[1],
            axis=1,
        )

        result = pd.DataFrame(index=mat_transpose.index)
        result[DeaColumns.PVALUE] = p_values.to_numpy()
        result[DeaColumns.LOG2FC] = calculate_foldchange(
            mat_transpose=mat_transpose,
            group1_samples=group1_samples,
            group2_samples=group2_samples,
            is_log2_transformed=True,
        )

        qvalues = pd.Series(
            multipletests(result[DeaColumns.PVALUE].dropna().values, method=fdr_method)[
                1
            ],
            name=DeaColumns.QVALUE,
            index=result[DeaColumns.PVALUE].dropna().index,
        )
        return result.merge(qvalues, left_index=True, right_index=True, how="left")
