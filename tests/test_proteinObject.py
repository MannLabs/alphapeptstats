from calendar import c
from math import remainder
from random import sample
from ssl import TLSVersion
import unittest
import pandas as pd
import logging
from unittest.mock import patch
import logging
import numpy as np
import pandas as pd

# from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

from alphastats.loader.BaseLoader import BaseLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.proteinObject import proteinObject

logger = logging.getLogger(__name__)


class BaseTestProteinObject:
    #  parent class of test loader for common tests among loaders
    # this is wrapped in a nested class so it doesnt get called separatly when testing
    # plus to avoid multiple inheritance
    class BaseTest(unittest.TestCase):
        @patch("logging.Logger.error")
        def test_check_loader_no_error(self, mock):
            # check if loader is valid
            self.obj.check_loader(loader=self.loader)
            mock.assert_not_called()

        # @patch("logging.Logger.error")
        # def test_check_loader_error_invalid_column(self, mock):
        #  invalid index column
        #    self.loader.index_column = 100
        #    self.obj.check_loader(loader=self.loader)
        #    mock.assert_called_once()

        # @patch("logging.Logger.error")
        # def test_check_loader_error_empty_df(self, mock):
        # empty dataframe
        #    self.loader.rawdata = pd.DataFrame()
        #    self.obj.check_loader(loader=self.loader)
        #    mock.assert_called_once()

        @patch("logging.Logger.error")
        def test_check_loader_error_invalid_loader(self, mock):
            #  invalid loader, class
            df = pd.DataFrame()
            self.obj.check_loader(loader=df)
            mock.assert_called_once()

        def test_load_metadata(self):
            #  is dataframe loaded
            self.assertIsInstance(self.obj.metadata, pd.DataFrame)
            self.assertFalse(self.obj.metadata.empty)

        @patch("logging.Logger.error")
        def test_load_metadata_missing_sample_column(self, mock):
            # is error raised when name of sample column is missing
            path = self.metadata_path
            self.obj.load_metadata(file_path=path, sample_column="wrong_sample_column")
            mock.assert_called_once()

        @patch("logging.Logger.warning")
        def test_load_metadata_warning(self, mock):
            # is dataframe None and is warning produced
            file_path = "wrong/file.xxx"
            self.obj.load_metadata(file_path=file_path, sample_column="sample")
            mock.assert_called_once()

        def test_create_matrix(self):
            #  matrix dimensions
            self.assertEqual(self.obj.mat.shape, self.matrix_dim)
            # does the matrix only contain floats/integers and NAs
            is_dtype_numeric = list(
                set(list(map(pd.api.types.is_numeric_dtype, self.obj.mat.dtypes)))
            )
            self.assertEqual(is_dtype_numeric, [True])

        @patch("logging.Logger.info")
        def test_preprocess_filter(self, mock):
            # is warning raised when filter columns are none
            # is info printed if contamination columns get removed
            # is the new matrix smaller than the older matrix
            self.obj.preprocess(remove_contaminations=True)
            self.assertEqual(self.obj.mat.shape, self.matrix_dim_filtered)
            #  info has been printed at least once
            mock.assert_called()


class TestAlphaPeptProteinObject(BaseTestProteinObject.BaseTest):
    #  do testing which requires extra files only on TestAlphaPeptProteinObject
    # to reduce the amount of compariosn files required
    def setUp(self):
        self.loader = AlphaPeptLoader(file="testfiles/alphapept_results_proteins.csv")
        self.metadata_path = "testfiles/alphapept_metadata.xlsx"
        self.obj = proteinObject(
            loader=self.loader,
            metadata_path="testfiles/alphapept_metadata.xlsx",
            sample_column="sample",
        )
        # expected dimensions of matrix
        self.matrix_dim = (2, 3781)
        self.matrix_dim_filtered = (2, 3743)

    def test_load_metadata_fileformats(self):
        # test if different fileformats get loaded correctly
        print("txt")
        metadata_path = "testfiles/alphapept_metadata.txt"
        self.obj.load_metadata(file_path=metadata_path, sample_column="sample")
        self.assertEqual(self.obj.metadata.shape, (2, 2))

        print("tsv")
        metadata_path = "testfiles/alphapept_metadata.tsv"
        self.obj.load_metadata(file_path=metadata_path, sample_column="sample")
        self.assertEqual(self.obj.metadata.shape, (2, 2))

        print("csv")
        metadata_path = "testfiles/alphapept_metadata.csv"
        self.obj.load_metadata(file_path=metadata_path, sample_column="sample")
        self.assertEqual(self.obj.metadata.shape, (2, 2))

    def test_preprocess_normalize_zscore(self):
        self.obj.mat = pd.DataFrame({"a": [2, 5, 4], "b": [5, 4, 4], "c": [0, 10, 8]})
        # zscore Normalization
        self.obj.preprocess(normalization="zscore")
        expected_mat = pd.DataFrame(
            {
                "a": [-1.33630621, 1.06904497, 0.26726124],
                "b": [1.41421356, -0.70710678, -0.70710678],
                "c": [-1.38873015, 0.9258201, 0.46291005],
            }
        )
        # self.assertEqual(self.obj.mat.values, expected_mat.values)
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_normalize_quantile(self):
        self.obj.mat = pd.DataFrame({"a": [2, 5, 4], "b": [5, 4, 4], "c": [0, 10, 8]})
        # Quantile Normalization
        self.obj.preprocess(normalization="quantile")
        expected_mat = pd.DataFrame(
            {"a": [0.0, 1.0, 0.5], "b": [1.0, 0.0, 0.0], "c": [0.0, 1.0, 0.5]}
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_normalize_linear(self):
        self.obj.mat = pd.DataFrame({"a": [2, 5, 4], "b": [5, 4, 4], "c": [0, 10, 8]})
        # Linear Normalization
        self.obj.preprocess(normalization="linear")
        expected_mat = pd.DataFrame(
            {
                "a": [0.37139068, 0.42107596, 0.40824829],
                "b": [0.92847669, 0.33686077, 0.40824829],
                "c": [0.0, 0.84215192, 0.81649658],
            }
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_mean(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(imputation="mean")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_median(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(imputation="median")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_knn(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(imputation="knn")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_calculate_ttest_fc(self):
        # are df dimension correct
        # are calculations correct
        # take first row
        pass

    def test_plot_volcano(self):
        #  https://campus.datacamp.com/courses/unit-testing-for-data-science-in-python/testing-models-plots-and-much-more?ex=11
        pass

    def test_plot_sampledistribution(self):
        pass

    def test_plot_correlation_matrix(self):
        pass


class TestMaxQuantProteinObject(BaseTestProteinObject.BaseTest):
    def setUp(self):
        self.loader = MaxQuantLoader(file="testfiles/maxquant_proteinGroups.txt")
        self.metadata_path = "testfiles/maxquant_metadata.xlsx"
        self.obj = proteinObject(
            loader=self.loader,
            metadata_path="testfiles/maxquant_metadata.xlsx",
            sample_column="sample",
        )
        # expected dimensions of matrix
        self.matrix_dim = (312, 2611)
        self.matrix_dim_filtered = (312, 2409)


class TestDIANNProteinObject(BaseTestProteinObject.BaseTest):
    def setUp(self):
        self.loader = DIANNLoader(file="testfiles/diann_report_final.pg_matrix.tsv")
        self.metadata_path = "testfiles/diann_metadata.xlsx"
        self.obj = proteinObject(
            loader=self.loader,
            metadata_path="testfiles/diann_metadata.xlsx",
            sample_column="analytical_sample external_id",
        )
        # expected dimensions of matrix
        self.matrix_dim = (20, 10)
        self.matrix_dim_filtered = (20, 10)


class TestFragPipeProteinObject(BaseTestProteinObject.BaseTest):
    def setUp(self):
        self.loader = FragPipeLoader(
            file="testfiles/fragpipe_combined_proteins.tsv",
            intensity_column="[experiment] Razor Intensity",
        )
        self.metadata_path = "testfiles/fragpipe_metadata.xlsx"
        self.obj = proteinObject(
            loader=self.loader,
            metadata_path="testfiles/fragpipe_metadata.xlsx",
            sample_column="analytical_sample external_id",
        )
        # expected dimensions of matrix
        self.matrix_dim = (20, 10)
        self.matrix_dim_filtered = (20, 10)


if __name__ == "__main__":
    unittest.main()
