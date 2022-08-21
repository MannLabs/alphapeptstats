import unittest
import pandas as pd
import logging
from unittest.mock import patch
import logging


from alphastats.loader.BaseLoader import BaseLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class BaseTestLoader:
    #  parent class of test loader for common tests among loaders
    # this is wrapped in a nested class so it doesnt get called separatly when testing
    # plus to avoid multiple inheritance
    class BaseTest(unittest.TestCase):
        @contextmanager
        def assertNotRaises(self, exc_type):
            try:
                yield None
            except exc_type:
                raise self.failureException("{} raised".format(exc_type.__name__))

        def test_dataformat(self):
            # check if loaded data is pandas dataframe
            self.assertIsInstance(self.obj.rawdata, pd.DataFrame)

        def test_check_if_columns_are_present_error(self):
            # check if columns are present
            # check if error gets raised when column is not present
            with self.assertRaises(KeyError):
                self.obj.confidence_column = "wrong_column"
                self.obj.check_if_columns_are_present()

        def test_check_if_columns_are_present_no_error(self):
            # check if columns are present
            # check if error gets raised when column is not present
            with self.assertNotRaises(KeyError):
                self.obj.check_if_columns_are_present()

        @patch("logging.Logger.warning")
        def test_check_if_indexcolumn_is_unique_warning(self, mock):
            #  check if indexcolumn is unique
            # check if error gets raised when duplicate
            self.obj.rawdata[self.obj.index_column] = "non unique"
            self.obj.check_if_indexcolumn_is_unique()
            mock.assert_called_once()

        # @patch("logging.Logger.warning")
        # def test_check_if_indexcolumn_is_unique_no_warning(self,mock):
        #  check if indexcolumn is unique
        # self.obj.check_if_indexcolumn_is_unique()
        # mock.assert_not_called()

        def test_check_if_file_exists(self):
            # check if error gets raised when file doesnt exist
            with self.assertRaises(OSError):
                wrong_file_path = "wrong/file/path"
                self.obj.check_if_file_exists(file=wrong_file_path)

        def test_add_contaminantion_column(self):
            column_added = "contamination_library" in self.obj.rawdata
            self.assertTrue(column_added)
            self.assertEqual(self.obj.rawdata["contamination_library"].dtype, "bool")

        def test_df_dimensions(self):
            # test if dataframe gets loaded correctly
            # are there as many rows and column we expect
            self.assertEqual(self.obj.rawdata.shape, self.df_dim)


class TestAlphaPeptLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = AlphaPeptLoader(file="testfiles/alphapept/results_proteins.csv")
        self.hdf_file = "testfiles/alphapept/results.hdf"
        # expected dim of rawdata df
        self.df_dim = (3781, 8)

    def test_load_hdf_protein_table(self):
        hdf_format = AlphaPeptLoader(file=self.hdf_file)
        # test if hdf file gets loaded
        self.assertEqual(hdf_format.rawdata.shape, self.df_dim)

    def test_add_contamination_reverse_column(self):
        # check if contamination column has been added
        column_added = "Reverse" in self.obj.rawdata
        self.assertTrue(column_added)
        # check if column contains True and False
        self.assertEqual(self.obj.rawdata.Reverse.dtype, "bool")

    def test_standardize_protein_group_column(self):
        # check if column ProteinGroup has been added
        column_added = "ProteinGroup" in self.obj.rawdata
        self.assertTrue(column_added)

        # test function with different entries
        entry_one = "sp|P0DMV9|HS71B_HUMAN,sp|P0DMV8|HS71A_HUMAN"
        entry_one_protein_id = self.obj.standardize_protein_group_column(
            entry=entry_one
        )
        self.assertEqual(entry_one_protein_id, "P0DMV9;P0DMV8")

        entry_two = "ENSEMBL:ENSBTAP00000007350"
        entry_two_protein_id = self.obj.standardize_protein_group_column(
            entry=entry_two
        )
        self.assertEqual(entry_two_protein_id, "ENSBTAP00000007350")


class TestMaxQuantLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = MaxQuantLoader(file="testfiles/maxquant/proteinGroups.txt")
        self.df_dim = (2611, 2531)

    def test_set_filter_columns_to_true_false(self):
        # check if + has been replaced by TRUE FALSE
        self.assertEqual(self.obj.rawdata["Reverse"].dtype, "bool")
        self.assertEqual(self.obj.rawdata["Only identified by site"].dtype, "bool")
        self.assertEqual(self.obj.rawdata["Potential contaminant"].dtype, "bool")


class TestDIANNLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = DIANNLoader(file="testfiles/diann/report_final.pg_matrix.tsv")
        self.df_dim = (10, 26)

    def add_tag_to_sample_columns(self):
        # get number of columns that have tag
        n_taged_samples = len(
            self.obj.rawdata.filter(like="_Intensity").columns.to_list()
        )
        self.assertEqual(n_taged_samples, 20)


class TestFragPipeLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = FragPipeLoader(file="testfiles/fragpipe/combined_proteins.tsv")
        self.df_dim = (10, 37)


if __name__ == "__main__":
    unittest.main()
