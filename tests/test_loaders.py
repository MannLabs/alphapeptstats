import unittest
import pandas as pd
import logging
from unittest.mock import patch
from requests import RequestException
import logging


from alphastats.loader.BaseLoader import BaseLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader

logger = logging.getLogger(__name__)

class BaseTestLoader:
    #  parent class of test loader for common tests among loaders
    # this is wrapped in a nested class so it doesnt get called separatly when testing
    # plus to avoid multiple inheritance
    class BaseTest(unittest.TestCase):
        def test_dataformat(self):
            # check if loaded data is pandas dataframe
            self.assertIsInstance(self.obj.rawdata, pd.DataFrame)

        @patch("requests.get")
        def test_check_if_columns_are_present_error(self, get_mock):
            get_mock.side_effect = RequestException
            with self.assertLogs() as captured:
                # check if columns are present
                # check if error gets raised when column is not present
                self.obj.confidence_column = "wrong_column"
                self.obj.check_if_columns_are_present()
            self.assertEqual(len(captured.records), 1)
            # self.assertEqual(captured.records[0].level, logging.ERROR)
            #self.assertRaises(KeyError, self.obj.check_if_columns_are_present())
        
        @patch("requests.get")
        def test_check_if_columns_are_present_no_error(self, get_mock):
            get_mock.side_effect = RequestException
            with self.assertLogs() as captured:
                # check if columns are present
                # check if error gets raised when column is not present
                self.obj.check_if_columns_are_present()
            self.assertEqual(captured.records, None)

        @patch("logging.Logger.error")
        def test_check_if_columns_are_present_no_error2(self, mock):
            self.obj.check_if_columns_are_present()
            #elf.assertEqual(captured.records, None)
            mock.assert_not_called()
           
        @patch("logging.Logger.warning")
        def test_check_if_indexcolumn_is_unique_warning(self,mock):
            #  check if indexcolumn is unique
            # check if error gets raised when duplicate
           # with self.assertLogs() as ctx:
            self.obj.rawdata[self.obj.index_column] = "non unique"
            self.obj.check_if_indexcolumn_is_unique()
                # check if one record = warning gets captured
                #self.assertEqual(len(ctx.records), 1)
            mock.assert_called_once()
            #self.assertWarns(self.obj.check_if_indexcolumn_is_unique())
            #pass
        
        @patch("logging.Logger.warning")
        def test_check_if_indexcolumn_is_unique_no_warning(self,mock):
            #  check if indexcolumn is unique
            # check if error gets raised when duplicate
           # with self.assertLogs() as ctx:
            self.obj.check_if_indexcolumn_is_unique()
                # check if one record = warning gets captured
                #self.assertEqual(len(ctx.records), 1)
            mock.assert_not_called()
            #self.assertWarns(self.obj.check_if_indexcolumn_is_unique())
            #pass

        @patch("requests.get")
        def test_check_if_file_exists(self, get_mock):
            # check if error gets raised when file doesnt exist
            get_mock.side_effect = RequestException
            with self.assertLogs() as captured:
                # check if columns are present
                # check if error gets raised when column is not present
                wrong_file_path = "wrong/file/path"
                self.obj.check_if_file_exists(file=wrong_file_path)
            self.assertEqual(len(captured.records), 1)
            #self.assertEqual(captured.records[0].level, logging.ERROR)
            #self.assertRaises(OSError, self.obj.check_if_file_exists(file=wrong_file_path))


class TestAlphaPeptLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = AlphaPeptLoader(file="testfiles/alphapept_results_proteins.csv")
        # self.hdf_file =""

    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        #  are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 3781)
        self.assertEqual(n_columns, 7)

    def test_load_hdf_protein_table(self):
        #  TODO get corresponding HDF file
        # hdf_format = AlphaPeptLoader(file=self.hdf_file)
        # test if hdf file gets loaded
        # n_rows = hdf_format.shape[0]
        # n_columns = hdf_format.shape[1]
        # self.assertEqual(n_rows, 5)
        # self.assertEqual(n_columns, 3781)
        pass

    def test_add_contamination_column(self):
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
        self.obj = MaxQuantLoader(file="testfiles/maxquant_proteinGroups.txt")

    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        #  are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 2611)
        self.assertEqual(n_columns, 2530)

    def test_set_filter_columns_to_true_false(self):
        # check if + has been replaced by TRUE FALSE
        self.assertEqual(self.obj.rawdata["Reverse"].dtype, "bool")
        self.assertEqual(self.obj.rawdata["Only identified by site"].dtype, "bool")
        self.assertEqual(self.obj.rawdata["Potential contaminant"].dtype, "bool")


class TestDIANNLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = DIANNLoader(file="testfiles/diann_report_final.pg_matrix.tsv")

    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        #  are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 10)
        self.assertEqual(n_columns, 25)


class TestFragPipeLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = FragPipeLoader(file="testfiles/fragpipe_combined_proteins.tsv")

    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        #  are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 10)
        self.assertEqual(n_columns, 36)


if __name__ == "__main__":
    unittest.main()
