import unittest
import pandas as pd

from alphastats.loader.BaseLoader import BaseLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader


class BaseTestLoader:
    # parent class of test loader for common tests among loaders
    # this is wrapped in a nested class so it doesnt get called separatly when testing
    # plus to avoid multiple inheritance
    class BaseTest(unittest.TestCase):
        def test_dataformat(self):
            # check if loaded data is pandas dataframe
            self.assertIsInstance(self.obj.rawdata, pd.DataFrane)

        def test_check_if_columns_are_present(self):
            # check if columns are present
            # check if error gets raised when column is not present
            pass
            
        def test_check_if_indexcolumn_is_unique(self):
            # check if indexcolumn is unique
            # check if error gets raised when duplicate
            pass
    
        def test_check_if_file_exists(self, file):
            # check if error gets raised when file doesnt exist

            # # check if dataframe is loaded
            # check dimensions  
            # self.assertRaises(ValueError)
            pass

       

class TestAlphaPeptLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = AlphaPeptLoader(file="../testfiles/alphapept_results_proteins.csv")

    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        # are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 5)
        self.assertEqual(n_columns, 3781)

    def test_load_hdf_protein_table(self):
        # hdf_format = AlphaPeptLoader(file="../testfiles/alphapept_results.hdf")
        # test if hdf file gets loaded
        # n_rows = hdf_format.shape[0]
        # n_columns = hdf_format.shape[1]
        # self.assertEqual(n_rows, 5)
        # self.assertEqual(n_columns, 3781)
        pass

    def test_add_contamination_column(self):
         # check if contamination column has been added
        pass

    def test_standardize_protein_group_column(self):
        pass
    
        # check if contamination column has been added
        # # self.assertRaises(ValueError)


class TestMaxQuantLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = MaxQuantLoader(file="../testfiles/maxquant_proteinGroups.txt")
    
    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        # are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 2611)
        self.assertEqual(n_columns, 2530)
 
    def test_set_filter_columns_to_true_false(self):
        # check if + has been replaced by TRUE FALSE
        pass


class TestDIANNLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = DIANNLoader(file="../testfiles/diann_report_final.pg_matrix.tsv")

    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        # are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 10)
        self.assertEqual(n_columns, 25)


class TesFragPipeLoader(BaseTestLoader.BaseTest):
    def setUp(self):
        self.obj = FragPipeLoader(file="../testfiles/fragpipe_combined_proteins.tsv")
    
    def test_df_dimensions(self):
        # test if dataframe gets loaded correctly
        # are there as many rows and column we expect
        n_rows = self.obj.rawdata.shape[0]
        n_columns = self.obj.rawdata.shape[1]
        self.assertEqual(n_rows, 10)
        self.assertEqual(n_columns, 36)

       


if __name__ == '__main__':
    unittest.main()

