from calendar import c
from http.cookiejar import LoadError
from math import remainder

# from multiprocessing.sharedctypes import Value
from random import sample
import unittest
from xml.sax.handler import property_interning_dict
import pandas as pd
import logging
from unittest.mock import patch
import logging
import numpy as np
import pandas as pd
import plotly
from contextlib import contextmanager

# from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

from alphastats.loader.BaseLoader import BaseLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.DataSet import DataSet

from alphastats.DataSet_Statistics import Statistics
from alphastats.DataSet_Plot import Plot
from alphastats.utils import LoaderError


logger = logging.getLogger(__name__)


class BaseTestDataSet:
    # parent class of test loader for common tests among loaders
    # this is wrapped in a nested class so it doesnt get called separatly when testing
    # plus to avoid multiple inheritance
    class BaseTest(unittest.TestCase):
        @contextmanager
        def assertNotRaises(self, exc_type):
            try:
                yield None
            except exc_type:
                raise self.failureException("{} raised".format(exc_type.__name__))

        def test_check_loader_no_error(self):
            with self.assertNotRaises(ValueError):
                self.obj._check_loader(loader=self.loader)

        def test_check_loader_error_invalid_column(self):
            #  invalid index column
            with self.assertRaises(ValueError):
                self.loader.index_column = 100
                self.obj._check_loader(loader=self.loader)

        def test_check_loader_error_empty_df(self):
            # empty dataframe
            with self.assertRaises(ValueError):
                self.loader.rawdata = pd.DataFrame()
                self.obj._check_loader(loader=self.loader)

        def test_check_loader_error_invalid_loader(self):
            #  invalid loader, class
            with self.assertRaises(LoaderError):
                df = pd.DataFrame()
                self.obj._check_loader(loader=df)

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

        @patch("logging.Logger.warning")
        def test_check_values_warning(self, mock):
            # is dataframe None and is warning produced
            data = {
                "A": [10, 11, 12, 13, 14],
                "B": [23, 22, 24, 22, 25],
                "C": [66, 72, np.inf, 68, -np.inf],
            }
            self.obj.mat = pd.DataFrame(data)
            self.obj._check_matrix_values()
            mock.assert_called_once()

        @patch("logging.Logger.info")
        def test_preprocess_filter(self, mock):
            # is info printed if contamination columns get removed
            # is the new matrix smaller than the older matrix
            self.obj.preprocess(remove_contaminations=True)
            self.assertEqual(self.obj.mat.shape, self.matrix_dim_filtered)
            #  info has been printed at least once
            mock.assert_called_once()

        @patch("logging.Logger.info")
        def test_preprocess_filter_already_filter(self, mock):
            # is warning raised when filter columns are none
            # is info printed if contamination columns get removed
            # is the new matrix smaller than the older matrix
            self.assertFalse(
                self.obj.preprocessing_info.get("Contaminations have been removed")
            )
            self.obj.preprocess(remove_contaminations=True)
            self.assertEqual(self.obj.mat.shape, self.matrix_dim_filtered)
            self.assertTrue(
                self.obj.preprocessing_info.get("Contaminations have been removed")
            )
            self.obj.preprocess(remove_contaminations=True)
            self.assertEqual(self.obj.mat.shape, self.matrix_dim_filtered)

        @patch("logging.Logger.info")
        def test_preprocess_filter_no_filter_columns(self, mock):
            self.obj.filter_columns = []
            self.obj.preprocess(remove_contaminations=True)
            mock.assert_called_once()

        def test_preprocess_normalization_invalid_method(self):
            with self.assertRaises(ValueError):
                self.obj.preprocess(normalization="wrong method")

        def test_preprocess_imputation_invalid_method(self):
            with self.assertRaises(ValueError):
                self.obj.preprocess(imputation="wrong method")

        def test_calculate_ttest_fc(self):
            # get groups from comparison column
            groups = list(set(self.obj.metadata[self.comparison_column].to_list()))
            group1, group2 = groups[0], groups[1]
            if self.obj.software != "AlphaPept":
                df = self.obj.calculate_ttest_fc(
                    column=self.comparison_column, group1=group1, group2=group2
                )  # check if dataframe gets created
                self.assertTrue(isinstance(df, pd.DataFrame))
                self.assertFalse(df.empty)
            else:
                with self.assertRaises(NotImplementedError):
                    # alphapept has only two samples should throw error
                    self.obj.calculate_ttest_fc(
                        column=self.comparison_column, group1=group1, group2=group2
                    )

        def test_imputation_mean(self):
            self.obj.preprocess(imputation="mean")
            self.assertFalse(self.obj.mat.isna().values.any())

        def test_imputation_median(self):
            self.obj.preprocess(imputation="median")
            self.assertFalse(self.obj.mat.isna().values.any())

        def test_imputation_knn(self):
            self.obj.preprocess(imputation="knn")
            self.assertFalse(self.obj.mat.isna().values.any())

        def test_plot_sampledistribution(self):
            plot = self.obj.plot_sampledistribution(log_scale=True)
            # check if it is a figure
            self.assertIsInstance(plot, plotly.graph_objects.Figure)
            # convert plotly objec to dict
            plot_dict = plot.to_plotly_json()
            # check if plotly object is not empty
            self.assertEqual(len(plot_dict.get("data")), 1)
            #  check if it is logscale
            self.assertEqual(plot_dict.get("layout").get("yaxis").get("type"), "log")


class TestAlphaPeptDataSet(BaseTestDataSet.BaseTest):
    #  do testing which requires extra files only on TestAlphaPeptDataSet
    # to reduce the amount of compariosn files required
    def setUp(self):
        self.loader = AlphaPeptLoader(file="testfiles/alphapept/results_proteins.csv")
        self.metadata_path = "testfiles/alphapept/metadata.csv"
        self.obj = DataSet(
            loader=self.loader,
            metadata_path=self.metadata_path,
            sample_column="sample",
        )
        # expected dimensions of matrix
        self.matrix_dim = (2, 3781)
        self.matrix_dim_filtered = (2, 3707)
        #  metadata column to compare for PCA, t-test, etc.
        self.comparison_column = "disease"

    def test_load_metadata_fileformats(self):
        # test if different fileformats get loaded correctly
        metadata_path = "testfiles/alphapept/metadata.txt"
        self.obj.load_metadata(file_path=metadata_path, sample_column="sample")
        self.assertEqual(self.obj.metadata.shape, (2, 2))

        metadata_path = "testfiles/alphapept/metadata.tsv"
        self.obj.load_metadata(file_path=metadata_path, sample_column="sample")
        self.assertEqual(self.obj.metadata.shape, (2, 2))

        metadata_path = "testfiles/alphapept/metadata.csv"
        self.obj.load_metadata(file_path=metadata_path, sample_column="sample")
        self.assertEqual(self.obj.metadata.shape, (2, 2))
    
    @patch("logging.Logger.warning")
    def test_remove_misc_samples_in_metadata(self, mock):
        df = pd.DataFrame(
            {"sample": ["A", "B", "C"], "b": ["disease", "health", "disease"]}
        )
        obj = DataSet(
            loader=self.loader,
            metadata_path=df,
            sample_column="sample",
        )
        # is sample C removed
        self.assertEqual(self.obj.metadata.shape, (2, 2))
        mock.assert_called_once()

    def test_preprocess_remove_samples(self):
        sample_list = ["A"]
        self.obj.preprocess(remove_samples=sample_list)
        self.assertEqual(self.obj.mat.shape, (1, 3781))

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

    def test_preprocess_imputation_mean_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(imputation="mean")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_median_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(imputation="median")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_knn_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(imputation="knn")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_randomforest_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(imputation="randomforest")
        expected_mat = pd.DataFrame(
            {
                "a": [2.00000000e00, -9.22337204e12, 4.00000000e00],
                "b": [5.00000000e00, 4.00000000e00, 4.0],
                "c": [-9.22337204e12, 1.00000000e01, -9.22337204e12],
            }
        )
        pd.util.testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_plot_sampledistribution_group(self):
        plot = self.obj.plot_sampledistribution(
            method="box", color="disease", log_scale=False
        )
        # check if it is a figure
        self.assertIsInstance(plot, plotly.graph_objects.Figure)
        # convert plotly object to dict
        plot_dict = plot.to_plotly_json()
        #  check if it doesnt get transformed to logscale
        self.assertEqual(plot_dict.get("layout").get("yaxis").get("type"), None)
        # check if there are two groups control and disease
        self.assertEqual(plot_dict.get("data")[0].get("legendgroup"), "control")
        #  check that it is boxplot and not violinplot
        is_boxplot = "boxmode" in plot_dict.get("layout").keys()
        self.assertTrue(is_boxplot)

    def test_plot_correlation_matrix(self):
        plot = self.obj.plot_correlation_matrix()
        plot_dict = plot.to_plotly_json()
        correlation_calculations_expected = [1.0, 0.999410773629427]
        self.assertEqual(
            plot_dict.get("data")[0].get("z")[0].tolist(),
            correlation_calculations_expected,
        )

    def test_plot_clustermap(self):
        self.obj.preprocess(imputation="knn")
        plot = self.obj.plot_clustermap()
        first_row = plot.data2d.iloc[0].to_list()
        expected = [487618.5371077078, 1293013.103298046]
        self.assertEqual(first_row, expected)

    def test_plot_clustermap_with_label_bar(self):
        self.obj.preprocess(imputation="knn")
        plot = self.obj.plot_clustermap(label_bar=[self.comparison_column])
        first_row = plot.data2d.iloc[0].to_list()
        expected = [487618.5371077078, 1293013.103298046]
        self.assertEqual(first_row, expected)



class TestMaxQuantDataSet(BaseTestDataSet.BaseTest):
    def setUp(self):
        self.loader = MaxQuantLoader(file="testfiles/maxquant/proteinGroups.txt")
        self.metadata_path = "testfiles/maxquant/metadata.xlsx"
        self.obj = DataSet(
            loader=self.loader,
            metadata_path=self.metadata_path,
            sample_column="sample",
        )
        # expected dimensions of matrix
        self.matrix_dim = (312, 2611)
        self.matrix_dim_filtered = (312, 2409)
        self.comparison_column = "disease"

    def test_plot_pca_group(self):
        pca_plot = self.obj.plot_pca(group=self.comparison_column)
        # 5 different disease
        self.assertEqual(len(pca_plot.to_plotly_json().get("data")), 5)

    def test_plot_pca_circles(self):
        pca_plot = self.obj.plot_pca(group=self.comparison_column, circle=True)
        # are there 5 circles drawn - for each group
        number_of_groups = len(pca_plot.to_plotly_json().get("layout").get("shapes"))
        self.assertEqual(number_of_groups, 5)

    def test_plot_umap_group(self):
        umap_plot = self.obj.plot_umap(group=self.comparison_column)
        # 5 different disease
        self.assertEqual(len(umap_plot.to_plotly_json().get("data")), 5)

    def test_plot_umap_circles(self):
        umap_plot = self.obj.plot_umap(group=self.comparison_column, circle=True)
        # are there 5 circles drawn - for each group
        number_of_groups = len(umap_plot.to_plotly_json().get("layout").get("shapes"))
        self.assertEqual(number_of_groups, 5)

    def test_preprocess_subset(self):
        df = self.obj._subset()
        self.assertEqual(df.shape, (48, 2611))

    @patch.object(Statistics, "calculate_tukey")
    def test_anova_without_tukey(self, mock):
        anova_results = self.obj.anova(column="disease", protein_ids="all", tukey=False)
        self.assertEqual(anova_results["ANOVA_pvalue"][1], 0.4469688936240973)
        self.assertEqual(anova_results.shape, (2615, 2))
        # check if tukey isnt called
        mock.assert_not_called()

    def test_plot_intenstity_subgroup(self):
        plot = self.obj.plot_intensity(protein_id="K7ERI9;A0A024R0T8;P02654;K7EJI9;K7ELM9;K7EPF9;K7EKP1", group="disease",subgroups=["healthy", "liver cirrhosis"], add_significance=True)
        plot_dict = plot.to_plotly_json()
        self.assertEqual(len(plot_dict.get("data")), 2)

    @patch("logging.Logger.warning")
    def test_plot_intenstity_subgroup_significance_warning(self, mock):
        plot = self.obj.plot_intensity(protein_id="K7ERI9;A0A024R0T8;P02654;K7EJI9;K7ELM9;K7EPF9;K7EKP1", group="disease", add_significance=True)
        plot_dict = plot.to_plotly_json()
        self.assertEqual(len(plot_dict.get("data")), 2)
        mock.assert_called_once()

    def test_anova_with_tukey(self):
        # with first 100 protein ids
        self.obj.preprocess(imputation="mean")
        id_list = self.obj.mat.columns.tolist()[0:100]
        results = self.obj.anova(column="disease", protein_ids=id_list, tukey=True)
        self.assertEqual(results.shape, (100, 10))

        # with one protein id
        protein_id = "A0A024R4J8;Q92876"
        results = self.obj.anova(column="disease", protein_ids=protein_id, tukey=True)
        self.assertEqual(results.shape[1], 10)

    def test_calculate_tukey(self):
        protein_id = "K7ERI9;A0A024R0T8;P02654;K7EJI9;K7ELM9;K7EPF9;K7EKP1"
        tukey_df = self.obj.calculate_tukey(
            protein_id=protein_id, group="disease", df=None
        )
        self.assertEqual(tukey_df["p-tukey"][0], 0.674989009816342)

    def test_ancova(self):
        ancova_df = self.obj.ancova(
            protein_id="K7ERI9;A0A024R0T8;P02654;K7EJI9;K7ELM9;K7EPF9;K7EKP1",
            covar="Triglycerides measurement (14740000)",
            between="disease",
        )
        expected_value = 0.7375624497867097
        given_value = ancova_df["p-unc"][1]
        decimal_places = 7
        self.assertAlmostEqual(expected_value, given_value, decimal_places)

    def test_plot_volcano_with_labels(self):
        plot = self.obj.plot_volcano(
            column="disease", group1="healthy", group2="liver cirrhosis", method="ttest", labels=True
        )
        n_labels = len(plot.to_plotly_json().get("layout").get("annotations"))
        self.assertTrue(n_labels > 20)

    def test_plot_volcano_with_labels_proteins(self):
        # remove gene names
        self.obj.gene_names = None
        plot = self.obj.plot_volcano(
            column="disease", group1="healthy", group2="liver cirrhosis", method="ttest", labels=True
        )
        n_labels = len(plot.to_plotly_json().get("layout").get("annotations"))
        self.assertTrue(n_labels > 20)

class TestDIANNDataSet(BaseTestDataSet.BaseTest):
    def setUp(self):
        self.loader = DIANNLoader(file="testfiles/diann/report_final.pg_matrix.tsv")
        self.metadata_path = "testfiles/diann/metadata.xlsx"
        self.obj = DataSet(
            loader=self.loader,
            metadata_path=self.metadata_path,
            sample_column="analytical_sample external_id",
        )
        # expected dimensions of matrix
        self.matrix_dim = (20, 10)
        self.matrix_dim_filtered = (20, 10)
        self.comparison_column = "grouping1"

    def test_plot_intensity_violin(self):
        # Violinplot
        plot = self.obj.plot_intensity(
            protein_id="A0A075B6H7", group="grouping1", method="violin"
        )
        plot_dict = plot.to_plotly_json()
        self.assertIsInstance(plot, plotly.graph_objects.Figure)
        # are two groups plotted
        self.assertEqual(len(plot_dict.get("data")), 2)

    def test_plot_intensity_box(self):
        # Boxplot
        plot = self.obj.plot_intensity(
            protein_id="A0A075B6H7", group="grouping1", method="box", log_scale=True
        )
        plot_dict = plot.to_plotly_json()
        #  log scale
        self.assertEqual(plot_dict.get("layout").get("yaxis").get("type"), "log")
        is_boxplot = "boxmode" in plot_dict.get("layout").keys()
        self.assertTrue(is_boxplot)

    def test_plot_intensity_scatter(self):
        # Scatterplot
        plot = self.obj.plot_intensity(
            protein_id="A0A075B6H7", group="grouping1", method="scatter"
        )
        plot_dict = plot.to_plotly_json()
        self.assertIsInstance(plot, plotly.graph_objects.Figure)
        # are two groups plotted
        self.assertEqual(plot_dict.get("data")[0].get("type"), "scatter")

    def test_plot_intensity_wrong_method(self):
        with self.assertRaises(ValueError):
            self.obj.plot_intensity(protein_id="A0A075B6H7", group="grouping1", method="wrong")

    def test_plot_clustermap_noimputation(self):
        # raises error when data is not imputed
        with self.assertRaises(ValueError):
            self.obj.plot_clustermap()

    def test_plot_dendogram(self):
        self.obj.preprocess(imputation="mean")
        fig = self.obj.plot_dendogram()

    def test_plot_tsne(self):
        plot_dict = self.obj.plot_tsne().to_plotly_json()
        # check if everything get plotted
        self.assertEqual(len(plot_dict.get("data")[0].get("x")), 20)

    def test_plot_dendogram_navalues(self):
        with self.assertRaises(ValueError):
            self.obj.plot_dendogram()

    def test_plot_dendogram_not_imputed(self):
        with self.assertRaises(ValueError):
            self.obj.plot_dendogram()

    def test_volcano_plot_anova(self):
        self.obj.preprocess(imputation="knn")
        plot = self.obj.plot_volcano(
            column="grouping1", group1="Healthy", group2="Disease", method="anova"
        )
        expected_y_value = 0.09437708068494619
        y_value = plot.to_plotly_json().get("data")[0].get("y")[1]
        self.assertAlmostEqual(y_value, expected_y_value)

    def test_volcano_plot_ttest(self):
        self.obj.preprocess(imputation="knn")
        plot = self.obj.plot_volcano(
            column="grouping1", group1="Healthy", group2="Disease", method="ttest"
        )
        y_value = plot.to_plotly_json().get("data")[0].get("y")[1]
        self.assertAlmostEqual(round(y_value, 1), 0.1)

    def test_volcano_plot_wrongmethod(self):
        with self.assertRaises(ValueError):
            self.obj.plot_volcano(
                column="grouping1",
                group1="Healthy",
                group2="Disease",
                method="wrongmethod",
            )

class TestFragPipeDataSet(BaseTestDataSet.BaseTest):
    def setUp(self):
        self.loader = FragPipeLoader(
            file="testfiles/fragpipe/combined_proteins.tsv",
            intensity_column="[sample] Razor Intensity",
        )
        self.metadata_path = "testfiles/fragpipe/metadata.xlsx"
        self.obj = DataSet(
            loader=self.loader,
            metadata_path=self.metadata_path,
            sample_column="analytical_sample external_id",
        )
        # expected dimensions of matrix
        self.matrix_dim = (20, 10)
        self.matrix_dim_filtered = (20, 10)
        self.comparison_column = "grouping1"


if __name__ == "__main__":
    unittest.main()
