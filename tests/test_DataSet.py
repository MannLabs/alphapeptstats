from calendar import c
from http.cookiejar import LoadError
from math import remainder

# from multiprocessing.sharedctypes import Value
from random import sample
import unittest
import pandas as pd
import logging
from unittest.mock import patch
import logging
import numpy as np
import pandas as pd
import plotly
from contextlib import contextmanager
import shutil
import os
import copy
import dictdiffer

# from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

from alphastats.loader.BaseLoader import BaseLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.SpectronautLoader import SpectronautLoader
from alphastats.loader.GenericLoader import GenericLoader
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
                self.loader.rawinput = pd.DataFrame()
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
            self.obj.sample = "wrong_sample_column"
            self.obj.load_metadata(file_path=path)
            mock.assert_called_once()

        @patch("logging.Logger.warning")
        def test_load_metadata_warning(self, mock):
            # is dataframe None and is warning produced
            file_path = "wrong/file.xxx"
            self.obj.load_metadata(file_path=file_path)
            mock.assert_called_once()

        def test_create_matrix(self):
            #  matrix dimensions
            self.assertEqual(self.obj.mat.shape, self.matrix_dim)
            # does the matrix only contain floats/integers and NAs
            is_dtype_numeric = list(
                set(list(map(pd.api.types.is_numeric_dtype, self.obj.mat.dtypes)))
            )
            self.assertEqual(is_dtype_numeric, [True])

        def test_preprocess_print_info(self):
            self.obj.preprocess_print_info()

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
            """
            Raises Error when method is not available for Normalization
            """
            with self.assertRaises(ValueError):
                self.obj.preprocess(normalization="wrong method")

        def test_preprocess_imputation_invalid_method(self):
            with self.assertRaises(ValueError):
                self.obj.preprocess(imputation="wrong method")

        def test_imputation_mean(self):
            self.obj.preprocess(imputation="mean")
            self.assertFalse(self.obj.mat.isna().values.any())

        def test_imputation_median(self):
            self.obj.preprocess(imputation="median")
            self.assertFalse(self.obj.mat.isna().values.any())

        def test_imputation_knn(self):
            self.obj.preprocess(imputation="knn")
            self.assertFalse(self.obj.mat.isna().values.any())

        def test_plot_sampledistribution_wrong_method(self):
            """
            Raises Error when method is not available for plotting Sampledistribution
            """
            with self.assertRaises(ValueError):
                self.obj.plot_sampledistribution(method="wrong_method")

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

        def test_reset_preprocessing(self):
            self.assertEqual(self.obj.mat.shape, self.matrix_dim)

            self.obj.preprocess(remove_contaminations=True)
            self.assertEqual(self.obj.mat.shape, self.matrix_dim_filtered)

            self.obj.reset_preprocessing()
            self.assertEqual(self.obj.mat.shape, self.matrix_dim)


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

    def test_dataset_without_metadata(self):
        obj = DataSet(loader=self.loader)
        self.assertEqual(obj.mat.shape[0], obj.metadata.shape[0])

    def test_load_metadata_fileformats(self):
        # test if different fileformats get loaded correctly
        metadata_path = "testfiles/alphapept/metadata.txt"
        self.obj.load_metadata(file_path=metadata_path)
        self.assertEqual(self.obj.metadata.shape, (2, 2))

        metadata_path = "testfiles/alphapept/metadata.tsv"
        self.obj.load_metadata(file_path=metadata_path)
        self.assertEqual(self.obj.metadata.shape, (2, 2))

        metadata_path = "testfiles/alphapept/metadata.csv"
        self.obj.load_metadata(file_path=metadata_path)
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
        #  is sample C removed
        self.assertEqual(self.obj.metadata.shape, (2, 2))
        mock.assert_called_once()

    def test_load_metadata_df(self):
        if self.metadata_path.endswith(".csv"):
            df = pd.read_csv(self.metadata_path)
        else:
            df = pd.read_excel(self.metadata_path)
        obj = DataSet(
            loader=self.loader,
            metadata_path=df,
            sample_column="sample",
        )
        self.assertIsInstance(obj.metadata, pd.DataFrame)
        self.assertFalse(obj.metadata.empty)

    def test_preprocess_remove_samples(self):
        sample_list = ["A"]
        self.obj.preprocess(remove_samples=sample_list)
        self.assertEqual(self.obj.mat.shape, (1, 3781))

    def test_preprocess_normalize_zscore(self):
        self.obj.mat = pd.DataFrame({"a": [2, 5, 4], "b": [5, 4, 4], "c": [0, 10, 8]})
        # zscore Normalization
        self.obj.preprocess(log2_transform=False, normalization="zscore")
        expected_mat = pd.DataFrame(
            {
                "a": [-0.162221, -0.508001, -0.707107],
                "b": [1.297771, -0.889001, -0.707107],
                "c": [-1.135550, 1.397001, 1.414214],
            }
        )
        pd._testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_normalize_quantile(self):
        self.obj.mat = pd.DataFrame({"a": [2, 5, 4], "b": [5, 4, 4], "c": [0, 10, 8]})
        # Quantile Normalization
        self.obj.preprocess(log2_transform=False, normalization="quantile")
        expected_mat = pd.DataFrame(
            {"a": [0.5, 0.5, 0.0], 
             "b": [1.0, 0.0, 0.0], 
             "c": [0.0, 1.0, 1.0]}
        )
        pd._testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_normalize_linear(self):
        # !!! normalizes by row and not by feature
        self.obj.mat = pd.DataFrame({"a": [2, 5, 4], "b": [5, 4, 4], "c": [0, 10, 8]})
        # Linear Normalization
        self.obj.preprocess(log2_transform=False, normalization="linear")
        expected_mat = pd.DataFrame(
            {
                "a": [0.37139068, 0.42107596, 0.40824829],
                "b": [0.92847669, 0.33686077, 0.40824829],
                "c": [0.0, 0.84215192, 0.81649658],
            }
        )
        pd._testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_normalize_vst(self):
        self.obj.mat = pd.DataFrame({"a": [2, 5, 4], "b": [5, 4, 4], "c": [0, 10, 8]})
        # Linear Normalization
        self.obj.preprocess(log2_transform=False, normalization="vst")
        expected_mat = pd.DataFrame(
            {
                "a": [-0.009526, -0.236399, -0.707107],
                "b": [	1.229480, -1.089313, -0.707107],
                "c": [-1.219954, 1.325712, 1.414214],
            }
        )
        pd._testing.assert_frame_equal(self.obj.mat.round(2), expected_mat.round(2))

    def test_preprocess_imputation_mean_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(log2_transform=False, imputation="mean")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd._testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_median_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(log2_transform=False, imputation="median")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd._testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_knn_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(log2_transform=False, imputation="knn")
        expected_mat = pd.DataFrame(
            {"a": [2.0, 3.0, 4.0], "b": [5.0, 4.0, 4.0], "c": [10.0, 10.0, 10.0]}
        )
        pd._testing.assert_frame_equal(self.obj.mat, expected_mat)

    def test_preprocess_imputation_randomforest_values(self):
        self.obj.mat = pd.DataFrame(
            {"a": [2, np.nan, 4], "b": [5, 4, 4], "c": [np.nan, 10, np.nan]}
        )
        self.obj.preprocess(log2_transform=False, imputation="randomforest")
        expected_mat = pd.DataFrame(
            {
                "a": [2.0, 3.0, 4.0],
                "b": [5.0, 4.0, 4.0],
                "c": [10.0, 10.0, 10.0],
            }
        )
        pd._testing.assert_frame_equal(self.obj.mat, expected_mat)

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
        self.obj.preprocess(log2_transform=False, imputation="knn")
        plot = self.obj.plot_clustermap()
        first_row = plot.data2d.iloc[0].to_list()
        expected = [487618.5371077078, 1293013.103298046]
        self.assertEqual(first_row, expected)

    def test_plot_clustermap_with_label_bar(self):
        self.obj.preprocess(log2_transform=False, imputation="knn")
        plot = self.obj.plot_clustermap(label_bar=self.comparison_column)
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
        self.matrix_dim = (312, 2596)
        self.matrix_dim_filtered = (312, 2397)
        self.comparison_column = "disease"

    def test_load_evidence_wrong_sample_names(self):
        with self.assertRaises(ValueError):
            loader = MaxQuantLoader(
                file="testfiles/maxquant/proteinGroups.txt",
                evidence_file="testfiles/maxquant_go/evidence.txt",
            )
            DataSet(
                loader=loader,
                metadata_path=self.metadata_path,
                sample_column="sample",
            )

    def test_plot_pca_group(self):
        pca_plot = self.obj.plot_pca(group=self.comparison_column)
        # 5 different disease
        self.assertEqual(len(pca_plot.to_plotly_json().get("data")), 5)

    def test_data_completeness(self):
        self.obj.preprocess(log2_transform=False, data_completeness=0.7)
        self.assertEqual(self.obj.mat.shape[1], 159)

    def test_plot_pca_circles(self):
        pca_plot = self.obj.plot_pca(group=self.comparison_column, circle=True)
        # are there 5 circles test_preprocess_imputation_randomforest_values - for each group
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

    def test_plot_volcano_with_grouplist(self):
        fig = self.obj.plot_volcano(
            method="ttest",
            group1=["1_31_C6", "1_32_C7", "1_57_E8"],
            group2=["1_71_F10", "1_73_F12"],
        )

    def test_plot_volcano_with_grouplist_wrong_names(self):
        with self.assertRaises(ValueError):
            self.obj.plot_volcano(
                method="ttest",
                group1=["wrong_sample_name", "1_42_D9", "1_57_E8"],
                group2=["1_71_F10", "1_73_F12"],
            )

    def test_plot_volcano_compare_preprocessing_modes(self):
        result_list = self.obj.plot_volcano(
            method="ttest",
            group1=["1_31_C6", "1_32_C7", "1_57_E8"],
            group2=["1_71_F10", "1_73_F12"],
            compare_preprocessing_modes=True,
        )
        self.assertEqual(len(result_list), 12)

    def test_preprocess_subset(self):
        self.obj.preprocess(subset=True, log2_transform=False)
        self.assertEqual(self.obj.mat.shape, (48, 1364))

    @patch.object(Statistics, "tukey_test")
    def test_anova_without_tukey(self, mock):
        anova_results = self.obj.anova(column="disease", protein_ids="all", tukey=False)
        self.assertEqual(anova_results["ANOVA_pvalue"][1], 0.4469688936240973)
        self.assertEqual(anova_results.shape, (2600, 2))
        # check if tukey isnt called
        mock.assert_not_called()

    def test_plot_intenstity_subgroup(self):
        plot = self.obj.plot_intensity(
            protein_id="K7ERI9;A0A024R0T8;P02654;K7EJI9;K7ELM9;K7EPF9;K7EKP1",
            group="disease",
            subgroups=["healthy", "liver cirrhosis"],
            add_significance=True,
        )
        plot_dict = plot.to_plotly_json()
        self.assertEqual(len(plot_dict.get("data")), 3)

    @patch("logging.Logger.warning")
    def test_plot_intenstity_subgroup_significance_warning(self, mock):
        import streamlit as st

        st.session_state["gene_to_prot_id"] = {}
        plot = self.obj.plot_intensity(
            protein_id="K7ERI9;A0A024R0T8;P02654;K7EJI9;K7ELM9;K7EPF9;K7EKP1",
            group="disease",
            add_significance=True,
        )
        plot_dict = plot.to_plotly_json()
        self.assertEqual(len(plot_dict.get("data")), 5)
        self.assertEqual(mock.call_count, 2)

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

    def test_tukey_test(self):
        protein_id = "K7ERI9;A0A024R0T8;P02654;K7EJI9;K7ELM9;K7EPF9;K7EKP1"
        tukey_df = self.obj.tukey_test(protein_id=protein_id, group="disease", df=None)
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
            column="disease",
            group1="healthy",
            group2="liver cirrhosis",
            method="ttest",
            labels=True,
            draw_line=False,
        )
        n_labels = len(plot.to_plotly_json().get("layout").get("annotations"))
        # self.assertTrue(n_labels > 20)

    def test_plot_volcano_wald(self):
        """
        Volcano Plot with wald test and list of samples
        """
        self.obj.preprocess(imputation="knn")
        self.obj.plot_volcano(
            group1=["1_31_C6", "1_32_C7", "1_33_C8"],
            group2=["1_78_G5", "1_77_G4", "1_76_G3"],
            method="ttest",
        )

        column_added = "_comparison_column" in self.obj.metadata.columns.to_list()
        self.assertTrue(column_added)

    def test_plot_volcano_sam(self):
        self.obj.preprocess(
            log2_transform=False, imputation="knn", normalization="zscore"
        )
        plot = self.obj.plot_volcano(
            column="disease",
            group1="type 2 diabetes mellitus",
            group2="type 2 diabetes mellitus|non-alcoholic fatty liver disease",
            method="sam",
            draw_line=True,
            perm=10,
        )

        # fdr lines get drawn
        line_1 = plot.to_plotly_json()["data"][-2].get("line").get("shape")
        line_2 = plot.to_plotly_json()["data"][-1].get("line").get("shape")

        self.assertEqual(line_1, "spline")
        self.assertEqual(line_2, "spline")

    def test_plot_volcano_list(self):
        self.obj.preprocess(imputation="mean")
        plot = self.obj.plot_volcano(
            method="ttest",
            group1=["1_31_C6", "1_32_C7", "1_57_E8"],
            group2=["1_71_F10", "1_73_F12"],
            color_list=self.obj.mat.columns.to_list()[0:20],
        )
        self.assertEqual(len(plot.to_plotly_json()["data"][0]["x"]), 20)

    def test_plot_clustermap_significant(self):
        import sys

        sys.setrecursionlimit(100000)
        self.obj.preprocess(imputation="knn")
        plot = self.obj.plot_clustermap(
            label_bar=self.comparison_column,
            only_significant=True,
            group=self.comparison_column,
            subgroups=["healthy", "liver cirrhosis"],
        )

    def test_plot_volcano_with_labels_proteins(self):
        # remove gene names
        self.obj.gene_names = None
        plot = self.obj.plot_volcano(
            column="disease",
            group1="healthy",
            group2="liver cirrhosis",
            method="ttest",
            labels=True,
        )
        n_labels = len(plot.to_plotly_json().get("layout").get("annotations"))

    def test_plot_volcano_with_labels_proteins_welch_ttest(self):
        # remove gene names
        self.obj.gene_names = None
        plot = self.obj.plot_volcano(
            column="disease",
            group1="healthy",
            group2="liver cirrhosis",
            method="welch-ttest",
            labels=True,
        )
        n_labels = len(plot.to_plotly_json().get("layout").get("annotations"))
        # self.assertTrue(n_labels > 20)

    def test_calculate_diff_exp_wrong(self):
        # get groups from comparison column
        with self.assertRaises(ValueError):
            self.obj.preprocess(imputation="knn")
            groups = list(set(self.obj.metadata[self.comparison_column].to_list()))
            group1, group2 = groups[0], groups[1]

            self.obj.diff_expression_analysis(
                column=self.comparison_column,
                group1=group1,
                group2=group2,
                method="wrong_method",
            )  # check if dataframe gets created

    def test_diff_expression_analysis_nocolumn(self):
        with self.assertRaises(ValueError):
            self.obj.diff_expression_analysis(
                group1="healthy", group2="liver cirrhosis"
            )

    def test_diff_expression_analysis_list(self):
        self.obj.diff_expression_analysis(
            group1=["1_31_C6", "1_32_C7", "1_33_C8"],
            group2=["1_78_G5", "1_77_G4", "1_76_G3"],
            method="ttest",
        )

        column_added = "_comparison_column" in self.obj.metadata.columns.to_list()
        self.assertTrue(column_added)

    def test_plot_intensity_non_sign(self):
        """
        No significant label is added to intensity plot
        """
        plot = self.obj.plot_intensity(
            protein_id="S6BAR0",
            group="disease",
            subgroups=["liver cirrhosis", "healthy"],
            add_significance=True,
        )

        annotation = (
            plot.to_plotly_json().get("layout").get("annotations")[1].get("text")
        )
        self.assertEqual(annotation, "-")

    def test_plot_intensity_sign(self):
        """
        Significant label * is added to intensity plot
        """
        plot = self.obj.plot_intensity(
            protein_id="Q9UL94",
            group="disease",
            subgroups=["liver cirrhosis", "healthy"],
            add_significance=True,
        )

        annotation = (
            plot.to_plotly_json().get("layout").get("annotations")[1].get("text")
        )
        self.assertEqual(annotation, "*")

    def test_plot_intensity_sign_01(self):
        """
        Significant label ** is added to intensity plot
        """
        plot = self.obj.plot_intensity(
            protein_id="Q96JD0;Q96JD1;P01721",
            group="disease",
            subgroups=["liver cirrhosis", "healthy"],
            add_significance=True,
        )

        annotation = (
            plot.to_plotly_json().get("layout").get("annotations")[1].get("text")
        )
        self.assertEqual(annotation, "**")

    def test_plot_intensity_sign_001(self):
        """
        Highly significant label is added to intensity plot
        """
        plot = self.obj.plot_intensity(
            protein_id="Q9BWP8",
            group="disease",
            subgroups=["liver cirrhosis", "healthy"],
            add_significance=True,
        )

        annotation = (
            plot.to_plotly_json().get("layout").get("annotations")[1].get("text")
        )
        self.assertEqual(annotation, "***")

    def test_plot_intensity_all(self):
        plot = self.obj.plot_intensity(
            protein_id="Q9BWP8",
            group="disease",
            subgroups=["liver cirrhosis", "healthy"],
            method="all",
            add_significance=True,
        )
        self.assertEqual(plot.to_plotly_json()["data"][0]["points"], "all")

    def test_plot_samplehistograms(self):
        fig = self.obj.plot_samplehistograms().to_plotly_json()
        self.assertEqual(312, len(fig["data"]))

    def test_batch_correction(self):
        self.obj.preprocess(subset=True, imputation="knn", normalization="linear")
        self.obj.batch_correction(batch="batch_artifical_added")
        first_value = self.obj.mat.values[0, 0]
        self.assertAlmostEqual(-0.00555, first_value, places=3)

    def test_multicova_analysis_invalid_covariates(self):
        self.obj.preprocess(imputation="knn", normalization="zscore", subset=True)
        res, _ = self.obj.multicova_analysis(
            covariates=[
                "disease",
                "Alkaline phosphatase measurement",
                "Body mass index ",
                "not here",
            ],
            subset={"disease": ["healthy", "liver cirrhosis"]},
        )
        self.assertEqual(res.shape[1], 45)

    # def test_perform_gsea(self):
    #     df = self.obj.perform_gsea(column="disease",
    #                             group1="healthy",
    #                                     group2="liver cirrhosis",
    #                                     gene_sets= 'KEGG_2019_Human')

    #     cholersterol_enhanced = 'Cholesterol metabolism' in df.index.to_list()
    #     self.assertTrue(cholersterol_enhanced)


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
            self.obj.plot_intensity(
                protein_id="A0A075B6H7", group="grouping1", method="wrong"
            )

    def test_plot_clustermap_noimputation(self):
        # raises error when data is not imputed
        with self.assertRaises(ValueError):
            self.obj.plot_clustermap()

    def test_plot_dendrogram(self):
        self.obj.preprocess(imputation="mean")
        fig = self.obj.plot_dendrogram()

    def test_plot_tsne(self):
        plot_dict = self.obj.plot_tsne().to_plotly_json()
        # check if everything get plotted
        self.assertEqual(len(plot_dict.get("data")[0].get("x")), 20)

    def test_plot_dendrogram_navalues(self):
        with self.assertRaises(ValueError):
            self.obj.plot_dendrogram()

    def test_plot_dendrogram_not_imputed(self):
        with self.assertRaises(ValueError):
            self.obj.plot_dendrogram()

    def test_volcano_plot_anova(self):
        self.obj.preprocess(imputation="knn")
        plot = self.obj.plot_volcano(
            column="grouping1", group1="Healthy", group2="Disease", method="anova"
        )
        expected_y_value = 0.040890177695653236
        y_value = plot.to_plotly_json().get("data")[0].get("y")[1]
        self.assertAlmostEqual(y_value, expected_y_value)

    def test_volcano_plot_ttest_no_column(self):
        with self.assertRaises(ValueError):
            self.obj.preprocess(imputation="knn")
            self.obj.plot_volcano(group1="Healthy", group2="Disease", method="ttest")

    def test_volcano_plot_wrongmethod(self):
        with self.assertRaises(ValueError):
            self.obj.plot_volcano(
                column="grouping1",
                group1="Healthy",
                group2="Disease",
                method="wrongmethod",
            )

    # def test_diff_expression_analysis_with_list(self):
    #     self.obj.preprocess(imputation="knn")
    #     column="grouping1"
    #     group1="Healthy"
    #     group2="Disease"
    #     group1_samples = self.obj.metadata[self.obj.metadata[column] == group1][
    #             "sample"
    #         ].tolist()
    #     group2_samples = self.obj.metadata[self.obj.metadata[column] == group2][
    #             "sample"
    #         ].tolist()
    #     self.obj.diff_expression_analysis(
    #         group1=group1_samples,
    #         group2=group2_samples)


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
        self.matrix_dim = (20, 6)
        self.matrix_dim_filtered = (20, 6)
        self.comparison_column = "grouping1"


class TestSpectronautDataSet(BaseTestDataSet.BaseTest):
    @classmethod
    def setUpClass(cls):

        if os.path.isfile("testfiles/spectronaut/results.tsv") == False:
            shutil.unpack_archive(
                "testfiles/spectronaut/results.tsv.zip", "testfiles/spectronaut/"
            )

        cls.cls_loader = SpectronautLoader(
            file="testfiles/spectronaut/results.tsv", filter_qvalue=False
        )
        cls.cls_metadata_path = "testfiles/spectronaut/metadata.xlsx"
        cls.cls_obj = DataSet(
            loader=cls.cls_loader,
            metadata_path=cls.cls_metadata_path,
            sample_column="sample",
        )

    def setUp(self):
        self.loader = copy.deepcopy(self.cls_loader)
        self.metadata_path = copy.deepcopy(self.cls_metadata_path)
        self.obj = copy.deepcopy(self.cls_obj)
        self.matrix_dim = (9, 2458)
        self.matrix_dim_filtered = (9, 2453)
        self.comparison_column = "condition"

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("testfiles/spectronaut/__MACOSX"):
            shutil.rmtree("testfiles/spectronaut/__MACOSX")

        os.remove("testfiles/spectronaut/results.tsv")

class TestGenericDataSet(BaseTestDataSet.BaseTest):
    @classmethod
    def setUpClass(cls):
        if os.path.isfile("testfiles/fragpipe/combined_proteins.tsv") == False:
            shutil.unpack_archive(
                "testfiles/fragpipe/combined_proteins.tsv.zip", "testfiles/fragpipe"
            )

        cls.cls_loader = GenericLoader(
            file="testfiles/fragpipe/combined_proteins.tsv",
            intensity_column=[
                "S1 Razor Intensity",	"S2 Razor Intensity", "S3 Razor Intensity",
                "S4 Razor Intensity",	"S5 Razor Intensity",	"S6 Razor Intensity",
                "S7 Razor Intensity", "S8 Razor Intensity"
            ],
            index_column="Protein",
            sep="\t"
        )
        cls.cls_metadata_path = "testfiles/fragpipe/metadata2.xlsx"
        cls.cls_obj = DataSet(
            loader=cls.cls_loader,
            metadata_path=cls.cls_metadata_path,
            sample_column="analytical_sample external_id",
        )

    def setUp(self):
        self.loader = copy.deepcopy(self.cls_loader)
        self.metadata_path = copy.deepcopy(self.cls_metadata_path)
        self.obj = copy.deepcopy(self.cls_obj)
        self.matrix_dim = (8, 5)
        self.matrix_dim_filtered = (8, 5)
        self.comparison_column = "grouping1"

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir("testfiles/fragpipe/__MACOSX"):
            shutil.rmtree("testfiles/fragpipe/__MACOSX")


if __name__ == "__main__":
    unittest.main()
