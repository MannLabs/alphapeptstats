import unittest
import pandas as pd
import numpy as np

from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.DataSet import DataSet


class BaseTestDataSet:
    # parent class of test loader for common tests among loaders
    # this is wrapped in a nested class so it doesnt get called separatly when testing
    # plus to avoid multiple inheritance
    class BaseTest(unittest.TestCase):
        def test_go_abundance_corretion(self):
            df = self.obj.go_abundance_correction(
                fg_sample=self.fg_sample, bg_sample=self.bg_sample
            )
            self.assertFalse(df.empty)

        def test_plot_scatter(self):
            df = self.obj.go_abundance_correction(
                fg_sample=self.fg_sample, bg_sample=self.bg_sample
            )
            plot_dict = df.plot_scatter().to_plotly_json()
            # colored in 4 different categories but could change when DB are updated
            self.assertTrue(len(plot_dict.get("data")) > 4)

        def test_plot_bar(self):
            df = self.obj.go_abundance_correction(
                fg_sample=self.fg_sample, bg_sample=self.bg_sample
            )
            plot = df.plot_scatter()

        def test_go_characterize_foreground(self):
            df = self.obj.go_characterize_foreground(
                tax_id=9606, protein_list=self.obj.mat.columns.to_list()[600:700]
            )
            self.assertFalse(df.empty)

        def test_go_compare_samples(self):
            df = self.obj.go_compare_samples(
                fg_sample=self.fg_sample, bg_sample=self.bg_sample
            )
            self.assertTrue(df.empty)

        def test_raise_error_no_evidence(self):
            with self.assertRaises(ValueError):
                self.obj.evidence_df = None
                self.obj.go_abundance_correction(
                    fg_sample=self.fg_sample, bg_sample=self.bg_sample
                )

        def test_go_abundance_correction_with_list(self):
            df = self.obj.go_abundance_correction(
                bg_sample=self.bg_sample,
                fg_protein_list=self.obj.mat.columns.to_list()[200:300],
            )
            self.assertTrue(df.empty)

        def test_go_genome_list(self):
            df = self.obj.go_genome(
                protein_list=self.obj.mat.columns.to_list()[600:700]
            )
            self.assertFalse(df.empty)

        def test_go_genome_sample(self):
            df = self.obj.go_genome(fg_sample=self.fg_sample)
            self.assertFalse(df.empty)

        def test_extract_protein_ids(self):
            # test function with different entries
            entry_one = "sp|P0DMV9|HS71B_HUMAN,sp|P0DMV8|HS71A_HUMAN"
            entry_one_protein_id = self.obj._extract_protein_ids(entry=entry_one)
            self.assertEqual(entry_one_protein_id, "P0DMV9;P0DMV8")

            entry_two = "ENSEMBL:ENSBTAP00000007350"
            entry_two_protein_id = self.obj._extract_protein_ids(entry=entry_two)
            self.assertEqual(entry_two_protein_id, "ENSBTAP00000007350")


class TestMaxQuantGODataSet(BaseTestDataSet.BaseTest):
    def setUp(self):
        self.loader = MaxQuantLoader(
            file="testfiles/maxquant_go/proteinGroups.txt",
            evidence_file="testfiles/maxquant_go/evidence.txt",
        )
        evidence_df = pd.read_csv("testfiles/maxquant_go/evidence.txt", sep="\t")
        metadata = pd.DataFrame({"sample": evidence_df["Raw file"].unique().tolist()})
        metadata["experiment"] = np.where(
            metadata["sample"].str.startswith("AC"), "A", "U"
        )

        self.obj = DataSet(
            loader=self.loader, metadata_path=metadata, sample_column="sample",
        )
        self.fg_sample = "AC399"
        self.bg_sample = "UT822"


if __name__ == "__main__":
    unittest.main()
