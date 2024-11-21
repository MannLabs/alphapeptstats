import logging
import unittest
from unittest.mock import MagicMock, patch

from alphastats.dataset.dataset import DataSet
from alphastats.llm.uniprot_utils import (
    extract_data,
    get_uniprot_data,
    select_uniprot_id_from_feature,
)
from alphastats.loader.maxquant_loader import MaxQuantLoader

logger = logging.getLogger(__name__)


class TestGPT(unittest.TestCase):
    def setUp(self):
        self.loader = MaxQuantLoader(file="testfiles/maxquant/proteinGroups.txt")
        self.metadata_path = "testfiles/maxquant/metadata.xlsx"
        self.obj = DataSet(
            loader=self.loader,
            metadata_path_or_df=self.metadata_path,
            sample_column="sample",
        )
        # expected dimensions of matrix
        self.matrix_dim = (312, 2596)
        self.matrix_dim_filtered = (312, 2397)
        self.comparison_column = "disease"


class TestGetUniProtData(unittest.TestCase):
    @patch("requests.get")
    def test_get_uniprot_data_success(self, mock_get):
        # Set up the mock to return a successful response with example data
        example_response = {
            "results": [
                {
                    "protein_name": "P12345",
                    "gene_names": "test_gene",
                    "cc_subcellular_location": "at home",
                },
            ]
        }
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: example_response
        )

        expected_result = {
            "protein_name": "P12345",
            "gene_names": "test_gene",
            "cc_subcellular_location": "at home",
        }
        result = get_uniprot_data("test_gene", "9606")["results"][0]

        # Verify that the result matches the expected result
        self.assertEqual(result, expected_result)
        # Verify that requests.get was called with the correct parameters
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_get_uniprot_data_failure(self, mock_get):
        # Set up the mock to return a failed response
        mock_get.return_value = MagicMock(status_code=500, text="Internal Server Error")

        result = get_uniprot_data("test_gene", "9606")

        # Verify that the function handles errors properly and returns None
        self.assertIsNone(result)

    @patch("requests.get")
    def test_get_uniprot_no_results(self, mock_get):
        # Set up the mock to return a successful response with no results
        example_response = {"results": []}
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: example_response
        )

        result = get_uniprot_data("test_gene", "9606")

        # Verify that the function handles no results found properly and returns None
        self.assertIsNone(result)


class TestExtractData(unittest.TestCase):
    def setUp(self):
        self.example_data = {
            "entryType": "protein",
            "primaryAccession": "P12345",
            "secondaryAccessions": ["P2345", "Q12345"],
            "organism": {
                "scientificName": "Homo sapiens",
                "commonName": "human",
                "taxonId": "9606",
                "lineage": ["Eukaryota", "Metazoa", "Chordata", "Mammalia", "Primates"],
            },
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Test Protein"}},
                "alternativeNames": [
                    {"fullName": {"value": "Protein Alt1"}},
                    {"fullName": {"value": "Protein Alt2"}},
                ],
                "flag": "Precursor",
            },
            "genes": [
                {
                    "geneName": {"value": "TEST"},
                    "synonyms": [{"value": "Test Syn1"}, {"value": "Test Syn2"}],
                }
            ],
            "comments": [
                {
                    "commentType": "FUNCTION",
                    "texts": [{"value": "Function description."}],
                },
                {
                    "commentType": "SUBUNIT",
                    "texts": [{"value": "Subunit description."}],
                },
                {
                    "commentType": "INTERACTION",
                    "interactions": [
                        {
                            "interactantOne": {"uniProtKBAccession": "Q12345"},
                            "interactantTwo": {"uniProtKBAccession": "Q23456"},
                            "numberOfExperiments": 5,
                        }
                    ],
                },
                {
                    "commentType": "SUBCELLULAR LOCATION",
                    "subcellularLocations": [
                        {"location": {"value": "Cytoplasm"}},
                        {"location": {"value": "Nucleus"}},
                    ],
                },
                {
                    "commentType": "TISSUE SPECIFICITY",
                    "texts": [{"value": "Expressed in liver."}],
                },
            ],
            "features": [
                {
                    "type": "domain",
                    "description": "ATP-binding region",
                    "location": {"start": {"value": 100}, "end": {"value": 200}},
                }
            ],
            "references": [
                {
                    "citation": {
                        "authors": ["Author A"],
                        "title": "Paper title",
                        "journal": "Journal",
                        "publicationDate": "2021-01",
                    }
                }
            ],
            "uniProtKBCrossReferences": [
                {
                    "database": "EMBL",
                    "id": "ABC123",
                    "properties": [{"key": "molecule type", "value": "mRNA"}],
                },
                {
                    "database": "GO",
                    "id": "ABC123",
                    "properties": [{"key": "name", "value": "P:some pathway"}],
                },
                {
                    "database": "Reactome",
                    "id": "ABC1234",
                    "properties": [{"key": "name", "value": "some pathway"}],
                },
            ],
        }

    def test_extract_data_success(self):
        result = extract_data(self.example_data)

        # Verify the top-level data extraction
        self.assertEqual(result["entryType"], "protein")
        self.assertEqual(result["primaryAccession"], "P12345")
        self.assertEqual(result["secondaryAccessions"], ["P2345", "Q12345"])

        # Verify protein details are extracted properly
        expected_protein = {
            "recommendedName": "Test Protein",
            "alternativeNames": ["Protein Alt1", "Protein Alt2"],
            "flag": "Precursor",
        }
        self.assertEqual(result["protein"], expected_protein)

        # Verify genes are extracted correctly
        expected_genes = {
            "geneName": "TEST",
            "synonyms": ["Test Syn1", "Test Syn2"],
        }
        self.assertEqual(result["genes"], expected_genes)

        # Verify function comments extraction
        expected_function_comments = ["Function description."]
        self.assertEqual(result["functionComments"], expected_function_comments)

        # Verify subunit comments extraction
        expected_subunit_comments = ["Subunit description."]
        self.assertEqual(result["subunitComments"], expected_subunit_comments)

        # Verify protein interactions are extracted correctly
        expected_interactions = [
            {
                "interactor": "Q23456",
                "numberOfExperiments": 5,
            }
        ]
        self.assertEqual(result["interactions"], expected_interactions)

        # Verify subcellular locations are extracted properly
        expected_locations = ["Cytoplasm", "Nucleus"]
        self.assertEqual(result["subcellularLocations"], expected_locations)

        # Verify tissue specificity extraction
        expected_tissue_specificity = ["Expressed in liver."]
        self.assertEqual(result["tissueSpecificity"], expected_tissue_specificity)

        # Verify cross references extraction
        expected_GOP = [
            {
                "id": "ABC123",
                "name": "some pathway",
            }
        ]
        self.assertEqual(result["GO Pathway"], expected_GOP)
        expected_reactome = [
            {
                "id": "ABC1234",
                "name": "some pathway",
            },
        ]
        self.assertEqual(result["Reactome"], expected_reactome)


class TestSelectID(unittest.TestCase):
    def setUp(self):
        self.results = [
            {
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "primaryAccession": "P1",
                "genes": [{"geneName": {"value": "G1"}}],
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Well annotated sp protein"}
                    }
                },
                "annotationScore": 5,
            },
            {
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "primaryAccession": "P2",
                "genes": [{"geneName": {"value": "G2"}}],
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Less well annotated sp protein"}
                    }
                },
                "annotationScore": 3,
            },
            {
                "entryType": "Inactive",
                "primaryAccession": "P3",
                "proteinDescription": {
                    "recommendedName": {"fullName": {"value": "Inactive protein"}}
                },
            },
            {
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "primaryAccession": "P4",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Immunoglobulin without gene name"}
                    }
                },
                "annotationScore": 3,
            },
            {
                "entryType": "UniProtKB unreviewed (TrEMBL)",
                "primaryAccession": "P5",
                "genes": [{"geneName": {"value": "G5"}}],
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {
                            "value": "Trembl protein with gene name, well annotated"
                        }
                    }
                },
                "annotationScore": 5,
            },
            {
                "entryType": "UniProtKB unreviewed (TrEMBL)",
                "primaryAccession": "P6",
                "genes": [{"geneName": {"value": "G1"}}],
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {
                            "value": "Trembl protein with gene name, less well annotated"
                        }
                    }
                },
                "annotationScore": 4,
            },
            {
                "entryType": "UniProtKB unreviewed (TrEMBL)",
                "primaryAccession": "P7",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Trembl protein without gene name"}
                    }
                },
                "annotationScore": 2,
            },
            {
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "primaryAccession": "P8",
                "genes": [{"geneName": {"value": "G1"}}],
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Redundant well annotated sp protein"}
                    }
                },
                "annotationScore": 5,
            },
        ]

    def get_example_results(self, ids: list):
        results = [
            result for result in self.results if result["primaryAccession"] in ids
        ]
        return results

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_single_id(self, mock_results):
        mock_results.return_value = self.get_example_results(["P7"])
        result = select_uniprot_id_from_feature("P7")
        self.assertEqual(result, self.get_example_results(["P7"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_return_active_wo_gene(self, mock_results):
        mock_results.return_value = self.get_example_results(["P7", "P3"])
        result = select_uniprot_id_from_feature("P7;P3")
        self.assertEqual(result, self.get_example_results(["P7"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_return_active_with_gene(self, mock_results):
        mock_results.return_value = self.get_example_results(["P7", "P3", "P6"])
        result = select_uniprot_id_from_feature("P7;P3;P6")
        self.assertEqual(result, self.get_example_results(["P6"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_return_active_immunoglobulin(self, mock_results):
        mock_results.return_value = self.get_example_results(["P7", "P3", "P4"])
        result = select_uniprot_id_from_feature("P7;P3;P4")
        self.assertEqual(result, self.get_example_results(["P4"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_single_swissprot_samegene(self, mock_results):
        mock_results.return_value = self.get_example_results(["P6", "P1"])
        result = select_uniprot_id_from_feature("P6;P1")
        self.assertEqual(result, self.get_example_results(["P1"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_single_swissprot_differentgene(self, mock_results):
        mock_results.return_value = self.get_example_results(["P5", "P1"])
        result = select_uniprot_id_from_feature("P5;P1")
        self.assertEqual(result, self.get_example_results(["P1"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_first_swissprot_samegenes(self, mock_results):
        mock_results.return_value = self.get_example_results(["P1", "P8"])
        result = select_uniprot_id_from_feature("P1;P8")
        self.assertEqual(result, self.get_example_results(["P1"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_better_trembl(self, mock_results):
        mock_results.return_value = self.get_example_results(["P5", "P6"])
        result = select_uniprot_id_from_feature("P5;P6")
        self.assertEqual(result, self.get_example_results(["P5"])[0])

    @patch("alphastats.llm.uniprot_utils.get_uniprot_data_for_ids")
    def test_select_better_swissprot(self, mock_results):
        mock_results.return_value = self.get_example_results(["P1", "P2"])
        result = select_uniprot_id_from_feature("P1;P2")
        self.assertEqual(result, self.get_example_results(["P1"])[0])


if __name__ == "__main__":
    unittest.main()
