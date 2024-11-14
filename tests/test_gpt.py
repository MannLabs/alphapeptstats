import logging
import unittest
from unittest.mock import MagicMock, patch

from alphastats.DataSet import DataSet
from alphastats.llm.uniprot_utils import extract_data, get_uniprot_data
from alphastats.loader.MaxQuantLoader import MaxQuantLoader

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
                }
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
        result = get_uniprot_data("test_gene", "9606")

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
        self.sample_data = {
            "entryType": "protein",
            "primaryAccession": "P12345",
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
                }
            ],
        }

    def test_extract_data_success(self):
        result = extract_data(self.sample_data)

        # Verify the top-level data extraction
        self.assertEqual(result["entryType"], "protein")
        self.assertEqual(result["primaryAccession"], "P12345")

        # Verify organism details are extracted correctly
        expected_organism = {
            "scientificName": "Homo sapiens",
            "commonName": "human",
            "taxonId": "9606",
            "lineage": ["Eukaryota", "Metazoa", "Chordata", "Mammalia", "Primates"],
        }
        self.assertEqual(result["organism"], expected_organism)

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
                "interactantOne": "Q12345",
                "interactantTwo": "Q23456",
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

        # Verify protein features are extracted correctly
        expected_features = [
            {
                "type": "domain",
                "description": "ATP-binding region",
                "location_start": 100,
                "location_end": 200,
            }
        ]
        self.assertEqual(result["features"], expected_features)

        # Verify references are extracted properly
        expected_references = [
            {
                "authors": ["Author A"],
                "title": "Paper title",
                "journal": "Journal",
                "publicationDate": "2021-01",
                "comments": [],
            }
        ]
        self.assertEqual(result["references"], expected_references)

        # Verify cross references extraction
        expected_cross_references = [
            {
                "database": "EMBL",
                "id": "ABC123",
                "properties": {"molecule type": "mRNA"},
            }
        ]
        self.assertEqual(result["crossReferences"], expected_cross_references)


if __name__ == "__main__":
    unittest.main()
