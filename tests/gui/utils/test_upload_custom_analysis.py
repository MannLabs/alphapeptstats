"""Unit tests for upload_custom_analysis utility functions."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from alphastats.dataset.keys import Cols, Regulation
from alphastats.gui.utils.upload_custom_analysis import (
    create_custom_result_component,
    parse_custom_analysis_file,
)


class TestParseCustomAnalysisFile:
    """Test cases for parse_custom_analysis_file function."""

    @patch("pandas.read_csv")
    def test_parse_valid_file(self, mock_read_csv):
        """Test parsing a valid custom analysis file."""
        # Mock the CSV data that would be read
        mock_df = pd.DataFrame(
            {
                "Significant": ["+", "", "+"],
                "Difference": [1.5, -0.8, -2.1],
                "Protein IDs": ["P12345", "P23456", "P34567"],
                "Gene names": ["GENE1", "GENE2", "GENE3"],
            }
        )
        mock_read_csv.return_value = mock_df

        # Create a mock uploaded file
        mock_file = Mock()

        result = parse_custom_analysis_file(mock_file)

        # Check that all required columns are present (function may add or keep some original columns)
        required_columns = {Cols.SIGNIFICANT, "Difference", Cols.INDEX, "Gene names"}
        assert required_columns.issubset(set(result.columns))

        # Check data content
        assert len(result) == 3
        assert result[Cols.INDEX].tolist() == ["P12345", "P23456", "P34567"]
        assert result["Gene names"].tolist() == ["GENE1", "GENE2", "GENE3"]
        assert result["Difference"].tolist() == [1.5, -0.8, -2.1]

        # Check regulation assignment
        assert result[Cols.SIGNIFICANT].tolist() == [
            Regulation.UP,
            Regulation.NON_SIG,
            Regulation.DOWN,
        ]

    @patch("pandas.read_csv")
    def test_parse_file_missing_columns(self, mock_read_csv):
        """Test parsing a file with missing required columns."""
        # Mock CSV with missing column
        mock_df = pd.DataFrame(
            {
                "Significant": ["+"],
                "Difference": [1.5],
                "Protein IDs": ["P12345"],
                # Missing 'Gene names'
            }
        )
        mock_read_csv.return_value = mock_df

        mock_file = Mock()

        with pytest.raises(
            ValueError, match="Missing required columns: \\['Gene names'\\]"
        ):
            parse_custom_analysis_file(mock_file)

    @patch("pandas.read_csv")
    def test_parse_file_with_nan_significance(self, mock_read_csv):
        """Test parsing file where Significant column contains NaN values."""
        import numpy as np

        mock_df = pd.DataFrame(
            {
                "Significant": [np.nan, "+"],
                "Difference": [1.5, -0.8],
                "Protein IDs": ["P12345", "P23456"],
                "Gene names": ["GENE1", "GENE2"],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_file = Mock()

        result = parse_custom_analysis_file(mock_file)

        # Check regulation assignment with NaN handling
        assert result[Cols.SIGNIFICANT].tolist() == [
            Regulation.NON_SIG,
            Regulation.DOWN,
        ]

    @patch("pandas.read_csv")
    def test_parse_file_edge_cases(self, mock_read_csv):
        """Test parsing file with edge cases like zero difference."""
        mock_df = pd.DataFrame(
            {
                "Significant": ["+", "+", "+"],
                "Difference": [0.0, 1.0, -1.0],
                "Protein IDs": ["P12345", "P23456", "P34567"],
                "Gene names": ["GENE1", "GENE2", "GENE3"],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_file = Mock()

        result = parse_custom_analysis_file(mock_file)

        # Zero difference should not be UP or DOWN (only positive/negative)
        expected_regulation = [Regulation.NON_SIG, Regulation.UP, Regulation.DOWN]
        assert result[Cols.SIGNIFICANT].tolist() == expected_regulation

    @patch("pandas.read_csv")
    def test_parse_file_with_spaces_in_significance(self, mock_read_csv):
        """Test parsing file where Significant column contains spaces."""
        mock_df = pd.DataFrame(
            {
                "Significant": [" ", "+"],
                "Difference": [1.5, -0.8],
                "Protein IDs": ["P12345", "P23456"],
                "Gene names": ["GENE1", "GENE2"],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_file = Mock()

        result = parse_custom_analysis_file(mock_file)

        # Space should be treated as non-significant
        assert result[Cols.SIGNIFICANT].tolist() == [
            Regulation.NON_SIG,
            Regulation.DOWN,
        ]


class TestCreateCustomResultComponent:
    """Test cases for create_custom_result_component function."""

    def create_sample_dataframe(self) -> pd.DataFrame:
        """Create a sample parsed dataframe for testing."""
        return pd.DataFrame(
            {
                Cols.SIGNIFICANT: [Regulation.UP, Regulation.DOWN, Regulation.NON_SIG],
                "Difference": [1.5, -2.1, 0.3],
                Cols.INDEX: ["P12345", "P23456", "P34567"],
                "Gene names": ["GENE1", "GENE2", "GENE3"],
            }
        )

    def test_create_result_component(self):
        """Test creating ResultComponent from parsed dataframe."""
        parsed_df = self.create_sample_dataframe()

        result_component, id_holder = create_custom_result_component(parsed_df)

        # Check ResultComponent attributes
        assert result_component.dataframe.equals(parsed_df)
        assert result_component.annotated_dataframe.equals(parsed_df)
        assert result_component.preprocessing == {}
        assert result_component.method == {}
        assert result_component.feature_to_repr_map == {}
        assert result_component._is_plottable is False

    def test_create_id_holder(self):
        """Test creating IdHolder from parsed dataframe."""
        parsed_df = self.create_sample_dataframe()

        result_component, id_holder = create_custom_result_component(parsed_df)

        # Check IdHolder has the expected mappings
        # IdHolder creates internal mappings, let's check that it was created properly
        assert id_holder is not None
        assert hasattr(id_holder, "feature_to_repr_map")
        assert hasattr(id_holder, "protein_to_features_map")
        assert hasattr(id_holder, "gene_to_features_map")

    def test_dataframe_independence(self):
        """Test that annotated_dataframe is independent of original dataframe."""
        parsed_df = self.create_sample_dataframe()

        result_component, _ = create_custom_result_component(parsed_df)

        # Modify original dataframe
        parsed_df.loc[0, "Difference"] = 999.0

        # annotated_dataframe should remain unchanged
        assert result_component.annotated_dataframe.loc[0, "Difference"] == 1.5
        assert result_component.dataframe.loc[0, "Difference"] == 999.0

    def test_empty_dataframe(self):
        """Test creating ResultComponent with empty dataframe."""
        empty_df = pd.DataFrame(
            columns=[Cols.SIGNIFICANT, "Difference", Cols.INDEX, "Gene names"]
        )

        result_component, id_holder = create_custom_result_component(empty_df)

        # Check that empty dataframes are handled correctly
        assert len(result_component.dataframe) == 0
        assert len(result_component.annotated_dataframe) == 0
        assert id_holder is not None


class TestIntegration:
    """Integration tests for the upload custom analysis workflow."""

    @patch("pandas.read_csv")
    def test_full_workflow(self, mock_read_csv):
        """Test the complete workflow from file upload to ResultComponent creation."""
        # Create mock data
        mock_df = pd.DataFrame(
            {
                "Significant": ["+", "", "+"],
                "Difference": [2.3, -1.2, -0.9],
                "Protein IDs": ["P11111", "P22222", "P33333"],
                "Gene names": ["GENEA", "GENEB", "GENEC"],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_file = Mock()

        # Parse file
        parsed_df = parse_custom_analysis_file(mock_file)

        # Create ResultComponent
        result_component, id_holder = create_custom_result_component(parsed_df)

        # Verify end-to-end workflow
        assert len(result_component.annotated_dataframe) == 3
        assert result_component.annotated_dataframe[Cols.SIGNIFICANT].tolist() == [
            Regulation.UP,
            Regulation.NON_SIG,
            Regulation.DOWN,
        ]
        assert id_holder is not None

    @patch("pandas.read_csv")
    def test_regulation_consistency(self, mock_read_csv):
        """Test that regulation assignment is consistent across the workflow."""
        # Test various significance and difference combinations
        mock_df = pd.DataFrame(
            {
                "Significant": ["+", "+", "+", "", "", " "],
                "Difference": [1.0, -1.0, 0.0, 1.0, -1.0, 2.0],
                "Protein IDs": [
                    "P00001",
                    "P00002",
                    "P00003",
                    "P00004",
                    "P00005",
                    "P00006",
                ],
                "Gene names": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"],
            }
        )
        mock_read_csv.return_value = mock_df

        expected_regulations = [
            Regulation.UP,  # +, 1.0
            Regulation.DOWN,  # +, -1.0
            Regulation.NON_SIG,  # +, 0.0 (zero difference)
            Regulation.NON_SIG,  # '', 1.0 (not significant)
            Regulation.NON_SIG,  # '', -1.0 (not significant)
            Regulation.NON_SIG,  # ' ', 2.0 (space = not significant)
        ]

        mock_file = Mock()

        parsed_df = parse_custom_analysis_file(mock_file)
        result_component, _ = create_custom_result_component(parsed_df)

        actual_regulations = result_component.annotated_dataframe[
            Cols.SIGNIFICANT
        ].tolist()
        assert actual_regulations == expected_regulations


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("pandas.read_csv")
    def test_invalid_data_types(self, mock_read_csv):
        """Test handling of invalid data types in Difference column."""
        mock_df = pd.DataFrame(
            {
                "Significant": ["+", "+"],
                "Difference": ["invalid", 1.5],  # String in numeric column
                "Protein IDs": ["P12345", "P23456"],
                "Gene names": ["GENE1", "GENE2"],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_file = Mock()

        # Should raise an error due to invalid data types
        with pytest.raises(TypeError):
            parse_custom_analysis_file(mock_file)

    @patch("pandas.read_csv")
    def test_all_missing_significance(self, mock_read_csv):
        """Test file where all entries are non-significant."""
        import numpy as np

        mock_df = pd.DataFrame(
            {
                "Significant": ["", np.nan, " "],
                "Difference": [1.5, -0.8, 2.1],
                "Protein IDs": ["P12345", "P23456", "P34567"],
                "Gene names": ["GENE1", "GENE2", "GENE3"],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_file = Mock()

        result = parse_custom_analysis_file(mock_file)

        # All should be non-significant
        assert all(reg == Regulation.NON_SIG for reg in result[Cols.SIGNIFICANT])
