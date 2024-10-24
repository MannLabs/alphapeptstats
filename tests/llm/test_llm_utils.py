import pandas as pd
import pytest

from alphastats.llm.llm_utils import (
    get_protein_id_for_gene_name,
    get_subgroups_for_each_group,
)


def test_get_subgroups_for_each_group_basic():
    """Test basic functionality with simple metadata."""
    # Create test metadata
    data = {
        "disease": ["cancer", "healthy", "cancer"],
        "treatment": ["drug_a", "placebo", "drug_b"],
    }
    metadata = pd.DataFrame(data)

    result = get_subgroups_for_each_group(metadata)

    expected = {
        "disease": ["cancer", "healthy"],
        "treatment": ["drug_a", "placebo", "drug_b"],
    }
    assert result == expected


def test_get_subgroups_for_each_group_empty():
    """Test with empty DataFrame."""
    metadata = pd.DataFrame()
    result = get_subgroups_for_each_group(metadata)
    assert result == {}


def test_get_subgroups_for_each_group_single_column():
    """Test with single column DataFrame."""
    data = {"condition": ["A", "B", "A"]}
    metadata = pd.DataFrame(data)

    result = get_subgroups_for_each_group(metadata)

    expected = {"condition": ["A", "B"]}
    assert result == expected


def test_get_subgroups_for_each_group_numeric_values():
    """Test with numeric values in DataFrame."""
    data = {"age_group": [20, 30, 20], "score": [1.5, 2.5, 1.5]}
    metadata = pd.DataFrame(data)

    result = get_subgroups_for_each_group(metadata)

    expected = {"age_group": ["20", "30"], "score": ["1.5", "2.5"]}
    assert result == expected


@pytest.fixture
def gene_to_prot_map():
    """Fixture for protein mapping dictionary."""
    return {
        "VCL": "P18206",
        "VCL;HEL114": "P18206;A0A024QZN4",
        "MULTI;GENE": "PROT1;PROT2;PROT3",
    }


def test_get_protein_id_direct_match(gene_to_prot_map):
    """Test when gene name directly matches a key."""
    result = get_protein_id_for_gene_name("VCL", gene_to_prot_map)
    assert result == "P18206"


def test_get_protein_id_compound_key(gene_to_prot_map):
    """Test when gene name is part of a compound key."""
    result = get_protein_id_for_gene_name("HEL114", gene_to_prot_map)
    assert result == "P18206;A0A024QZN4"


def test_get_protein_id_not_found(gene_to_prot_map):
    """Test when gene name is not found in mapping."""
    result = get_protein_id_for_gene_name("UNKNOWN", gene_to_prot_map)
    assert result == "UNKNOWN"


def test_get_protein_id_empty_map():
    """Test with empty mapping dictionary."""
    result = get_protein_id_for_gene_name("VCL", {})
    assert result == "VCL"


def test_get_protein_id_multiple_matches(gene_to_prot_map):
    """Test with a gene that appears in multiple compound keys."""
    result = get_protein_id_for_gene_name("MULTI", gene_to_prot_map)
    assert result == "PROT1;PROT2;PROT3"


def test_get_protein_id_case_sensitivity(gene_to_prot_map):
    """Test case sensitivity of gene name matching."""
    result = get_protein_id_for_gene_name("vcl", gene_to_prot_map)
    assert result == "vcl"  # Should not match 'VCL' due to case sensitivity
