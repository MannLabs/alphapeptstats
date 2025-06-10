import pytest

from alphastats.dataset.id_holder import IdHolder


def test_init_id_holder():
    """Test basic initialization and functionality of IdHolder."""
    features_list = ["id1;id2", "id3", "id4;id5"]
    proteins_list = ["id1;id2", "id3", "id4;id5"]
    gene_names_list = ["gene1;gene2", "gene3", "gene4"]

    # when
    id_holder = IdHolder(
        features_list=features_list,
        proteins_list=proteins_list,
        gene_names_list=gene_names_list,
    )

    expected_gene_to_features_map = {
        "gene1": ["id1;id2"],
        "gene2": ["id1;id2"],
        "gene3": ["id3"],
        "gene4": ["id4;id5"],
    }

    expected_protein_to_features_map = {
        "id1": ["id1;id2"],
        "id2": ["id1;id2"],
        "id3": ["id3"],
        "id4": ["id4;id5"],
        "id5": ["id4;id5"],
    }

    expected_feature_to_repr_map = {
        "id1;id2": "gene1;gene2",
        "id3": "gene3",
        "id4;id5": "gene4",
    }

    assert id_holder.gene_to_features_map == expected_gene_to_features_map
    assert id_holder.protein_to_features_map == expected_protein_to_features_map
    assert id_holder.feature_to_repr_map == expected_feature_to_repr_map


@pytest.fixture
def id_holder():
    """Create an IdHolder instance for testing."""
    features_list = ["id1;id2", "id3", "id4;id5"]
    proteins_list = ["id1;id2", "id3", "id4;id5"]
    gene_names_list = ["gene1;gene2", "gene3", "gene4"]

    return IdHolder(
        features_list=features_list,
        proteins_list=proteins_list,
        gene_names_list=gene_names_list,
    )


@pytest.mark.parametrize(
    "search_string,expected_result",
    [
        ("id1;id2", ["id1;id2"]),  # Search by feature key
        ("gene1;gene2", ["id1;id2"]),  # Search by representation value (gene name)
        ("id1", ["id1;id2"]),  # Search by individual protein ID
        ("gene3", ["id3"]),  # Search by individual gene name
        ("id4", ["id4;id5"]),  # Search by another individual protein ID
        ("gene4", ["id4;id5"]),  # Search by another individual gene name
    ],
)
def test_get_feature_ids_from_search_string(id_holder, search_string, expected_result):
    """Test get_feature_ids_from_search_string method with various inputs."""
    result = id_holder.get_feature_ids_from_search_string(search_string)
    assert result == expected_result


def test_get_feature_ids_from_search_string_raises_error(id_holder):
    """Test get_feature_ids_from_search_string raises ValueError for non-existent feature."""
    with pytest.raises(
        ValueError, match="Feature nonexistent is not in the \\(processed\\) data\\."
    ):
        id_holder.get_feature_ids_from_search_string("nonexistent")


@pytest.mark.parametrize(
    "features_list,expected_result",
    [
        (["id1", "gene3"], ["id1;id2", "id3"]),  # Mix of protein and gene
        (
            ["gene1;gene2", "id4"],
            ["id1;id2", "id4;id5"],
        ),  # Gene representation and protein
        (["id1;id2", "id3"], ["id1;id2", "id3"]),  # Direct feature keys
        (
            ["gene1", "gene2", "gene3"],
            ["id1;id2", "id1;id2", "id3"],
        ),  # Multiple genes, some duplicates
        (
            ["id1", "id2", "id3"],
            ["id1;id2", "id1;id2", "id3"],
        ),  # Multiple proteins, some duplicates
    ],
)
def test_get_multiple_feature_ids_from_strings(
    id_holder, features_list, expected_result
):
    """Test get_multiple_feature_ids_from_strings method with various inputs."""
    result = id_holder.get_multiple_feature_ids_from_strings(features_list)
    assert result == expected_result


def test_get_multiple_feature_ids_from_strings_with_warnings(id_holder):
    """Test get_multiple_feature_ids_from_strings issues warnings for unmapped features."""
    import warnings

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        result = id_holder.get_multiple_feature_ids_from_strings(
            ["id1", "nonexistent", "gene3"]
        )

        # Check that warning was issued
        assert len(warning_list) == 1
        assert "Could not find the following features: nonexistent" in str(
            warning_list[0].message
        )

        # Check that valid features were still returned
        assert result == ["id1;id2", "id3"]


def test_get_multiple_feature_ids_from_strings_raises_error_no_valid_features(
    id_holder,
):
    """Test get_multiple_feature_ids_from_strings raises ValueError when no valid features provided."""
    with pytest.raises(ValueError, match="No valid features provided\\."):
        id_holder.get_multiple_feature_ids_from_strings(
            ["nonexistent1", "nonexistent2"]
        )
