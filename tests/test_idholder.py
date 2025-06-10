from alphastats.dataset.idholder import IdHolder


def test_init_id_holder():
    """Test basic initialization and functionality of IdHolder."""
    features_list = ["protein1;protein2", "protein3", "protein4;protein5"]
    proteins_list = ["protein1;protein2", "protein3", "protein4;protein5"]
    gene_names_list = ["gene1;gene2", "gene3", "gene4"]

    # when
    id_holder = IdHolder(
        features_list=features_list,
        proteins_list=proteins_list,
        gene_names_list=gene_names_list,
    )

    expected_gene_to_features_map = {
        "gene1": ["protein1;protein2"],
        "gene2": ["protein1;protein2"],
        "gene3": ["protein3"],
        "gene4": ["protein4;protein5"],
    }

    expected_protein_to_features_map = {
        "protein1": ["protein1;protein2"],
        "protein2": ["protein1;protein2"],
        "protein3": ["protein3"],
        "protein4": ["protein4;protein5"],
        "protein5": ["protein4;protein5"],
    }

    expected_feature_to_repr_map = {
        "protein1;protein2": "gene1;gene2",
        "protein3": "gene3",
        "protein4;protein5": "gene4",
    }

    assert id_holder.gene_to_features_map == expected_gene_to_features_map
    assert id_holder.protein_to_features_map == expected_protein_to_features_map
    assert id_holder.feature_to_repr_map == expected_feature_to_repr_map
