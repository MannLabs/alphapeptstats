from alphastats.dataset.idholder import IdHolder


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
