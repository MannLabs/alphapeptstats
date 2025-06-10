import warnings
from collections import defaultdict
from typing import Optional


class IdHolder:
    """Class to hold the id dictionaries for features, proteins and genes."""

    def __init__(
        self,
        features_list: list[str],
        proteins_list: list[str],
        gene_names_list: Optional[list[str]],
        sep: str = ";",
    ):
        """Initialize the IdHolder.

        Args:
            features_list: list[str]: A list of features (usually protein groups).
            proteins_list: list[str]: A list of protein identifiers corresponding to the features.
            gene_names_list (Optional[list[str]]): A list of gene names corresponding to the features. Default is None.
            sep (str): The separator used to split gene and protein identifiers. Default is ";".
        """
        (
            self.gene_to_features_map,
            self.protein_to_features_map,
            self.feature_to_repr_map,
        ) = self._create_id_dicts(features_list, proteins_list, gene_names_list, sep)

    @staticmethod
    def _create_id_dicts(
        features_list: list[str],
        proteins_list: list[str],
        gene_names_list: Optional[list[str]] = None,
        sep: str = ";",
    ) -> tuple[dict, dict, dict]:
        """Create mappings from gene and protein to feature, and from feature to representation.

        Features are the entities measured in each sample, usually protein groups represented by semicolon separated protein ids.
        This is to maintain the many-to-many relationships between the three entities feature, protein and gene.

        This method processes the raw input data to generate three dictionaries:
        1. gene_to_features_map: Maps each gene to a list of features.
        2. protein_to_features_map: Maps each protein to a list of features.
        3. feature_to_repr_map: Maps each feature to its representation string.

        Args:
            features_list: list[str]: A list of features (usually protein groups).
            proteins_list: list[str]: A list of protein identifiers corresponding to the features.
            gene_names_list (Optional[list[str]]): A list of gene names corresponding to the features. Default is None.

            sep (str): The separator used to split gene and protein identifiers. Default is ";".

        Returns:
            Tuple[dict, dict, dict]: A tuple containing three dictionaries:
            - gene_to_features_map (dict): A dictionary mapping genes to features.
            - protein_to_features_map (dict): A dictionary mapping proteins to features.
            - feature_to_repr_map (dict): A dictionary mapping features to their representation strings.
        """

        features = set(features_list)
        gene_to_features_map = defaultdict(list)
        protein_to_features_map = defaultdict(list)
        feature_to_repr_map = {}

        for proteins, feature in zip(proteins_list, proteins_list):
            if feature not in features:
                continue
            # TODO: Shorten list if too many ids e.g. to id1;...(19) if 20 ids are present
            feature_to_repr_map[feature] = "ids:" + proteins
            for protein in proteins.split(sep):
                protein_to_features_map[protein].append(feature)

        if gene_names_list is not None:
            for genes, feature in zip(gene_names_list, proteins_list):
                if feature not in features:
                    continue
                if isinstance(genes, str):
                    for gene in genes.split(sep):
                        gene_to_features_map[gene].append(feature)
                    feature_to_repr_map[feature] = genes

        return gene_to_features_map, protein_to_features_map, feature_to_repr_map

    def get_feature_ids_from_search_string(self, string: str) -> list[str]:
        """Get the feature id from a string representing a feature.

        Goes through id mapping dictionaries and finds the completest match.

        Parameters
        ----------
        string : str
            The string representating the feature."""

        if string in self.feature_to_repr_map:
            return [string]
        representation_keys = [
            feature
            for feature, representation in self.feature_to_repr_map.items()
            if representation == string
        ]
        if representation_keys:
            return representation_keys
        if string in self.protein_to_features_map:
            return self.protein_to_features_map[string]
        if string in self.gene_to_features_map:
            return self.gene_to_features_map[string]
        raise ValueError(f"Feature {string} is not in the (processed) data.")

    def get_multiple_feature_ids_from_strings(self, features: list) -> list:
        """Get the feature ids from a list of strings representing features.

        Parameters
        ----------
        features : list
            A list of strings representing the features."""

        unmapped_features = []
        protein_ids = []
        for feature in features:
            try:
                for protein_id in self.get_feature_ids_from_search_string(feature):
                    protein_ids.append(protein_id)
            except ValueError:
                unmapped_features.append(feature)
        if unmapped_features:
            warnings.warn(
                f"Could not find the following features: {', '.join(unmapped_features)}"
            )
        if not protein_ids:
            raise ValueError("No valid features provided.")

        return protein_ids
