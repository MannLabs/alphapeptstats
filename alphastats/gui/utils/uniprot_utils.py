from typing import Union, List, Dict
import requests


import streamlit as st


uniprot_fields = [
    # Names & Taxonomy
    "gene_names",
    "organism_name",
    "protein_name",
    # Function
    "cc_function",
    "cc_catalytic_activity",
    "cc_activity_regulation",
    "cc_pathway",
    "kinetics",
    "ph_dependence",
    "temp_dependence",
    # Interaction
    "cc_interaction",
    "cc_subunit",
    # Expression
    "cc_tissue_specificity",
    "cc_developmental_stage",
    "cc_induction",
    # Gene Ontology (GO)
    "go",
    "go_p",
    "go_c",
    "go_f",
    # Pathology & Biotech
    "cc_disease",
    "cc_disruption_phenotype",
    "cc_pharmaceutical",
    "ft_mutagen",
    "ft_act_site",
    # Structure
    "cc_subcellular_location",
    "organelle",
    "absorption",
    # Publications
    "lit_pubmed_id",
    # Family & Domains
    "protein_families",
    "cc_domain",
    "ft_domain",
    # Protein-Protein Interaction Databases
    "xref_biogrid",
    "xref_intact",
    "xref_mint",
    "xref_string",
    # Chemistry Databases
    "xref_drugbank",
    "xref_chembl",
    "reviewed",
]


def get_uniprot_data(
    gene_name: str,
    organism_id: str,
    fields: List[str] = uniprot_fields,
) -> Dict:
    """
    Get data from UniProt for a given gene name and organism ID.

    Args:
        gene_name (str): The gene name to search for.
        organism_id (str, optional): The organism ID to search in. Defaults to streamlit session state.
        fields (list[str], optional): The fields to retrieve from UniProt. Defaults to uniprot_fields defined above.

    Returns:
        dict: The data retrieved from UniProt.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"(gene:{gene_name}) AND (reviewed:true) AND (organism_id:{organism_id})"

    response = requests.get(
        base_url, params={"query": query, "format": "json", "fields": ",".join(fields)}
    )

    if response.status_code != 200:
        print(
            f"Failed to retrieve data for {gene_name}. Status code: {response.status_code}"
        )
        print(response.text)
        return None

    data = response.json()

    if not data.get("results"):
        print(f"No UniProt entry found for {gene_name}")
        return None

    # Return the first result as a dictionary (assuming it's the most relevant)
    data = data["results"][0]
    # for key, value in data.items():
    #     print(f"data - {key}: {value}, {type(value)}")
    return data


def extract_data(data: Dict) -> Dict:
    """
    Extract relevant data from a UniProt entry.

    Args:
        data (dict): The data retrieved from UniProt.

    Returns:
        dict: The extracted data.
    """
    extracted = {}

    # 1. Entry Type
    extracted["entryType"] = data.get("entryType", None)

    # 2. Primary Accession
    extracted["primaryAccession"] = data.get("primaryAccession", None)

    # 3. Organism Details
    organism = data.get("organism", {})
    extracted["organism"] = {
        "scientificName": organism.get("scientificName", None),
        "commonName": organism.get("commonName", None),
        "taxonId": organism.get("taxonId", None),
        "lineage": organism.get("lineage", []),
    }

    # 4. Protein Details
    protein_description = data.get("proteinDescription", {})
    recommended_name = (
        protein_description.get("recommendedName", {})
        .get("fullName", {})
        .get("value", None)
    )
    alternative_names = [
        alt_name["fullName"]["value"]
        for alt_name in protein_description.get("alternativeNames", [])
    ]
    extracted["protein"] = {
        "recommendedName": recommended_name,
        "alternativeNames": alternative_names,
        "flag": protein_description.get("flag", None),
    }

    # 5. Gene Details
    genes = data.get("genes", [{}])[0]
    extracted["genes"] = {
        "geneName": genes.get("geneName", {}).get("value", None),
        "synonyms": [syn["value"] for syn in genes.get("synonyms", [])],
    }

    # 6. Functional Comments
    function_comments = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "FUNCTION"
        for text in comment.get("texts", [])
    ]
    extracted["functionComments"] = function_comments

    # 7. Subunit Details
    subunit_comments = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "SUBUNIT"
        for text in comment.get("texts", [])
    ]
    extracted["subunitComments"] = subunit_comments

    # 8. Protein Interactions
    interactions = []

    for c in data.get("comments", []):
        if c["commentType"] == "INTERACTION":
            for interaction in c.get("interactions", []):
                interactantOne = interaction["interactantOne"].get(
                    "uniProtKBAccession", None
                )
                interactantTwo = interaction["interactantTwo"].get(
                    "uniProtKBAccession", None
                )

                # Only append if both interactants are present
                if interactantOne and interactantTwo:
                    interactions.append(
                        {
                            "interactantOne": interactantOne,
                            "interactantTwo": interactantTwo,
                            "numberOfExperiments": interaction["numberOfExperiments"],
                        }
                    )
    extracted["interactions"] = interactions

    # 9. Subcellular Locations
    subcellular_locations_comments = [
        c["subcellularLocations"]
        for c in data.get("comments", [])
        if c["commentType"] == "SUBCELLULAR LOCATION"
    ]
    locations = [
        location["location"]["value"]
        for locations_comment in subcellular_locations_comments
        for location in locations_comment
    ]
    extracted["subcellularLocations"] = locations

    tissue_specificities = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "TISSUE SPECIFICITY"
        for text in comment.get("texts", [])
    ]
    extracted["tissueSpecificity"] = tissue_specificities

    # 11. Protein Features
    features = [
        {
            "type": feature["type"],
            "description": feature["description"],
            "location_start": feature["location"]["start"]["value"],
            "location_end": feature["location"]["end"]["value"],
        }
        for feature in data.get("features", [])
    ]
    extracted["features"] = features

    # 12. References
    references = [
        {
            "authors": ref["citation"].get("authors", []),
            "title": ref["citation"].get("title", ""),
            "journal": ref["citation"].get("journal", ""),
            "publicationDate": ref["citation"].get("publicationDate", ""),
            "comments": [c["value"] for c in ref.get("referenceComments", [])],
        }
        for ref in data.get("references", [])
    ]
    extracted["references"] = references

    # 13. Cross References
    cross_references = [
        {
            "database": ref["database"],
            "id": ref["id"],
            "properties": {
                prop["key"]: prop["value"] for prop in ref.get("properties", [])
            },
        }
        for ref in data.get("uniProtKBCrossReferences", [])
    ]
    extracted["crossReferences"] = cross_references
    return extracted


def get_info(genes_list: List[str], organism_id: str) -> List[str]:
    """
    Get info from UniProt for a list of genes.

    Args:
        genes_list (list[str]): A list of gene names to search for.
        organism_id (str, optional): The Uniprot organism ID to search in.

    Returns:
        list[str]: A list of gene functions."""
    results = {}

    for gene in genes_list:
        result = get_uniprot_data(gene, organism_id)

        # If result is retrieved for the gene, extract data and continue with the next gene
        if result:
            results[gene] = extract_data(result)
            continue

        # If no result is retrieved for the gene and the gene string does not contain a ";", continue with the next gene
        if ";" not in gene:
            print(f"Failed to retrieve data for {gene}")
            continue

        # If no result is retrieved for the gene and the gene string contains a ";", try to get data for each split gene
        split_genes = gene.split(";")
        for split_gene in split_genes:
            result = get_uniprot_data(split_gene.strip(), organism_id)
            if result:
                print(
                    f"Successfully retrieved data for {split_gene} (from split gene: {gene})"
                )
                results[gene] = extract_data(result)
                break

        # If still no result after trying split genes
        if not result:
            print(f"Failed to retrieve data for all parts of split gene: {gene}")
            # TODO: Handle this case further if necessary

    gene_functions = []
    for gene in results:
        if results[gene]["functionComments"]:
            gene_functions.append(f"{gene}: {results[gene]['functionComments']}")
        else:
            gene_functions.append(f"{gene}: ?")

    return gene_functions


def get_gene_function(gene_name: Union[str, Dict], organism_id=9606) -> str:
    """
    Get the gene function and description by UniProt lookup of gene identifier / name.

    Args:
        gene_name (Union[str, dict]): Gene identifier / name for UniProt lookup.
        organism_id (str): The UniProt organism ID to search in.

    Returns:
        str: The gene function and description.
    """
    if "organism" in st.session_state:
        organism_id = st.session_state["organism"]
    if isinstance(gene_name, dict):
        gene_name = gene_name["gene_name"]
    result = get_uniprot_data(gene_name, organism_id)
    if result and extract_data(result)["functionComments"]:
        return str(extract_data(result)["functionComments"])
    else:
        return "No data found"
