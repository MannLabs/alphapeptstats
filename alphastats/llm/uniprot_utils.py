import re
from typing import Dict, List, Union

import requests
import streamlit as st

from alphastats.gui.utils.ui_helper import StateKeys

# Fields are only relevant in the context of table output, with json you automatically get everything, with different keys
# uniprot_fields = [
#    # Names & Taxonomy
#    "gene_names",
#    "organism_name",
#    "protein_name",
#    # Function
#    "cc_function",
#    "cc_catalytic_activity",
#    "cc_activity_regulation",
#    "cc_pathway",
#    "kinetics",
#    "ph_dependence",
#    "temp_dependence",
#    # Interaction
#    "cc_interaction",
#    "cc_subunit",
#    # Expression
#    "cc_tissue_specificity",
#    "cc_developmental_stage",
#    "cc_induction",
#    # Gene Ontology (GO)
#    "go",
#    "go_p",
#    "go_c",
#    "go_f",
#    # Pathology & Biotech
#    "cc_disease",
#    "cc_disruption_phenotype",
#    "cc_pharmaceutical",
#    "ft_mutagen",
#    "ft_act_site",
#    # Structure
#    "cc_subcellular_location",
#    "organelle",
#    "absorption",
#    # Publications
#    "lit_pubmed_id",
#    # Family & Domains
#    "protein_families",
#    "cc_domain",
#    "ft_domain",
#    # Protein-Protein Interaction Databases
#    "xref_biogrid",
#    "xref_intact",
#    "xref_mint",
#    "xref_string",
#    # Chemistry Databases
#    "xref_drugbank",
#    "xref_chembl",
#    "reviewed",
# ]


def get_uniprot_data(
    protein_id: str = None,
    gene_name: str = None,
    organism_id: str = "9606",
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
    if protein_id is not None:
        query = f"accession:{protein_id}"
    elif gene_name is not None:
        query = f"(gene:{gene_name}) AND (organism_id:{organism_id})"
    else:
        raise ValueError("Please provide either protein id or gene name.")

    response = requests.get(base_url, params={"query": query, "format": "json"})

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
    extracted["entryType"] = data.get("entryType")

    # 2. Primary Accession
    extracted["primaryAccession"] = data.get("primaryAccession")

    # 3. Secondary Accessions
    extracted["secondaryAccessions"] = data.get("secondaryAccessions")

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
                            "interactor": interactantTwo,
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

    # 10. Tissue specificity
    tissue_specificities = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "TISSUE SPECIFICITY"
        for text in comment.get("texts", [])
    ]
    extracted["tissueSpecificity"] = tissue_specificities

    ## 11. Protein Features
    # features = [
    #    {
    #        "type": feature["type"],
    #        "description": feature["description"],
    #        "location_start": feature["location"]["start"]["value"],
    #        "location_end": feature["location"]["end"]["value"],
    #    }
    #    for feature in data.get("features", [])
    # ]
    # extracted["features"] = features

    ## 12. References
    # references = [
    #    {
    #        "authors": ref["citation"].get("authors", []),
    #        "title": ref["citation"].get("title", ""),
    #        "journal": ref["citation"].get("journal", ""),
    #        "publicationDate": ref["citation"].get("publicationDate", ""),
    #        "comments": [c["value"] for c in ref.get("referenceComments", [])],
    #    }
    #    for ref in data.get("references", [])
    # ]
    # extracted["references"] = references

    # 13. Cross References
    # cross_references = [
    #    {
    #        "database": ref["database"],
    #        "id": ref["id"],
    #        "properties": {
    #            prop["key"]: prop["value"] for prop in ref.get("properties", [])
    #        },
    #    }
    #    for ref in data.get("uniProtKBCrossReferences", [])
    #    if ref['database'] not in ['GO', 'Reactome']
    # ]
    # extracted["crossReferences"] = cross_references

    # 14. Pathway references
    pathway_references = [
        {
            "database": ref["database"],
            "id": ref["id"],
            "pathway": ref.get("properties")[0]["value"],
        }
        if ref["database"] == "Reactome"
        else {
            "database": ref["database"]
            + " "
            + {"P": "Pathway", "C": "Component", "F": "Function"}[
                ref.get("properties")[0]["value"][0]
            ],
            "id": ref["id"],
            "pathway": ref.get("properties")[0]["value"][2::],
        }
        for ref in data.get("uniProtKBCrossReferences", [])
        if ref["database"] in ["GO", "Reactome"]
    ]
    extracted["pathway_references"] = pathway_references

    # TODO: Add caution comments

    return extracted


# TODO unused?
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
        result = result["results"][0]

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
                result = result["results"][0]
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
    if StateKeys.ORGANISM in st.session_state:
        organism_id = st.session_state[StateKeys.ORGANISM]
    if isinstance(gene_name, dict):
        gene_name = gene_name["gene_name"]
    result = get_uniprot_data(gene_name, organism_id)
    if result:
        result = result["results"][0]
    if result and extract_data(result)["functionComments"]:
        return str(extract_data(result)["functionComments"])
    else:
        return "No data found"


def get_uniprot_data_for_ids(ids: list):
    return [get_uniprot_data(protein_id=id)["results"][0] for id in ids]


def select_uniprot_id_from_feature(
    feature: str,
):
    """Get uniprot information for a feaure.

    This function collects the results for all base ids (truncating isoform ids) and selects the one to use for feeding the LLM.
    It does so by reducing the number of results until 1 remains and if that is not straight forward, selects the best annotated (sp over trembl, high annotation score).
    """

    baseids = sorted(
        list(set([identifier.split("-")[0] for identifier in feature.split(";")]))
    )
    results = get_uniprot_data_for_ids(baseids)

    if len(results) == 1:
        return results[0]

    # remove inactive entries and ones without gene names (besides immunoglobulins)
    results = [
        result
        for result in results
        if result["entryType"] != "Inactive"
        and (
            result.get("genes", None) is not None
            or bool(
                re.match(
                    ".*globulin.*|^Ig.*",
                    result.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", ""),
                )
            )
        )
    ]

    if len(results) == 1:
        return results[0]
    elif len(results) == 0:
        return "No data found"

    # Go by gene names, swissprot and annotation scores
    sp_indices = [
        i
        for i, result in enumerate(results)
        if result["entryType"] == "UniProtKB reviewed (Swiss-Prot)"
    ]
    gene_names = [
        result.get("genes", [{}])[0].get("geneName", {}).get("value")
        for result in results
    ]
    annotation_scores = [result.get("annotationScore") for result in results]

    if (len(set(gene_names)) == 1 and len(sp_indices) > 0) or (
        len(set(gene_names)) > 1 and len(sp_indices) == 1
    ):
        # Either all the same gene name and any swissprot entries, or multiple gene names but only one swissprot entry
        index = sp_indices[0]
    elif len(sp_indices) == 0:
        # Multiple gene names and no swissprot entries present
        index = annotation_scores.index(max(annotation_scores))
    else:
        # Multiple gene names and multiple swissprot entries
        index = sp_indices[
            annotation_scores[sp_indices].index(max(annotation_scores[sp_indices]))
        ]
    return results[index]


def extract_fieldinformation_from_uniprotresult(
    result: dict,
    fields: list,
):
    if isinstance(result, str):
        return result
    information = str({k: v for k, v in extract_data(result).items() if k in fields})
    # TODO: Handle fields to format nice artifacts.
    return information


def get_information_for_feature(
    feature: str,
    fields: list = None,
):
    if fields is None:
        fields = ["primaryAccession", "genes", "functionComments"]
    result = select_uniprot_id_from_feature(feature)
    information = extract_fieldinformation_from_uniprotresult(result, fields)
    return information
