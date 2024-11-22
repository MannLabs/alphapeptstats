import re
from typing import Dict, List, Union

import requests
import streamlit as st

from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.ui_helper import StateKeys


class ExtractedFields(metaclass=ConstantsClass):
    DB = "entryType"
    ID = "primaryAccession"
    SECONDARYACC = "secondaryAccessions"
    NAME = "protein"
    GENE = "genes"
    FUNCTIONCOMM = "functionComments"
    SUBUNITCOMM = "subunitComments"
    INTERACTIONS = "interactions"
    SUBCELL = "subcellularLocations"
    TISSUE = "tissueSpecificity"
    GOP = "GO Pathway"
    GOF = "GO Function"
    GOC = "GO Component"
    REACTOME = "Reactome"


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


def _request_uniprot_data(
    protein_id: str = None,
    gene_name: str = None,
    organism_id: str = "9606",
) -> Union[Dict, None]:
    """
    Get data from UniProt for a given gene name and organism ID.

    Args:
        protein_id (str): Uniprot identifier of a protein
        gene_name (str): The gene name to search for.
        organism_id (str, optional): The organism ID to search in. Defaults to human, only used in the context of a gene name.

    Returns:
        One of dict, None, str:
            dict: The data retrieved from UniProt, if retrieval was successful.
            None: If the response code was not 200 or there were no results
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
            f"Failed to retrieve data for {query}. Status code: {response.status_code}"
        )
        print(response.text)
        return None

    data = response.json()

    if not data.get("results"):
        print(f"No UniProt entry found for {query}")
        return None

    return data


def _extract_annotations_from_uniprot_data(data: Dict) -> Dict:
    """
    Extract relevant data from a UniProt entry.
    Note: See the uniprot response model here: https://www.ebi.ac.uk/proteins/api/doc/#!/proteins/search (select json as content type), or inspect a suitable entry like https://rest.uniprot.org/uniprotkb/P12345.

    Args:
        data (dict): The data retrieved from UniProt.

    Returns:
        dict: The extracted data. Dictionary keys are defined in the ExtractedFields constants class.
    """
    extracted = {}

    # 1. Entry Type
    extracted[ExtractedFields.DB] = data.get("entryType")

    # 2. Primary Accession
    extracted[ExtractedFields.ID] = data.get("primaryAccession")

    # 3. Secondary Accessions
    extracted[ExtractedFields.SECONDARYACC] = data.get("secondaryAccessions")

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
    extracted[ExtractedFields.NAME] = {
        "recommendedName": recommended_name,
        "alternativeNames": alternative_names,
        "flag": protein_description.get("flag", None),
    }

    # 5. Gene Details
    genes = data.get("genes", [{}])[0]
    extracted[ExtractedFields.GENE] = {
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
    extracted[ExtractedFields.FUNCTIONCOMM] = function_comments

    # 7. Subunit Details
    subunit_comments = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "SUBUNIT"
        for text in comment.get("texts", [])
    ]
    extracted[ExtractedFields.SUBUNITCOMM] = subunit_comments

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
    extracted[ExtractedFields.INTERACTIONS] = interactions

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
    extracted[ExtractedFields.SUBCELL] = locations

    # 10. Tissue specificity
    tissue_specificities = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "TISSUE SPECIFICITY"
        for text in comment.get("texts", [])
    ]
    extracted[ExtractedFields.TISSUE] = tissue_specificities

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
    annotation_references = [
        {
            "database": ref["database"],
            "entry": {
                "id": ref["id"],
                "name": ref.get("properties")[0]["value"],
            },
        }
        if ref["database"] == "Reactome"
        else {
            "database": ref["database"]
            + " "
            + {"P": "Pathway", "C": "Component", "F": "Function"}[
                ref.get("properties")[0]["value"][0]
            ],
            "entry": {
                "id": ref["id"],
                "name": ref.get("properties")[0]["value"][2::],
            },
        }
        for ref in data.get("uniProtKBCrossReferences", [])
        if ref["database"] in ["GO", "Reactome"]
    ]
    extracted[ExtractedFields.GOP] = [
        entry["entry"]
        for entry in annotation_references
        if entry["database"] == "GO Pathway"
    ]
    extracted[ExtractedFields.GOC] = [
        entry["entry"]
        for entry in annotation_references
        if entry["database"] == "GO Component"
    ]
    extracted[ExtractedFields.GOF] = [
        entry["entry"]
        for entry in annotation_references
        if entry["database"] == "GO Function"
    ]
    extracted[ExtractedFields.REACTOME] = [
        entry["entry"]
        for entry in annotation_references
        if entry["database"] == "Reactome"
    ]

    # TODO: Add caution comments

    return extracted


# TODO: Depracate once LLM is fed with protein id based information
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
    result = _request_uniprot_data(gene_name, organism_id)
    if result:
        result = result["results"][0]
    if (
        result
        and _extract_annotations_from_uniprot_data(result)[ExtractedFields.FUNCTIONCOMM]
    ):
        return str(
            _extract_annotations_from_uniprot_data(result)[ExtractedFields.FUNCTIONCOMM]
        )
    else:
        return "No data found"


def _request_uniprot_data_from_ids(ids: list) -> List[Union[str, dict]]:
    """
    Retrieve UniProt data for a list of protein IDs.
    Args:
        ids (list): A list of protein IDs (strings) to retrieve data for.
    Returns:
        List[Union[str, dict]]: A list containing the retrieved data for each protein ID.
            - If retrieval is successful, the result is a dictionary containing the UniProt data.
            - If retrieval fails, the result is the string "Retrieval failed".
    """

    results = [_request_uniprot_data(protein_id=id) for id in ids]
    results = [
        "Retrieval failed" if result is None else result["results"][0]
        for result in results
    ]
    return results


def _select_uniprot_result_from_feature(
    feature: str,
) -> Union[str, Dict]:
    """Get uniprot information for a feaure.

    This function collects the results for all base ids (truncating isoform ids) and selects the one to use for feeding the LLM.
    It does so by reducing the number of results until 1 remains and if that is not straight forward, selects the best annotated (sp over trembl, high annotation score).

    Arguments:
        feature (str): Semicolon separated list of Uniprot ids

    Returs:
        dict or str: Either a dictionary containing the information from Uniprot for the selected id, or a str if no (useful) data was retrieved.
    """

    baseids = sorted(
        list(set([identifier.split("-")[0] for identifier in feature.split(";")]))
    )
    results = _request_uniprot_data_from_ids(baseids)

    if len(results) == 1:
        return results[0]

    # remove inactive entries and failed retrievals (would be str instance)
    results = [
        result
        for result in results
        if isinstance(result, dict)
        and result.get("entryType", "Inactive") != "Inactive"
    ]

    if len(results) == 1:
        return results[0]
    elif len(results) == 0:
        return "No useful data found"

    # remove ones without gene names (besides immunoglobulins)
    results = [
        result
        for result in results
        if result.get("genes", None) is not None
        or bool(
            re.match(
                ".*globulin.*|^Ig.*",
                result.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", ""),
            )
        )
    ]

    if len(results) == 1:
        return results[0]
    elif len(results) == 0:
        return "No useful data found"

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
        sp_annotation_scores = [
            score for i, score in enumerate(annotation_scores) if i in sp_indices
        ]
        index = sp_indices[sp_annotation_scores.index(max(sp_annotation_scores))]
    return results[index]


def _filter_extracted_annotations(
    result: Union[Dict, str],
    fields: list,
) -> Union[str, Dict]:
    """
    Wrapper around extraction of meaningful entries from uniprot json output. Handles errors and allows limiting the retrieved field keys.
    Args:
        result (dict): The UniProt result from which to extract information.
        fields (list): A list of fields to extract from the result.
    Returns:
        dict or str: A dictionary containing the extracted field information, or the result itself if it is a string.
    """
    if isinstance(result, str):
        return result
    information = {
        k: v
        for k, v in _extract_annotations_from_uniprot_data(result).items()
        if k in fields
    }
    return information


def _format_uniprot_field(
    field: str, content: Union[str, List, Dict, None]
) -> Union[str, None]:
    """
    Formats the content of a UniProt field based on the specified field type.
    Args:
        field (str): The type of UniProt field to format. Should be one of the ExtractedFields enum values.
        content (Union[str, List, Dict, None]): The content to format. The type and structure of this parameter depend on the field type.
    Returns:
        Union[str, None]: A formatted string representation of the content for the specified field, or None if the content is None or empty.
    """

    if content is None:
        return None
    if field == ExtractedFields.NAME:
        return (
            None
            if content["recommendedName"] is None
            else " is called " + content["recommendedName"]
            if "alternativeNames" not in content
            else " is called "
            + content["recommendedName"]
            + " (or "
            + "/".join(content["alternativeNames"])
            + ")"
        )
    if field == ExtractedFields.GENE:
        return (
            " without a gene symbol"
            if "geneName" not in content or content["geneName"] is None
            else " " + content["geneName"]
        )
    if field in [
        ExtractedFields.FUNCTIONCOMM,
        ExtractedFields.SUBUNITCOMM,
        ExtractedFields.TISSUE,
    ]:
        return " ".join(content) if len(content) > 0 else None
    if field == ExtractedFields.INTERACTIONS:
        return (
            "Interacts with " + ", ".join([i["interactor"] for i in content]) + "."
            if len(content) > 0
            else None
        )
    if field == ExtractedFields.SUBCELL:
        return "Locates to " + ", ".join(content) + "." if len(content) > 0 else None
    if field == ExtractedFields.GOP:
        return (
            "The protein is part of the GO cell biological pathway(s) "
            + ", ".join([el["name"] for el in content])
            + "."
            if len(content) > 0
            else None
        )
    if field == ExtractedFields.GOC:
        return (
            "Locates to "
            + ", ".join([el["name"] for el in content])
            + " by GO annotation."
            if len(content) > 0
            else None
        )
    if field == ExtractedFields.GOF:
        return (
            "By GO annotation the proteins molecular function(s) are "
            + ", ".join([el["name"] for el in content])
            + "."
            if len(content) > 0
            else None
        )
    if field == ExtractedFields.REACTOME:
        return (
            "The protein is part of the Reactome pathways "
            + ", ".join([el["name"] for el in content])
            + "."
            if len(content) > 0
            else None
        )
    return f"{field} of this protein is {str(content)}."


def get_annotations_for_feature(
    feature: str,
) -> Union[Dict, str]:
    """
    Retrieve annotations for a given feature from UniProt, after selecting the best suitable id to represent it.
    Args:
        feature (str): The feature for which annotations are to be retrieved.
    Returns:
        Union[Dict, str]: A dictionary containing the annotations if found,
                          otherwise a string indicating an error or empty result.
    """
    fields = ExtractedFields.get_values()
    result = _select_uniprot_result_from_feature(feature)
    annotations = _filter_extracted_annotations(result, fields)
    return annotations


def format_uniprot_annotation(information: dict, fields: list = None) -> str:
    """
    Formats UniProt annotation information into a readable string.
    Args:
        information (dict): A dictionary containing UniProt annotation data.
        fields (list, optional): A list of fields to include in the formatted output.
                                 If None, all fields in the information dictionary are included.
    Returns:
        str: A formatted string containing the requested UniProt annotation information.
    """

    if fields is None:
        fields = list(information.keys())

    # get requested fields
    texts = {
        field: _format_uniprot_field(field, information.get(field)) for field in fields
    }
    # remove empty fields
    texts = {field: text for field, text in texts.items() if text is not None}

    # assemble text
    if any(el in texts for el in ["genes", "protein"]):
        assembled_text = (
            "The protein" + texts.get("genes", "") + texts.get("protein", "") + "."
        )
    else:
        assembled_text = ""
    if any(key not in ["genes", "protein"] for key in texts):
        assembled_text += "\nUniprot information:\n- "
        assembled_text += "\n- ".join(
            [
                text
                for field, text in texts.items()
                if field not in ["genes", "protein"] and text is not None
            ]
        )

    return assembled_text
