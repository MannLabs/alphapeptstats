import logging
import re
import time
from typing import Dict, List, Union

import requests

from alphastats.dataset.keys import ConstantsClass

BASE_URL = "https://rest.uniprot.org/idmapping/"
ID_MAPPING_RUN_URL = BASE_URL + "run"
ID_MAPPING_STATUS_URL = BASE_URL + "status/"
ID_MAPPING_RESULTS_URL = BASE_URL + "uniprotkb/results/"
DEFAULT_TIMEOUT_S = 300
POLL_INTERVAL_S = 3
RETRIEVAL_FAILED_MESSAGE = "Retrieval failed"

_logger = logging.getLogger(__name__)


class ExtractedUniprotFields(metaclass=ConstantsClass):
    """Dictionary keys for the extracted uniprot fields, as created by _extract_annotations_from_uniprot_data.

    The order also determines the order in which the fields are displayed in the interface and largely also the formatted output."""

    DB = "entryType"
    ID = "primaryAccession"
    SECONDARYACC = "secondaryAccessions"
    NAME = "protein"
    GENE = "genes"
    FUNCTIONCOMM = "functionComments"
    SUBUNITCOMM = "subunitComments"
    CAUTIONCOMM = "cautionComments"
    SUBCELL = "subcellularLocations"
    TISSUE = "tissueSpecificity"
    INTERACTIONS = "interactions"
    GOP = "GO Pathway"
    GOF = "GO Function"
    GOC = "GO Component"
    REACTOME = "Reactome"


def _request_uniprot_data(
    protein_id: str = None,
) -> Dict:
    """
    Get data from UniProt for a given Uniprot identifier.

    Args:
        protein_id (str): Uniprot identifier of a protein

    Returns:
        dict: The data retrieved from UniProt, if retrieval was successful. If the response code was not 200 or there were no results the dictionary is empty.
    """
    base_url = "https://rest.uniprot.org/uniprotkb"
    if protein_id is not None:
        query = f"{base_url}/{protein_id}"
    else:
        raise ValueError("Please provide a protein id.")

    response = requests.get(query, headers={"Accept": "application/json"})

    if response.status_code != 200:
        print(
            f"Failed to retrieve data for {query}. Status code: {response.status_code}"
        )
        print(response.text)
        return {}

    data = response.json()

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
    extracted[ExtractedUniprotFields.DB] = data.get("entryType")

    # 2. Primary Accession
    extracted[ExtractedUniprotFields.ID] = data.get("primaryAccession")

    # 3. Secondary Accessions
    extracted[ExtractedUniprotFields.SECONDARYACC] = data.get("secondaryAccessions")

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
    extracted[ExtractedUniprotFields.NAME] = {
        "recommendedName": recommended_name,
        "alternativeNames": alternative_names,
        "flag": protein_description.get("flag", None),
    }

    # 5. Gene Details
    genes = data.get("genes", [{}])[0]
    extracted[ExtractedUniprotFields.GENE] = {
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
    extracted[ExtractedUniprotFields.FUNCTIONCOMM] = function_comments

    # 7. Subunit Details
    subunit_comments = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "SUBUNIT"
        for text in comment.get("texts", [])
    ]
    extracted[ExtractedUniprotFields.SUBUNITCOMM] = subunit_comments

    # 7.1. Caution Comments
    caution_comments = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "CAUTION"
        for text in comment.get("texts", [])
    ]
    extracted[ExtractedUniprotFields.CAUTIONCOMM] = caution_comments

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
    extracted[ExtractedUniprotFields.INTERACTIONS] = interactions

    # 9. Subcellular Locations
    subcellular_locations_comments = [
        c.get(
            "subcellularLocations", []
        )  # sucbcellular location comments without subcellular locations
        for c in data.get("comments", [])
        if c["commentType"] == "SUBCELLULAR LOCATION"
    ]
    locations = [
        location["location"]["value"]
        for locations_comment in subcellular_locations_comments
        for location in locations_comment
    ]
    extracted[ExtractedUniprotFields.SUBCELL] = locations

    # 10. Tissue specificity
    tissue_specificities = [
        text["value"]
        for comment in data.get("comments", [])
        if comment["commentType"] == "TISSUE SPECIFICITY"
        for text in comment.get("texts", [])
    ]
    extracted[ExtractedUniprotFields.TISSUE] = tissue_specificities

    # 14. Pathway references
    extracted[ExtractedUniprotFields.GOP] = []
    extracted[ExtractedUniprotFields.GOC] = []
    extracted[ExtractedUniprotFields.GOF] = []
    extracted[ExtractedUniprotFields.REACTOME] = []
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref["database"] == "Reactome":
            database = ExtractedUniprotFields.REACTOME
            entry = {
                "id": ref["id"],
                "name": ref.get("properties")[0]["value"],
            }
        elif ref["database"] == "GO":
            database = {
                "P": ExtractedUniprotFields.GOP,
                "C": ExtractedUniprotFields.GOC,
                "F": ExtractedUniprotFields.GOF,
            }[ref.get("properties")[0]["value"][0]]
            entry = {
                "id": ref["id"],
                "name": ref.get("properties")[0]["value"][2::],
            }
        else:
            continue
        extracted[database].append(entry)

    return extracted


def _fetch_uniprot_mapping_results(ids: List[str]) -> Dict[str, dict]:
    """Submit *ids* to the UniProt ID mapping service and return a mapping.

    The returned dict maps **every** recognised accession - both the original
    *from* accession as supplied by the caller *and* the canonical
    ``primaryAccession`` returned by UniProt - to the full UniProtKB record
    (``dict``).  This guarantees that look-ups via deprecated, secondary or
    isoform accessions resolve correctly and eliminates the accidental loss of
    entries when multiple inputs collapse onto the same canonical record.

    Retrieve UniProt data for a list of protein IDs.
    Args:
        ids (list): A list of protein IDs (strings) to retrieve data for.
    Returns:
        Dict[str, dict]: A mapping of protein IDs to their corresponding UniProt data.
            - If retrieval is successful, the result is a dictionary containing the UniProt data.
            - If retrieval fails, the result is an empty dictionary.
    """

    try:
        resp = requests.post(
            ID_MAPPING_RUN_URL,
            data={
                "ids": ",".join(ids),
                "from": "UniProtKB_AC-ID",
                "to": "UniProtKB",
            },
        )
        resp.raise_for_status()
        job_id = resp.json().get("jobId")
        if not job_id:
            _logger.warning("UniProt returned no job ID")
            return {}

        start = time.time()
        while True:
            status_resp = requests.get(
                f"{ID_MAPPING_STATUS_URL}{job_id}", allow_redirects=False
            )
            if status_resp.status_code == 303 or "Location" in status_resp.headers:
                # Results are ready
                break
            if time.time() - start > DEFAULT_TIMEOUT_S:
                _logger.warning("UniProt mapping job timed out")
                return {}
            time.sleep(POLL_INTERVAL_S)

        results_resp = requests.get(
            f"{ID_MAPPING_RESULTS_URL}{job_id}", params={"size": 500}
        )
        results_resp.raise_for_status()

        data = results_resp.json()
        all_results = data.get("results", [])

        while data.get("next"):
            results_resp = requests.get(data["next"])
            results_resp.raise_for_status()
            data = results_resp.json()
            all_results.extend(data.get("results", []))

        mapping: Dict[str, dict] = {}
        for item in all_results:
            to_entry = item.get("to")
            if not isinstance(to_entry, dict):
                continue

            from_acc = item.get("from")
            if from_acc:
                mapping[from_acc] = to_entry

            primary_acc = to_entry.get("primaryAccession")
            if primary_acc:
                mapping[primary_acc] = to_entry

        return mapping

    except Exception as exc:
        _logger.warning("UniProt mapping failed: %s", exc)
        return {}


def _request_uniprot_data_from_ids(ids: list) -> List[Union[str, dict]]:
    """Retrieve UniProtKB records for *ids* in bulk.

    The helper is *lenient* regarding the exact input format:
    • Each element in *ids* may be a **single accession** (e.g. ``"P60709"``)
      *or* a **semicolon-separated list** of accessions (matching the format
      used in the dataset feature column, e.g. ``"P10643;A8K2T4"``).
    • Isoform suffixes (``"-1"``) are stripped because the mapping API only
      accepts canonical base accessions.

    The function performs *one* mapping call for the union of all input
    accessions and then, for every original element in *ids*, returns the first
    successfully mapped record (dict).  Unresolved entries yield the sentinel
    string RETRIEVAL_FAILED_MESSAGE so callers can distinguish them.
    """

    if not ids:
        return []

    all_base_ids: list[str] = []
    for identifier in ids:
        for token in identifier.split(";"):
            base = token.split("-")[0]
            if base and base not in all_base_ids:
                all_base_ids.append(base)

    mapping_dict = _fetch_uniprot_mapping_results(all_base_ids)

    if not mapping_dict:
        return [RETRIEVAL_FAILED_MESSAGE] * len(ids)

    resolved: List[Union[str, dict]] = []
    for identifier in ids:
        annotation: Union[str, dict] = RETRIEVAL_FAILED_MESSAGE
        for token in identifier.split(";"):
            base = token.split("-")[0]
            result = mapping_dict.get(base)
            if isinstance(result, dict):
                annotation = result
                break
        resolved.append(annotation)

    return resolved


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

    results = _select_valid_unprot_results(results)

    if len(results) == 1:
        return results[0]
    elif len(results) == 0:
        return "No useful data found"

    best_result = _select_best_annotated_uniprot_result(results)
    return best_result


def _select_valid_unprot_results(results) -> List[Dict]:
    """
    Filter out invalid uniprot results. This includes inactive entries and if multiple ones remain, entries without gene names (except for immunoglobulins).

    Args:
        results (list): List of uniprot results.

    Returns:
        list: List of valid uniprot results.
    """
    # remove inactive entries and failed retrievals (would be str instance)
    results = [
        result
        for result in results
        if isinstance(result, dict)
        and result.get("entryType", "Inactive") != "Inactive"
    ]

    if len(results) <= 1:
        return results

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
    return results


def _select_best_annotated_uniprot_result(results) -> Dict:
    """Going by quality of annotation select the best uniprot result.
    It prioritizes swissprot entries over trembl entries and entries with higher annotation scores over lower ones.

    Args:
        results (list): List of uniprot results.

    Returns:
        dict: The best annotated uniprot result."""
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

    n_sp_indices = len(sp_indices)
    n_gene_names = len(set(gene_names))

    if (n_gene_names == 1 and n_sp_indices > 0) or (
        n_gene_names > 1 and n_sp_indices == 1
    ):
        # Either all the same gene name and any swissprot entries, or multiple gene names but only one swissprot entry
        index = sp_indices[0]
    elif n_sp_indices == 0:
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
    if (
        content is None
        or len(content) == 0
        or field not in ExtractedUniprotFields.get_values()
    ):
        return None
    if field == ExtractedUniprotFields.NAME:
        if content["recommendedName"] is None:
            return None
        result = f" is called {content['recommendedName']}"
        if alt_names := content.get("alternativeNames"):
            result += f" (or {'/'.join(alt_names)})"
        return result
    if field == ExtractedUniprotFields.GENE:
        return (
            " without a gene symbol"
            if "geneName" not in content or content["geneName"] is None
            else " " + content["geneName"]
        )
    if field == ExtractedUniprotFields.INTERACTIONS:
        return (
            "Interacts with " + ", ".join([i["interactor"] for i in content]) + "."
            if len(content) > 0
            else None
        )
    if field in [
        ExtractedUniprotFields.FUNCTIONCOMM,
        ExtractedUniprotFields.SUBUNITCOMM,
        ExtractedUniprotFields.TISSUE,
        ExtractedUniprotFields.CAUTIONCOMM,
    ]:
        return " ".join(content)

    if isinstance(content, list):
        elements = ", ".join(
            [el if isinstance(el, str) else el["name"] for el in content]
        )
        formatstrings = {
            ExtractedUniprotFields.SUBCELL: "Locates to {}.",
            ExtractedUniprotFields.GOP: "The protein is part of the GO cell biological pathway(s) {}.",
            ExtractedUniprotFields.GOC: "Locates to {} by GO annotation.",
            ExtractedUniprotFields.GOF: "According to GO annotations the proteins molecular function(s) are {}.",
            ExtractedUniprotFields.REACTOME: "The protein is part of the Reactome pathways {}.",
        }
        return formatstrings.get(field, field + " of this protein is {}.").format(
            elements
        )

    return f"{field} of this protein is {str(content)}."


def __get_annotations_for_feature(
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
    fields = ExtractedUniprotFields.get_values()
    result = _select_uniprot_result_from_feature(feature)
    annotations = _filter_extracted_annotations(result, fields)
    return annotations


def get_uniprot_state_key(selected_analysis_key: str) -> str:
    """Get analysis specific session state key for uniprot integration checkbox."""
    return f"{selected_analysis_key}_integrate_uniprot".replace(" ", "")


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

    if isinstance(information, str):
        return information

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


def get_annotations_for_features(
    features: List[str],
) -> Dict[str, Union[Dict, str]]:  # noqa: D401
    """Retrieve annotations for many features with a single UniProt mapping call.

    This minimises network traffic by batching all base accessions across *features*
    into one mapping job via :pyfunc:`_request_uniprot_data_from_ids` and then
    selecting the best-annotated entry for each feature using the existing
    selection helpers.

    Parameters
    ----------
    features : list[str]
        Semicolon-separated UniProt accession strings as used in the dataset.

    Returns
    -------
    dict
        Mapping *feature* → annotation (dict or RETRIEVAL_FAILED_MESSAGE string).
    """
    fields = ExtractedUniprotFields.get_values()
    if not features:
        return {}

    feature_to_baseids = {
        feature: [identifier.split("-")[0] for identifier in feature.split(";")]
        for feature in features
    }

    all_ids: List[str] = list(
        dict.fromkeys(
            base_id for baseids in feature_to_baseids.values() for base_id in baseids
        )
    )

    all_results = _request_uniprot_data_from_ids(all_ids)
    id_to_result = dict(zip(all_ids, all_results))

    annotations: Dict[str, Union[Dict, str]] = {}

    for feature, baseids in feature_to_baseids.items():
        annotation: Union[Dict, str] = RETRIEVAL_FAILED_MESSAGE
        for base_id in baseids:
            result = id_to_result.get(base_id, RETRIEVAL_FAILED_MESSAGE)
            if isinstance(result, dict):
                annotation = result
                break
        annotations[feature] = _filter_extracted_annotations(annotation, fields)
    return annotations
