import copy
from typing import Optional, Union
from pathlib import Path
import requests

import time
import random
import json

from Bio import Entrez
import openai
import pandas as pd
import streamlit as st

from alphastats.plots.DimensionalityReduction import DimensionalityReduction


Entrez.email = "lebedev_mikhail@outlook.com"  # Always provide your email address when using NCBI services.

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


def get_subgroups_for_each_group(
    metadata: pd.DataFrame,
) -> dict:
    """
    Get the unique values for each column in the metadata file.

    Args:
        metadata (pd.DataFrame, optional): The metadata dataframe (which sample has which disease/treatment/condition/etc).

    Returns:
        dict: A dictionary with the column names as keys and a list of unique values as values.
    """
    groups = [str(i) for i in metadata.columns.to_list()]
    group_to_subgroup_values = {
        group: get_unique_values_from_column(group, metadata=metadata)
        for group in groups
    }
    return group_to_subgroup_values


def get_unique_values_from_column(column: str, metadata: pd.DataFrame) -> list[str]:
    """
    Get the unique values from a column in the metadata file.

    Args:
        column (str): The name of the column in the metadata file.
        metadata (pd.DataFrame, optional): The metadata dataframe (which sample has which disease/treatment/condition/etc).

    Returns:
        list[str]: A list of unique values from the column.
    """
    unique_values = metadata[column].unique().tolist()
    return [str(i) for i in unique_values]


def display_proteins(overexpressed: list[str], underexpressed: list[str]) -> None:
    """
    Display a list of overexpressed and underexpressed proteins in a Streamlit app.

    Args:
        overexpressed (list[str]): A list of overexpressed proteins.
        underexpressed (list[str]): A list of underexpressed proteins.
    """

    # Start with the overexpressed proteins
    link = "https://www.uniprot.org/uniprotkb?query="
    overexpressed_html = "".join(
        f'<a href = {link + protein}><li style="color: green;">{protein}</li></a>'
        for protein in overexpressed
    )
    # Continue with the underexpressed proteins
    underexpressed_html = "".join(
        f'<a href = {link + protein}><li style="color: red;">{protein}</li></a>'
        for protein in underexpressed
    )

    # Combine both lists into one HTML string
    full_html = f"<ul>{overexpressed_html}{underexpressed_html}</ul>"

    # Display in Streamlit
    st.markdown(full_html, unsafe_allow_html=True)


def get_assistant_functions(
    gene_to_prot_id_dict: dict,
    metadata: pd.DataFrame,
    subgroups_for_each_group: dict,
) -> list[dict]:
    """
    Get a list of assistant functions for function calling in the ChatGPT model.
    You can call this function with no arguments, arguments are given for clarity on what changes the behavior of the function.
    For more information on how to format functions for Assistants, see https://platform.openai.com/docs/assistants/tools/function-calling

    Args:
        gene_to_prot_id_dict (dict, optional): A dictionary with gene names as keys and protein IDs as values.
        metadata (pd.DataFrame, optional): The metadata dataframe (which sample has which disease/treatment/condition/etc).
        subgroups_for_each_group (dict, optional): A dictionary with the column names as keys and a list of unique values as values. Defaults to get_subgroups_for_each_group().
    Returns:
        list[dict]: A list of assistant functions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_gene_function",
                "description": "Get the gene function and description by UniProt lookup of gene identifier/name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene_name": {
                            "type": "string",
                            "description": "Gene identifier/name for UniProt lookup",
                        },
                    },
                    "required": ["gene_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "st.session_state.dataset.plot_intensity",
                "description": "Create an intensity plot based on protein data and analytical methods.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "protein_id": {
                            "type": "string",
                            "enum": [i for i in gene_to_prot_id_dict.keys()],
                            "description": "Identifier for the protein of interest",
                        },
                        "group": {
                            "type": "string",
                            "enum": [str(i) for i in metadata.columns.to_list()],
                            "description": "Column name in the dataset for the group variable",
                        },
                        "subgroups": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Specific subgroups within the group to analyze. For each group you need to look up the subgroups in the dict {str(get_subgroups_for_each_group(st.session_state['dataset'].metadata))} or present user with them first if you are not sure what to choose",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["violin", "box", "scatter", "all"],
                            "description": "The method of plot to create",
                        },
                        "add_significance": {
                            "type": "boolean",
                            "description": "Whether to add significance markers to the plot",
                        },
                        "log_scale": {
                            "type": "boolean",
                            "description": "Whether to use a logarithmic scale for the plot",
                        },
                    },
                    "required": ["protein_id", "group"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "perform_dimensionality_reduction",
                "description": "Perform dimensionality reduction on a given dataset and generate a plot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group": {
                            "type": "string",
                            "description": "The name of the group column in the dataset",
                            "enum": [str(i) for i in metadata.columns.to_list()],
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pca", "umap", "tsne"],
                            "description": "The dimensionality reduction method to apply",
                        },
                        "circle": {
                            "type": "boolean",
                            "description": "Flag to draw circles around groups in the scatterplot",
                        },
                    },
                    "required": ["group", "method", "circle"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "st.session_state.dataset.plot_sampledistribution",
                "description": "Generates a histogram plot for each sample in the dataset matrix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "The name of the group column in the dataset to color the samples by",
                            "enum": [str(i) for i in metadata.columns.to_list()],
                        },
                        "method": {
                            "type": "string",
                            "enum": ["violin", "box"],
                            "description": "The method of plot to create",
                        },
                    },
                    "required": ["group", "method"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "st.session_state.dataset.plot_volcano",
                "description": "Generates a volcano plot based on two subgroups of the same group",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "items": {"type": "string"},
                            "description": f"Column name in the dataset for the group variable. Must be from {str(subgroups_for_each_group.keys())} and group1 and group2 must be from THIS very group.",
                        },
                        "group1": {
                            "type": "string",
                            "items": {"type": "string"},
                            "description": f"Specific subgroup within the group to analyze. For each group you get from prompt you need to look up the subgroups in the dict {str(subgroups_for_each_group)} or present user with them first if you are not sure what to choose. You can use ONLY 1.",
                        },
                        "group2": {
                            "type": "string",
                            "items": {"type": "string"},
                            "description": f"Second subgroup from the same group in {str(subgroups_for_each_group)} or present user with them first if you are not sure what to choose. You can use ONLY 1.",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["wald", "ttest", "anova", "welch-ttest", "sam"],
                            "description": "The method of plot to create",
                        },
                        "labels": {
                            "type": "boolean",
                            "description": "Whether to add gene names to the plot",
                        },
                        "min_fc": {
                            "type": "string",
                            "enum": ["0", "1", "2"],
                            "description": "Minimal foldchange cutoff that is considered significant",
                        },
                        "alpha": {
                            "type": "string",
                            "enum": ["0.01", "0.02", "0.03", "0.04", "0.05"],
                            "description": "Alpha value for significance",
                        },
                        "draw_line": {
                            "type": "boolean",
                            "description": "Whether to draw lines for significance",
                        },
                        "perm": {
                            "type": "string",
                            "enum": ["1", "10", "100", "1000"],
                            "description": "Number of permutations for SAM",
                        },
                        "fdr": {
                            "type": "string",
                            "enum": ["0.005", "0.01", "0.05", "0.1"],
                            "description": "False Discovery Rate cutoff for SAM",
                        },
                    },
                "required": ["column", "group1", "group2"],
                },
            },
        },
        {"type": "code_interpreter"},
    ]


def perform_dimensionality_reduction(group, method, circle, **kwargs):
    dr = DimensionalityReduction(
        st.session_state.dataset, group, method, circle, **kwargs
    )
    return dr.plot


def get_uniprot_data(
    gene_name: str,
    organism_id: str,
    fields: list[str] = uniprot_fields,
) -> dict:
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
    for key, value in data.items():
        print(f"data - {key}: {value}, {type(value)}")
    return data


def extract_data(data: dict) -> dict:
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
        c["texts"][0]["value"]
        for c in data.get("comments", [])
        if c["commentType"] == "FUNCTION"
    ]

    extracted["functionComments"] = function_comments

    # 7. Subunit Details
    subunit_comments = [
        c["texts"][0]["value"]
        for c in data.get("comments", [])
        if c["commentType"] == "SUBUNIT"
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
    locations = [
        c["subcellularLocations"][0]["location"]["value"]
        for c in data.get("comments", [])
        if c["commentType"] == "SUBCELLULAR LOCATION"
    ]
    extracted["subcellularLocations"] = locations

    # 10. Tissue Specificity
    tissue_specificities = [
        c["texts"][0]["value"]
        for c in data.get("comments", [])
        if c["commentType"] == "TISSUE SPECIFICITY"
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


def get_info(genes_list: list[str], organism_id: str) -> list[str]:
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
            gene_functions.append(f"{gene}: {results[gene]['functionComments'][0]}")
        else:
            gene_functions.append(f"{gene}: ?")

    return gene_functions


def get_gene_function(gene_name: Union[str, dict], organism_id: str = None) -> str:
    """
    Get the gene function and description by UniProt lookup of gene identifier / name.

    Args:
        gene_name (Union[str, dict]): Gene identifier / name for UniProt lookup.
        organism_id (str): The UniProt organism ID to search in.

    Returns:
        str: The gene function and description.
    """
    if not organism_id:
        organism_id = st.session_state["organism"]
    if type(gene_name) == dict:
        gene_name = gene_name["gene_name"]
    result = get_uniprot_data(gene_name, organism_id)
    if result and extract_data(result)["functionComments"]:
        return extract_data(result)["functionComments"][0]
    else:
        return "No data found"


def turn_args_to_float(json_string: Union[str, bytes, bytearray]) -> dict:
    """
    Turn all values in a JSON string to floats if possible.

    Args:
        json_string (Union[str, bytes, bytearray]): The JSON string to convert.

    Returns:
        dict: The converted JSON string as a dictionary.
    """
    data = json.loads(json_string)
    for key, value in data.items():
        if isinstance(value, str):
            try:
                data[key] = float(value)
            except ValueError:
                continue
    return data


def get_gene_to_prot_id_mapping(gene_id: str) -> str:
    """Get protein id from gene id. If gene id is not present, return gene id, as we might already have a gene id.
    'VCL;HEL114' -> 'P18206;A0A024QZN4;V9HWK2;B3KXA2;Q5JQ13;B4DKC9;B4DTM7;A0A096LPE1'
    Args:
        gene_id (str): Gene id

    Returns:
        str: Protein id or gene id if not present in the mapping.
    """
    import streamlit as st
    session_state_copy = dict(copy.deepcopy(st.session_state))
    if "gene_to_prot_id" not in session_state_copy:
        session_state_copy["gene_to_prot_id"] = {}
    if gene_id in session_state_copy["gene_to_prot_id"]:
        return session_state_copy["gene_to_prot_id"][gene_id]
    for gene, prot_id in session_state_copy["gene_to_prot_id"].items():
        if gene_id in gene.split(";"):
            return prot_id
    return gene_id


def wait_for_run_completion(
    client: openai.OpenAI, thread_id: int, run_id: int, check_interval: int = 2
) -> Optional[list]:
    """
    Wait for a run and function calls to complete and return the plots, if they were created by function calling.

    Args:
        client (openai.OpenAI): The OpenAI client.
        thread_id (int): The thread ID.
        run_id (int): The run ID.
        check_interval (int, optional): The interval to check for run completion. Defaults to 2.

    Returns:
        Optional[list]: A list of plots, if any.
    """
    plots = []
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_id
        )
        plot_functions = {
            "create_intensity_plot",
            "perform_dimensionality_reduction",
            "create_sample_histogram",
            "st.session_state.dataset.plot_volcano",
            "st.session_state.dataset.plot_sampledistribution",
            "st.session_state.dataset.plot_intensity",
            "st.session_state.dataset.plot_pca",
            "st.session_state.dataset.plot_umap",
            "st.session_state.dataset.plot_tsne",
        }
        if run_status.status == "completed":
            print("Run is completed!")
            if plots:
                print("Returning plots")
                return plots
            break
        elif run_status.status == "requires_action":
            print("requires_action", run_status)
            print(
                [
                    st.session_state.plotting_options[i]["function"].__name__
                    for i in st.session_state.plotting_options
                ]
            )
            print(plot_functions)
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                print("### calling:", tool_call.function.name)
                if tool_call.function.name == "get_gene_function":
                    print(
                        type(tool_call.function.arguments), tool_call.function.arguments
                    )
                    prompt = json.loads(tool_call.function.arguments)["gene_name"]
                    image_url = get_gene_function(prompt)
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": image_url,
                        },
                    )
                elif (
                    tool_call.function.name
                    in [
                        st.session_state.plotting_options[i]["function"].__name__
                        for i in st.session_state.plotting_options
                    ]
                    or tool_call.function.name in plot_functions
                ):
                    args = tool_call.function.arguments
                    args = turn_args_to_float(args)
                    print(f"{tool_call.function.name}(**{args})")
                    image = eval(f"{tool_call.function.name}(**{args})")
                    image_json = image.to_json()

                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "output": image_json},
                    )
                    plots.append(image)

            if tool_outputs:
                _run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run_id, tool_outputs=tool_outputs
                )
        else:
            print("Run is not yet completed. Waiting...", run_status.status, run_id)
            time.sleep(check_interval)


def send_message_save_thread(client: openai.OpenAI, message: str) -> Optional[list]:
    """
    Send a message to the OpenAI ChatGPT model and save the thread in the session state, return plots if GPT called a function to create them.

    Args:
        client (openai.OpenAI): The OpenAI client.
        message (str): The message to send to the ChatGPT model.

    Returns:
        Optional[list]: A list of plots, if any.
    """
    message = client.beta.threads.messages.create(
        thread_id=st.session_state["thread_id"], role="user", content=message
    )

    run = client.beta.threads.runs.create(
        thread_id=st.session_state["thread_id"],
        assistant_id=st.session_state["assistant"].id,
    )
    try:
        plots = wait_for_run_completion(client, st.session_state["thread_id"], run.id)
    except KeyError as e:
        print(e)
        plots = None
    messages = client.beta.threads.messages.list(
        thread_id=st.session_state["thread_id"]
    )
    st.session_state.messages = []
    for num, message in enumerate(messages.data[::-1]):
        role = message.role
        if message.content:
            content = message.content[0].text.value
        else:
            content = "Sorry, I was unable to process this message. Try again or change your request."
        st.session_state.messages.append({"role": role, "content": content})
    if not plots:
        return
    return plots


def try_to_set_api_key(api_key: str = None) -> None:
    """
    Checks if the OpenAI API key is available in the environment / system variables.
    If the API key is not available, saves the key to secrets.toml in the repository root directory.

    Args:
        api_key (str, optional): The OpenAI API key. Defaults to None.

    Returns:
        None
    """
    if api_key and "api_key" not in st.session_state:
        st.session_state["openai_api_key"] = api_key
        secret_path = Path(st.secrets._file_paths[-1])
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        with open(secret_path, "w") as f:
            f.write(f'openai_api_key = "{api_key}"')
        openai.OpenAI.api_key = api_key
        return
    try:
        openai.OpenAI.api_key = st.secrets["openai_api_key"]
    except:
        st.write(
            "OpenAI API key not found in environment variables. Please enter your API key to continue."
        )
