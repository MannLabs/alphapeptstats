"""This module contains functions to generate prompts for the LLM model."""

import os
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageToolCall

from alphastats.dataset.dataset import DataSet
from alphastats.llm.llm_utils import get_subgroups_for_each_group


def get_system_message(dataset: DataSet) -> str:
    """Get the system message for the LLM model."""
    subgroups = get_subgroups_for_each_group(dataset.metadata)

    newline = os.linesep

    return (
        f"You are a proteomics expert specializing in molecular biology, biochemistry, and systems biology.{newline}"
        f"Analyze the differentially expressed proteins (upregulated and downregulated) from our proteomics experiment comparing two conditions, focusing on protein connections and potential disease roles.{newline}{newline}"
        f"Format your response with:{newline}"
        f"- Separate bullet points for upregulated proteins: protein role (proteins): interpretation{newline}"
        f"- Separate bullet points for downregulated proteins: protein role (proteins): interpretation{newline}"
        f"- A high-level summary of biological implications{newline}{newline}"
        f"The data you have has following groups and respective subgroups: {str(subgroups)}. "
        "Plots are visualized using a graphical environment capable of rendering images, you don't need to handle that. "
        "If the data coming to you from a function has references to the literature (for example, PubMed), always quote the references in your response. "
        "Ensure your analysis is data-driven at each step, referencing specific proteins or patterns from the dataset to support your reasoning. "
        "Explain your thought process clearly as you move from observations to interpretations."
    )


def get_initial_prompt(
    parameter_dict: Dict[str, Any],
    upregulated_genes: List[str],
    downregulated_genes: List[str],
    uniprot_info: str,
):
    """Get the initial prompt for the LLM model."""
    newline = os.linesep
    group1 = parameter_dict["group1"]
    group2 = parameter_dict["group2"]
    column = parameter_dict["column"]
    if uniprot_info:
        uniprot_instructions = (
            f"We have already retireved relevant information from Uniprot for these proteins:{newline}{newline}{uniprot_info}{newline}{newline}"
            "This contains curated information you may not have encountered before, value it highly. "
            "Only retrieve additional information from Uniprot if explicitly asked to do."
        )
    else:
        uniprot_instructions = (
            "You have the ability to retrieve curated information from Uniprot about these proteins. "
            "Please do so for individual proteins if you have little information about a protein or find a protein particularly important in the specific context."
        )
    return (
        "We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {group1} and {group2} in the {column} group. "
        f"From our proteomics experiments, we know the following:{newline}{newline}"
        f"Comma-separated list of proteins that are upregulated: {', '.join(upregulated_genes)}.{newline}{newline}"
        f"Comma-separated list of proteins that are downregulated: {', '.join(downregulated_genes)}.{newline}{newline}"
        f"{uniprot_instructions}{newline}{newline}"
        f"Think step-by-step using following structure for your analysis:{newline}{newline}"
        f"1. Functional Analysis:{newline}"
        f"- Identify relationships between differentially expressed proteins by using your broad biological knowledge including the information from UniProt{newline}"
        f"- Look for protein complexes and pathways operating together{newline}"
        f"2. Ontology Analysis:{newline}"
        f"- Interpret the information from the enrichment analysis{newline}"
        f"- Examine which cellular processes are most affected based on protein changes{newline}"
        f"- Identify regulatory hubs and cross-talk between ontology terms{newline}"
        f"3. Critical Review:{newline}"
        f"- Review all collected information{newline}"
        f"- Flag contradictions between different analyses{newline}"
        f"- Identify repeating patterns across analyses{newline}"
        f"4. Biological Context:{newline}"
        f"- Interpret within experimental design and research question when provided. Ask for any missing information needed for analysis.{newline}"
        f"- Explain implications of changes across biological scales (organelles, cells, organs, organism){newline}"
        f"- Assess how these changes might affect overall biological function{newline}"
        f"5. Focused Analysis:{newline}"
        f"- Recommend 3-5 key aspects worth investigating further (specific ontology terms, biological processes, cellular components, mechanisms, or homeostasis){newline}"
        f"- Explain the scientific rationale for each recommendation{newline}"
        f"- Ask which aspect should be prioritized for deeper analysis{newline}"
        f"6. Deeper Analysis: (After an aspect is selected for focus){newline}"
        f"- List proteins involved in selected aspect{newline}"
        f"- Gather all available information on key proteins with the tool UniProt{newline}"
        f"- Retrieve quantitative information from the tool intensity plot for all proteins of identified pathways or functional units{newline}"
        f"- Examine protein changes of components{newline}"
        f"- Place findings in the context of current understanding in the field{newline}"
        f"- Identify aspects that confirm or challenge existing models{newline}"
        f"- Develop hypotheses explaining observed patterns{newline}"
        f"- Suggest follow-up experiments to validate your hypotheses{newline}"
    )


def get_tool_call_message(tool_calls: List[ChatCompletionMessageToolCall]) -> str:
    """Get a string representation of the tool calls made by the LLM model."""
    return "\n".join(
        [
            f"Calling function: {tool_call.function.name} "
            f"with arguments: {tool_call.function.arguments}"
            for tool_call in tool_calls
        ]
    )
