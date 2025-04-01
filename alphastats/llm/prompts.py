"""This module contains functions to generate prompts for the LLM model."""

import os
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageToolCall

from alphastats.dataset.dataset import DataSet
from alphastats.llm.llm_utils import get_subgroups_for_each_group


def get_system_message(dataset: DataSet) -> str:
    """Get the system message for the LLM model."""
    subgroups = get_subgroups_for_each_group(dataset.metadata)

    return (
        f"You are a proteomics expert specializing in molecular biology, biochemistry, and systems biology.{os.linesep}"
        f"Analyze the differentially expressed proteins (upregulated and downregulated) from the user's proteomics experiment comparing two conditions, focusing on protein connections and potential disease roles.{os.linesep}{os.linesep}"
        f"Format your response with:{os.linesep}"
        f"- Separate bullet points for upregulated proteins: protein role (proteins): interpretation{os.linesep}"
        f"- Separate bullet points for downregulated proteins: protein role (proteins): interpretation{os.linesep}"
        f"- A high-level summary of biological implications{os.linesep}{os.linesep}"
        f"The data you have has following groups and respective subgroups: {str(subgroups)}."
        f"Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that. "
        f"If the data coming to you from a function has references to the literature (for example, PubMed), always quote the references in your response. "
        f"Ensure your analysis is data-driven at each step, referencing specific proteins or patterns from the dataset to support your reasoning. "
        f"Explain your thought process clearly as you move from observations to interpretations."
    )


def get_initial_prompt(
    parameter_dict: Dict[str, Any],
    upregulated_genes: List[str],
    downregulated_genes: List[str],
    uniprot_info: str,
):
    """Get the initial prompt for the LLM model."""
    group1 = parameter_dict["group1"]
    group2 = parameter_dict["group2"]
    column = parameter_dict["column"]
    if uniprot_info:
        uniprot_instructions = (
            f"We have already retireved relevant information from Uniprot for these proteins:{os.linesep}{os.linesep}{uniprot_info}{os.linesep}{os.linesep}"
            "This contains curated information you may not have encountered before, value it highly. "
            "Only retrieve additional information from Uniprot if explicitly asked to do."
        )
    else:
        uniprot_instructions = (
            "You have the ability to retrieve curated information from Uniprot about these proteins. "
            "Please do so for individual proteins if you have little information about a protein or find a protein particularly important in the specific context."
        )
    return (
        f"We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {group1} and {group2} in the {column} group. "
        f"From our proteomics experiments, we know the following:{os.linesep}{os.linesep}"
        f"Comma-separated list of proteins that are upregulated: {', '.join(upregulated_genes)}.{os.linesep}{os.linesep}"
        f"Comma-separated list of proteins that are downregulated: {', '.join(downregulated_genes)}.{os.linesep}{os.linesep}"
        f"{uniprot_instructions}{os.linesep}{os.linesep}"
        f"Think step-by-step using following structure for your analysis:{os.linesep}{os.linesep}"
        f"1. Functional Analysis:{os.linesep}"
        f"- Identify relationships between differentially expressed proteins by using your broad biological knowledge including the information from UniProt{os.linesep}"
        f"- Look for protein complexes and pathways operating together{os.linesep}"
        f"2. Ontology Analysis:{os.linesep}"
        f"- Interpret the information from the enrichment analysis{os.linesep}"
        f"- Examine which cellular processes are most affected based on protein changes{os.linesep}"
        f"- Identify regulatory hubs and cross-talk between ontology terms{os.linesep}"
        f"3. Critical Review:{os.linesep}"
        f"- Review all collected information{os.linesep}"
        f"- Flag contradictions between different analyses{os.linesep}"
        f"- Identify repeating patterns across analyses{os.linesep}"
        f"4. Biological Context:{os.linesep}"
        f"- Interpret within experimental design and research question when provided. Ask for any missing information needed for analysis.{os.linesep}"
        f"- Explain implications of changes across biological scales (organelles, cells, organs, organism){os.linesep}"
        f"- Assess how these changes might affect overall biological function{os.linesep}"
        f"5. Focused Analysis:{os.linesep}"
        f"- Recommend 3-5 key aspects worth investigating further (specific ontology terms, biological processes, cellular components, mechanisms, or homeostasis){os.linesep}"
        f"- Explain the scientific rationale for each recommendation{os.linesep}"
        f"- Ask user which aspect they want to prioritize for deeper analysis{os.linesep}"
        f"6. Deep Dive Analysis: (After user selects a focus){os.linesep}"
        f"- List proteins involved in selected aspect{os.linesep}"
        f"- Gather all available information on key proteins with the tool UniProt{os.linesep}"
        f"- Retrieve quantitative information from the tool intensity plot for all proteins of identified pathways or functional units{os.linesep}"
        f"- Examine protein changes of components{os.linesep}"
        f"- Place findings in the context of current understanding in the field{os.linesep}"
        f"- Identify aspects that confirm or challenge existing models{os.linesep}"
        f"- Develop hypotheses explaining observed patterns{os.linesep}"
        f"- Suggest follow-up experiments to validate your hypotheses{os.linesep}"
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
