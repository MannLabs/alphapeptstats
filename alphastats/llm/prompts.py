"""This module contains functions to generate prompts for the LLM model."""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
from openai.types.chat import ChatCompletionMessageToolCall

from alphastats.dataset.dataset import DataSet
from alphastats.dataset.keys import ConstantsClass
from alphastats.llm.llm_utils import get_subgroups_for_each_group

newline = os.linesep


def get_system_message(dataset: DataSet) -> str:
    """Get the system message for the LLM model."""
    subgroups = get_subgroups_for_each_group(dataset.metadata)

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
        "Explain your thought process clearly as you move from observations to interpretations. "
        "The next message will be referred to as the initial prompt by tools and potentially by us."
    )


def _get_experimental_design_prompt(
    parameter_dict: dict[str, Any],
) -> str:
    group1 = parameter_dict["group1"]
    group2 = parameter_dict["group2"]
    column = parameter_dict["column"]
    return (
        "We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {group1} and {group2} in the {column} grouping."
    )


def _get_protein_data_prompt(
    upregulated_features: list[str],
    downregulated_features: list[str],
    uniprot_info: str,
    feature_to_repr_map: dict,
    parameter_dict: dict,
    enrichment_data: Optional | pd.DataFrame = None,
) -> str:
    """Get the initial prompt for the LLM model."""
    group1 = parameter_dict["group1"]
    group2 = parameter_dict["group2"]
    if uniprot_info:
        uniprot_instructions = (
            f"We have already retrieved relevant information from Uniprot for these proteins:{newline}{newline}{uniprot_info}{newline}{newline}"
            "This contains curated information you may not have encountered before, value it highly. "
            "Only retrieve additional information from Uniprot if explicitly asked to do."
        )
    else:
        uniprot_instructions = (
            "You have the ability to retrieve curated information from Uniprot about these proteins. "
            "Please do so for individual proteins if you have little information about a protein or find a protein particularly important in the specific context."
        )
    if enrichment_data is not None:
        enrichment_prompt = (
            f"{newline}{newline}We have also performed an enrichment analysis of all regulated proteins to identify overrepresented ontology terms. These are the results in markdown tabular format:{newline}{newline}"
            f"{enrichment_data.to_markdown(index=False)}"
        )
    else:
        enrichment_prompt = ""

    upregulated_genes = list(
        map(
            feature_to_repr_map.get,
            upregulated_features,
        )
    )

    downregulated_genes = list(
        map(
            feature_to_repr_map.get,
            downregulated_features,
        )
    )
    return (
        f"From our proteomics experiments, we know the following:{newline}{newline}"
        f"Comma-separated list of proteins that are upregulated (high in '{group1}'): {', '.join(upregulated_genes)}.{newline}{newline}"
        f"Comma-separated list of proteins that are downregulated (high in '{group2}'): {', '.join(downregulated_genes)}.{newline}{newline}"
        f"{uniprot_instructions}{enrichment_prompt}"
    )


class LLMInstructionKeys(metaclass=ConstantsClass):
    """Keys for the LLM instructions."""

    SIMPLE = "simple"
    CUSTOM = "customize prompt"
    CHAIN_OF_THOUGHT = "chain of thought"


LLMInstructions = {
    LLMInstructionKeys.SIMPLE: (
        "Help us understand the potential connections between these proteins and how they might be contributing "
        "to the differences. After that provide a high level summary."
    ),
    LLMInstructionKeys.CUSTOM: "<<<Please give instructions to the LLM model on how to generate a response.>>>",
    LLMInstructionKeys.CHAIN_OF_THOUGHT: (
        f"Think step-by-step using following structure for your analysis:{newline}{newline}"
        f"1. Functional Analysis:{newline}"
        f"- Identify relationships between differentially expressed proteins by using your broad biological knowledge including the information from UniProt{newline}"
        f"- Look for protein complexes and pathways operating together{newline}"
        f"2. Ontology Analysis:{newline}"
        f"- Interpret the information from the enrichment analysis (run an analysis with gprofiler and background included in case no result is provided){newline}"
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
    ),
}


def _get_initial_instruction(preset: str | None = "simple") -> str:
    if preset in LLMInstructions:
        return LLMInstructions[preset]
    else:
        return LLMInstructions[LLMInstructionKeys.CUSTOM]


def get_initial_prompt(
    experimental_design_prompt: str,
    protein_data_prompt: str,
    initial_instruction: str,
) -> str:
    """Get the initial prompt for the LLM model."""
    return f"{newline}{newline}".join(
        [experimental_design_prompt, protein_data_prompt, initial_instruction]
    )


def get_tool_call_message(tool_calls: list[ChatCompletionMessageToolCall]) -> str:
    """Get a string representation of the tool calls made by the LLM model."""
    return "\n".join(
        [
            f"Calling function: {tool_call.function.name} "
            f"with arguments: {tool_call.function.arguments}"
            for tool_call in tool_calls
        ]
    )


ITERABLE_ARTIFACT_REPRESENTATION_PROMPT = "Function {} with arguments {} returned a {}, containing {} elements, some of which are non-trivial to represent as text."
SINGLE_ARTIFACT_REPRESENTATION_PROMPT = "Function {} with arguments {} returned a {}."
IMAGE_REPRESENTATION_PROMPT = (
    "This is a visualization result that will be provided as an image."
)
NO_REPRESENTATION_PROMPT = (
    " There is currently no text representation for this artifact that can be interpreted meaningfully. "
    "If the user asks for guidance how to interpret the artifact please rely on the description of the tool function and the arguments it was called with."
)

IMAGE_ANALYSIS_PROMPT = (
    "The previous tool call generated the following image. "
    "Please analyze it in the context of our current discussion and your previous actions."
)
