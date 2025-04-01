"""This module contains functions to generate prompts for the LLM model."""

from __future__ import annotations

import os
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall

from alphastats.dataset.dataset import DataSet
from alphastats.dataset.keys import ConstantsClass
from alphastats.llm.llm_utils import get_subgroups_for_each_group


def get_system_message(dataset: DataSet) -> str:
    """Get the system message for the LLM model."""
    subgroups = get_subgroups_for_each_group(dataset.metadata)

    return (
        f"You are an expert biologist and have extensive experience in molecular biology, medicine and biochemistry.{os.linesep}"
        "A user will present you with data regarding proteins upregulated. They seek your "
        "expertise in understanding the connections between these proteins and their potential role "
        f"in disease genesis. {os.linesep}"
        f"Provide a detailed and insightful, yet concise response based on the given information. Use formatting to make your response more human readable."
        f"The data you have has following groups and respective subgroups: {str(subgroups)}."
        "Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that. If the data coming to"
        " you from a function has references to the literature (for example, PubMed), always quote the references in your response."
        " The next message will be referred to as the initial prompt by tools and potentially by the user."
    )


def _get_experimental_design_prompt(
    parameter_dict: dict[str, Any],
) -> str:
    group1 = parameter_dict["group1"]
    group2 = parameter_dict["group2"]
    column = parameter_dict["column"]
    return (
        f"We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {group1} and {group2} in the {column} group. "
    )


def _get_protein_data_prompt(
    upregulated_genes: list[str],
    downregulated_genes: list[str],
    uniprot_info: str,
):
    """Get the initial prompt for the LLM model."""
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
        f"From our proteomics experiments, we know that the following ones are upregulated: {', '.join(upregulated_genes)}.{os.linesep}{os.linesep}"
        f"Here is a comma-separated list of proteins that are downregulated: {', '.join(downregulated_genes)}.{os.linesep}{os.linesep}"
        f"{uniprot_instructions}"
    )


class LLMInstructionKeys(metaclass=ConstantsClass):
    """Keys for the LLM instructions."""

    SIMPLE = "simple"
    CUSTOM = "customize prompt"


LLMInstructions = {
    LLMInstructionKeys.SIMPLE: (
        "Help us understand the potential connections between these proteins and how they might be contributing "
        "to the differences. After that provide a high level summary."
    ),
    LLMInstructionKeys.CUSTOM: "Please give instructions to the LLM model on how to generate a response.",
}


def _get_initial_instruction(preset: None | str = "simple"):
    if preset == LLMInstructionKeys.SIMPLE:
        return LLMInstructions[LLMInstructionKeys.SIMPLE]
    else:
        return LLMInstructions[LLMInstructionKeys.CUSTOM]


def get_initial_prompt(
    experimental_design_prompt: str,
    protein_data_prompt: str,
    initial_instruction: str,
) -> str:
    """Get the initial prompt for the LLM model."""
    return f"{os.linesep}{os.linesep}".join(
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
