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
        f"You are an expert biologist and have extensive experience in molecular biology, medicine and biochemistry.{os.linesep}"
        "A user will present you with data regarding proteins upregulated in certain cells "
        "sourced from UniProt and abstracts from scientific publications. They seek your "
        "expertise in understanding the connections between these proteins and their potential role "
        f"in disease genesis. {os.linesep}"
        f"Provide a detailed and insightful, yet concise response based on the given information. Use formatting to make your response more human readable."
        f"The data you have has following groups and respective subgroups: {str(subgroups)}."
        "Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that. If the data coming to"
        " you from a function has references to the literature (for example, PubMed), always quote the references in your response."
    )


def get_initial_prompt(
    parameter_dict: Dict[str, Any],
    upregulated_genes: List[str],
    downregulated_genes: List[str],
):
    """Get the initial prompt for the LLM model."""
    group1 = parameter_dict["group1"]
    group2 = parameter_dict["group2"]
    column = parameter_dict["column"]
    return (
        f"We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {group1} and {group2} in the {column} group. "
        f"From our proteomics experiments, we know that the following ones are upregulated: {', '.join(upregulated_genes)}.{os.linesep}{os.linesep}"
        f"Here is the list of proteins that are downregulated: {', '.join(downregulated_genes)}.{os.linesep}{os.linesep}"
        f"Help us understand the potential connections between these proteins and how they might be contributing "
        f"to the differences. After that provide a high level summary"
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
