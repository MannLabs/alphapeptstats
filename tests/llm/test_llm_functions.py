"""Test that the function definitions in the LLM match the actual functions:

- all parameters defined for the LLM are present in the function definitions
- all non-default parameters in the function definitions are required for the LLM
"""

import inspect
from typing import Callable, Dict

import pandas as pd

from alphastats.dataset.dataset import DataSet
from alphastats.llm.llm_functions import (
    GENERAL_FUNCTION_MAPPING,
    get_assistant_functions,
    get_general_assistant_functions,
)


def _get_method_parameters(method: Callable) -> Dict:
    """Get the parameters of a method as a dictionary, excluding "self"."""
    signature = inspect.signature(method)

    params = dict(signature.parameters)
    if "self" in params:
        del params["self"]

    return params


def _get_class_methods(cls):
    """Get all methods of a class as a dictionary, excluding "self"."""
    return {
        member[0]: member[1]
        for member in inspect.getmembers(cls, predicate=inspect.isfunction)
        if member[0] != "self"
    }


def assert_parameters(method_definition: Callable, llm_function_dict_):
    """Assert that the parameters of a method match the parameters defined in the dict used by the LLM."""

    # suffix '_' denotes LLM-related variables
    parameters = _get_method_parameters(method_definition)
    parameters_without_default = [
        param
        for param in parameters
        if parameters[param].default == inspect.Parameter.empty
    ]

    parameters_dict_ = llm_function_dict_["function"]["parameters"]
    parameters_ = parameters_dict_["properties"].keys()

    # are all in parameters_ available in the function?
    assert set(parameters_).issubset(set(parameters))

    # are all the parameters w/o default in the function filled in parameters_?
    assert set(parameters_without_default).issubset(set(parameters_))

    # are all required parameters marked as 'required'?
    assert set(parameters_without_default).issubset(set(parameters_dict_["required"]))


def test_general_assistant_functions():
    """Test that the general assistant functions in the LLM match the actual functions."""
    # suffix '_' denotes LLM-related variables
    assistant_functions_dict = get_general_assistant_functions()

    for llm_function_dict_ in assistant_functions_dict:
        name_ = llm_function_dict_["function"]["name"]

        method_definition = GENERAL_FUNCTION_MAPPING.get(name_, None)

        if method_definition is None:
            raise ValueError(f"Function not found in test: {name_}")

        assert_parameters(method_definition, llm_function_dict_)


def test_assistant_functions():
    """Test that the assistant functions in the LLM match the actual functions."""
    # suffix '_' denotes LLM-related variables
    assistant_functions_dict = get_assistant_functions({}, pd.DataFrame(), {})

    all_dataset_methods = _get_class_methods(DataSet)

    for llm_function_dict_ in assistant_functions_dict:
        name_ = llm_function_dict_["function"]["name"]

        method_definition = all_dataset_methods.get(name_, None)

        if method_definition is None:
            raise ValueError(f"Function not found in test: {name_}")

        assert_parameters(method_definition, llm_function_dict_)
