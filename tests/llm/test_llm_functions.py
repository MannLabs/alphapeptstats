"""Test that the function definitions in the LLM match the actual functions:

- all parameters defined for the LLM are present in the function definitions
- all non-default parameters in the function definitions are required for the LLM
"""

import inspect
from typing import Callable, Dict
from unittest.mock import patch

import pandas as pd
import pytest

from alphastats.dataset.dataset import DataSet
from alphastats.gui.utils.ui_helper import StateKeys
from alphastats.llm.llm_functions import (
    GENERAL_FUNCTION_MAPPING,
    get_annotation_from_store_by_feature_list,
    get_annotation_from_uniprot_by_feature_list,
    get_assistant_functions,
    get_general_assistant_functions,
    get_uniprot_info_for_search_string,
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


def test_get_annotation_from_store_by_feature_list():
    """Test that the function retrieves the correct entry from annotation store."""
    annotation_store = {
        "feature1": {"annotation": "annotation1"},
        "feature3": {"annotation": "annotation3"},
        "feature2": {"annotation": "annotation2"},
    }

    features = ["feature1"]
    result = get_annotation_from_store_by_feature_list(features, annotation_store)
    assert result == {"annotation": "annotation1"}

    features = ["feature2", "feature3"]
    result = get_annotation_from_store_by_feature_list(features, annotation_store)
    assert result == {"annotation": "annotation2"}

    features = []
    result = get_annotation_from_store_by_feature_list(features, annotation_store)
    assert result is None

    features = ["feature4"]
    result = get_annotation_from_store_by_feature_list(features, annotation_store)
    assert result is None


@patch("alphastats.llm.llm_functions.get_annotations_for_feature")
def test_get_annotation_from_uniprot_by_feature_list(mock_uniprot_annotation):
    """Test that the function retrieves the correct entry from UniProt."""

    # just one feature
    features = ["id1"]
    mock_uniprot_annotation.return_value = "id1annotation"
    result = get_annotation_from_uniprot_by_feature_list(features)
    mock_uniprot_annotation.assert_called_with("id1")
    assert result == ("id1annotation", "id1")

    # same base ids
    features = ["id2", "id2;id2-3"]
    get_annotation_from_uniprot_by_feature_list(features)
    mock_uniprot_annotation.assert_called_with("id2")

    # different base ids, no longest element
    features = ["id2", "id3"]
    get_annotation_from_uniprot_by_feature_list(features)
    mock_uniprot_annotation.assert_called_with("id2")

    # different base ids, longest element
    features = ["id2", "id2;id3"]
    get_annotation_from_uniprot_by_feature_list(features)
    mock_uniprot_annotation.assert_called_with("id2;id3")

    features = []
    with pytest.raises(ValueError, match="No features provided"):
        get_annotation_from_uniprot_by_feature_list(features)


@patch("alphastats.llm.llm_functions.format_uniprot_annotation")
@patch("alphastats.llm.llm_functions.get_annotation_from_uniprot_by_feature_list")
@patch("alphastats.llm.llm_functions.get_annotation_from_store_by_feature_list")
@patch("streamlit.session_state", new_callable=dict)
def test_get_uniprot_info_for_llm_input(
    mock_session_state,
    mock_get_annotation_from_store,
    mock_get_annotation_from_uniprot,
    mock_format_uniprot_annotation,
):
    """Test that the function retrieves the correct UniProt information for the LLM input."""

    class DummyDataset:
        _feature_to_repr_map = {
            "id1;id4": "gene2;gene4",
            "id2": "gene2",
            "id3": "gene1",
            "id5;id1": "ids:id5",
        }
        _gene_to_features_map = {
            "gene1": ["id3"],
            "gene2": ["id1;id4", "id2"],
            "gene4": ["id1;id4"],
        }
        _protein_to_features_map = {
            "id1": ["id1;id4", "id5;id1"],
            "id2": ["id2"],
            "id3": ["id3"],
            "id4": ["id1;id4"],
            "id5": ["id5;id1"],
        }

    mock_session_state[StateKeys.DATASET] = DummyDataset()
    mock_session_state[StateKeys.ANNOTATION_STORE] = {}
    mock_session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = []

    # mock_session_state.__getitem__().__setitem__.assert_called_with("id1", "id1")

    # repr
    llm_input = "gene2;gene4"
    mock_get_annotation_from_store.return_value = None
    mock_get_annotation_from_uniprot.return_value = (
        {"annotation": "annotation1"},
        "id1;id4",
    )
    mock_format_uniprot_annotation.return_value = ""
    result = get_uniprot_info_for_search_string(llm_input)
    mock_get_annotation_from_store.assert_called_with(["id1;id4"], {})
    mock_get_annotation_from_uniprot.assert_called_with(["id1;id4"])
    mock_session_state.__getitem__().__setitem__.assert_called_with(
        "id1;id4", {"annotation": "annotation1"}
    )
    assert result == "gene2;gene4: "

    mock_get_annotation_from_store.return_value = ({"annotation": "annotation1"}, "id2")
    # repr that is also gene
    llm_input = "gene2"
    get_uniprot_info_for_search_string(llm_input)
    mock_get_annotation_from_store.assert_called_with(["id2"], {})
    mock_get_annotation_from_uniprot.assert_called_once()

    # feature id
    llm_input = "id1;id4"
    get_uniprot_info_for_search_string(llm_input)
    mock_get_annotation_from_store.assert_called_with(["id1;id4"], {})

    # gene
    llm_input = "gene4"
    get_uniprot_info_for_search_string(llm_input)
    mock_get_annotation_from_store.assert_called_with(["id1;id4"], {})

    # protein
    llm_input = "id1"
    get_uniprot_info_for_search_string(llm_input)
    mock_get_annotation_from_store.assert_called_with(["id1;id4", "id5;id1"], {})

    # not valid
    llm_input = "id6"
    with pytest.raises(ValueError, match="id6 not found in dataset."):
        get_uniprot_info_for_search_string(llm_input)
