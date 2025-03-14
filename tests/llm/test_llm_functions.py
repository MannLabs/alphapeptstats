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
from alphastats.gui.utils.state_keys import StateKeys
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


testdata_test_get_annotation_from_uniprot_by_feature_list = [
    (["id1"], "id1", "id1annotation"),  # just one feature
    (["id2", "id2;id2-3"], "id2", "id2annotation"),  # same base ids
    (["id2", "id3"], "id2", "id2annotation"),  # different base ids, no longest element
    (
        ["id2", "id2;id3"],
        "id2;id3",
        "id2:id3annotation",
    ),  # different base ids, longest element
]


@pytest.mark.parametrize(
    "features, call_arg, return_value",
    testdata_test_get_annotation_from_uniprot_by_feature_list,
)
@patch("alphastats.llm.llm_functions.get_annotations_for_feature")
def test_get_annotation_from_uniprot_by_feature_list(
    mock_uniprot_annotation, features, call_arg, return_value
):
    """Test that the function retrieves the correct entry from UniProt."""

    # just one feature
    mock_uniprot_annotation.return_value = return_value
    result = get_annotation_from_uniprot_by_feature_list(features)
    mock_uniprot_annotation.assert_called_with(call_arg)
    assert result == (return_value, call_arg)


@patch("alphastats.llm.llm_functions.get_annotations_for_feature")
def test_get_annotation_from_uniprot_by_feature_list_error(mock_uniprot_annotation):
    with pytest.raises(ValueError, match="No features provided"):
        get_annotation_from_uniprot_by_feature_list([])
    mock_uniprot_annotation.assert_not_called()


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


testdata_test_get_uniprot_info_for_search_string = [
    ({}, "gene2;gene4", ["id1;id4"], "gene2;gene4: "),  # repr
    ({"id2": {}}, "gene2", ["id2"], "gene2: "),  # repr over gene key
    ({}, "id1;id4", ["id1;id4"], "id1;id4: "),  # feature id
    ({}, "gene4", ["id1;id4"], "gene4: "),  # by gene
    ({}, "id1", ["id1;id4", "id5;id1"], "id1: "),  # by protein
]


@pytest.mark.parametrize(
    "store_init, llm_input, call_arg, expected_result",
    testdata_test_get_uniprot_info_for_search_string,
)
@patch("alphastats.llm.llm_functions.format_uniprot_annotation")
@patch("alphastats.llm.llm_functions.get_annotation_from_uniprot_by_feature_list")
@patch("alphastats.llm.llm_functions.get_annotation_from_store_by_feature_list")
@patch("streamlit.session_state", new_callable=dict)
def test_get_uniprot_info_for_search_string(
    mock_session_state,
    mock_get_annotation_from_store,
    mock_get_annotation_from_uniprot,
    mock_format_uniprot_annotation,
    store_init,
    llm_input,
    call_arg,
    expected_result,
):
    """Test that the function retrieves the correct UniProt information for the LLM input."""

    mock_session_state[StateKeys.DATASET] = DummyDataset()
    mock_session_state[StateKeys.ANNOTATION_STORE] = store_init.copy()
    mock_session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = []
    mock_get_annotation_from_uniprot.return_value = ({}, call_arg[0])
    mock_format_uniprot_annotation.return_value = ""
    mock_get_annotation_from_store.side_effect = (
        get_annotation_from_store_by_feature_list
    )

    result = get_uniprot_info_for_search_string(llm_input)
    assert mock_get_annotation_from_store.call_args[0][0] == call_arg

    # if the feature is not in the store, the function should call get_annotation_from_uniprot_by_feature_list
    if len(store_init) == 0:
        mock_get_annotation_from_uniprot.assert_called_with(call_arg)
    else:
        mock_get_annotation_from_uniprot.assert_not_called()

    # it should always be in the store afterwards
    assert call_arg[0] in mock_session_state[StateKeys.ANNOTATION_STORE]
    assert result == expected_result


@patch("streamlit.session_state", new_callable=dict)
def test_get_uniprot_info_for_search_string_error(mock_session_state):
    # not valid
    mock_session_state[StateKeys.DATASET] = DummyDataset()
    mock_session_state[StateKeys.ANNOTATION_STORE] = {}
    mock_session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = []
    llm_input = "id6"
    with pytest.raises(ValueError, match="id6 not found in dataset."):
        get_uniprot_info_for_search_string(llm_input)
