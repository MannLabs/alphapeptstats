"""Test that the function definitions in the LLM match the actual functions:

- all parameters defined for the LLM are present in the function definitions
- all non-default parameters in the function definitions are required for the LLM
"""

import inspect
from typing import Callable, Dict
from unittest.mock import MagicMock, patch

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
from alphastats.loader.generic_loader import GenericLoader


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

    # are all required parameters defined
    assert set(parameters_dict_["required"]).issubset(set(parameters_))


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


def SyntheticDataSet():
    # Create a mock with the same interface as DataSet
    loader = GenericLoader(
        file="testfiles/synthetic/preprocessing_pentests.csv",
        intensity_column="Intensity [sample]",
        index_column="Protein IDs",
        gene_names_column="Gene names",
    )
    metadata_path = "testfiles/synthetic/preprocessing_pentests_metadata.csv"
    obj = DataSet(
        loader=loader,
        metadata_path_or_df=metadata_path,
        sample_column="sample",
    )
    obj.id_holder = MagicMock()
    return obj


testdata_test_get_uniprot_info_for_search_string = [
    ({}, "gene1", ["id1;id2"], "gene1: "),
    ({"id2": {}}, "gene2", ["id2"], "gene2: "),
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

    dataset = SyntheticDataSet()
    dataset.id_holder.get_feature_ids_from_search_string.return_value = call_arg
    mock_session_state[StateKeys.DATASET] = dataset

    mock_session_state[StateKeys.ANNOTATION_STORE] = store_init.copy()
    mock_session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = []
    mock_get_annotation_from_uniprot.return_value = ({}, call_arg[0])
    mock_format_uniprot_annotation.return_value = ""
    mock_get_annotation_from_store.side_effect = (
        get_annotation_from_store_by_feature_list
    )

    # when
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
    dataset.id_holder.get_feature_ids_from_search_string.assert_called_with(llm_input)
