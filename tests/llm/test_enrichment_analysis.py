from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from alphastats.llm.enrichment_analysis import (
    StateKeys,
    _get_background,
    _get_functional_annotation_gprofiler,
    _get_functional_annotation_stringdb,
    _map_short_representation_to_stringdb,
    _shorten_representations,
    _wrap_exceptions_requests_post,
    get_enrichment_data,
)


@patch("alphastats.llm.enrichment_analysis.requests.post")
def test_successful_response(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = [
        {
            "term": "GO:0008150",
            "description": "biological_process",
            "p_value": 0.01,
            "ncbiTaxonId": 9606,
            "inputGenes": ["gene1", "gene2"],
        },
        {
            "term": "GO:0003674",
            "description": "molecular_function",
            "p_value": 0.05,
            "ncbiTaxonId": 9606,
            "inputGenes": ["gene3"],
        },
    ]
    mock_post.return_value = mock_response

    identifiers = ["gene1", "gene2", "gene3"]

    # When
    result = _get_functional_annotation_stringdb(identifiers)

    # Assert that the mock was called with the correct arguments
    mock_post.assert_called_with(
        url="https://string-db.org/api/json/enrichment",
        data={
            "identifiers": "%0d".join(identifiers),
            "species": "9606",
            "caller_identity": "alphapeptstats",
        },
        timeout=600,
    )

    expected_data = [
        {"term": "GO:0008150", "description": "biological_process", "p_value": 0.01},
        {"term": "GO:0003674", "description": "molecular_function", "p_value": 0.05},
    ]
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result, expected_df)


@patch("alphastats.llm.enrichment_analysis.requests.post")
def test_with_background_identifiers(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = [
        {
            "term": "GO:0008150",
            "description": "biological_process",
            "p_value": 0.01,
            "ncbiTaxonId": 9606,
            "inputGenes": ["gene1", "gene2"],
        },
    ]
    mock_post.return_value = mock_response

    identifiers = ["gene1", "gene2"]
    background_identifiers = ["gene3", "gene4"]

    # When
    result = _get_functional_annotation_stringdb(identifiers, background_identifiers)

    # Assert that the mock was called with the correct arguments
    mock_post.assert_called_with(
        url="https://string-db.org/api/json/enrichment",
        data={
            "identifiers": "%0d".join(identifiers),
            "background_string_identifiers": "%0d".join(background_identifiers),
            "species": "9606",
            "caller_identity": "alphapeptstats",
        },
        timeout=600,
    )

    expected_data = [
        {"term": "GO:0008150", "description": "biological_process", "p_value": 0.01},
    ]
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result, expected_df)


@patch("alphastats.llm.enrichment_analysis.requests.post")
def test_map_short_representation_to_string_success(mock_post):
    mock_response = Mock()
    mock_response.text = "input1\tquery1\tSTRING_ID1\ninput2\tquery2\tSTRING_ID2"
    mock_post.return_value = mock_response

    short_representations = ["input1", "input2"]
    result = _map_short_representation_to_stringdb(short_representations)

    # Assert that the mock was called with the correct arguments
    mock_post.assert_called_with(
        url="https://version-12-0.string-db.org/api/tsv-no-header/get_string_ids",
        data={
            "identifiers": "\r".join(short_representations),
            "species": "9606",
            "limit": 1,
            "echo_query": 1,
            "caller_identity": "alphapeptstats",
        },
        timeout=600,
    )

    expected_result = ["STRING_ID1", "STRING_ID2"]
    assert result == expected_result


@patch("alphastats.llm.enrichment_analysis.requests.post")
def test_map_short_representation_to_string_empty_response(mock_post):
    mock_response = Mock()
    mock_response.text = ""
    mock_post.return_value = mock_response

    short_representations = ["input1", "input2"]
    with pytest.raises(
        ValueError, match="No identifiers could be mapped to STRING identifiers."
    ):
        _map_short_representation_to_stringdb(short_representations)


def test_shorten_representations_custom_separator():
    representations = ["gene1:info1|info2", "gene2:info3|info4", "gene3:info5|info6"]
    result = _shorten_representations(representations, sep="|")
    expected = ["info1", "info3", "info5"]
    assert result == expected


def test_shorten_representations_no_separator():
    representations = ["gene1:info1", "gene2:info2", "gene3:info3"]
    result = _shorten_representations(representations)
    expected = ["info1", "info2", "info3"]
    assert result == expected


def test_shorten_representations_empty_list():
    representations = []
    result = _shorten_representations(representations)
    expected = []
    assert result == expected


def test_shorten_representations_no_colon():
    representations = ["gene1;info1", "gene2;info2", "gene3;info3"]
    result = _shorten_representations(representations)
    expected = ["gene1", "gene2", "gene3"]
    assert result == expected


def test_shorten_representations_mixed_format():
    representations = ["gene1:info1;info2", "gene2;info3", "gene3:info4"]
    result = _shorten_representations(representations)
    expected = ["info1", "gene2", "info4"]
    assert result == expected


@patch("alphastats.llm.enrichment_analysis.GProfiler")
def test_get_functional_annotation_gprofiler_success(mock_gprofiler):
    mock_gp_instance = Mock()
    mock_gp_instance.profile.return_value = pd.DataFrame(
        {
            "term": ["GO:0008150", "GO:0003674"],
            "description": ["biological_process", "molecular_function"],
            "p_value": [0.01, 0.05],
        }
    )
    mock_gprofiler.return_value = mock_gp_instance

    query = ["gene1", "gene2"]
    background = ["gene3", "gene4"]
    organism = "hsapiens"

    # When
    result = _get_functional_annotation_gprofiler(query, background, organism)

    # Assert that g:Profiler was called with the correct arguments
    mock_gp_instance.profile.assert_called_with(
        query=query, organism=organism, background=background
    )

    expected_df = pd.DataFrame(
        {
            "term": ["GO:0008150", "GO:0003674"],
            "description": ["biological_process", "molecular_function"],
            "p_value": [0.01, 0.05],
        }
    )

    pd.testing.assert_frame_equal(result, expected_df)


def test_get_functional_annotation_gprofiler_unsupported_organism():
    query = ["gene1", "gene2"]
    background = ["gene3", "gene4"]
    organism = "unsupported_organism"

    with pytest.raises(
        Warning, match=f"Organism {organism} not necessarily supported by g:Profiler"
    ):
        _get_functional_annotation_gprofiler(query, background, organism)


@patch("alphastats.llm.enrichment_analysis.GProfiler")
def test_get_functional_annotation_gprofiler_no_background(mock_gprofiler):
    mock_gp_instance = Mock()
    mock_gp_instance.profile.return_value = pd.DataFrame(
        {
            "term": ["GO:0008150"],
            "description": ["biological_process"],
            "p_value": [0.01],
        }
    )
    mock_gprofiler.return_value = mock_gp_instance

    query = ["gene1", "gene2"]
    organism = "hsapiens"

    # When
    result = _get_functional_annotation_gprofiler(query, organism=organism)

    # Assert that g:Profiler was called with the correct arguments
    mock_gp_instance.profile.assert_called_with(
        query=query, organism=organism, background=None
    )

    expected_df = pd.DataFrame(
        {
            "term": ["GO:0008150"],
            "description": ["biological_process"],
            "p_value": [0.01],
        }
    )

    pd.testing.assert_frame_equal(result, expected_df)


@patch("alphastats.llm.enrichment_analysis.GProfiler")
def test_get_functional_annotation_gprofiler_empty_query(mock_gprofiler):
    mock_gp_instance = Mock()
    mock_gp_instance.profile.return_value = pd.DataFrame()
    mock_gprofiler.return_value = mock_gp_instance

    query = []
    organism = "hsapiens"

    with pytest.raises(
        ValueError, match="No query genes provided for enrichment analysis."
    ):
        _get_functional_annotation_gprofiler(query, organism=organism)

    # Assert that g:Profiler was not called
    mock_gp_instance.profile.assert_not_called()


@patch("alphastats.llm.enrichment_analysis._get_functional_annotation_gprofiler")
@patch("alphastats.llm.enrichment_analysis._shorten_representations")
@patch("streamlit.session_state", {})
def test_get_enrichment_data_gprofiler(mock_shorten, mock_gprofiler):
    mock_shorten.side_effect = lambda x: x
    mock_gprofiler.return_value = pd.DataFrame(
        {
            "term": ["GO:0008150", "GO:0003674"],
            "description": ["biological_process", "molecular_function"],
            "p_value": [0.01, 0.05],
        }
    )

    difexpressed = ["gene1", "gene2"]
    background = ["gene1", "gene2", "gene3", "gene4"]
    organism_id = "9606"
    tool = "gprofiler"

    # When
    result = get_enrichment_data(difexpressed, organism_id, tool, background=background)

    # Assert that the mock functions were called with the correct arguments
    mock_shorten.assert_any_call(difexpressed)
    mock_shorten.assert_any_call(background)
    mock_gprofiler.assert_called_with(
        query=difexpressed,
        background=background,
        organism="hsapiens",
    )

    expected_df = pd.DataFrame(
        {
            "term": ["GO:0008150", "GO:0003674"],
            "description": ["biological_process", "molecular_function"],
            "p_value": [0.01, 0.05],
        }
    )

    pd.testing.assert_frame_equal(result, expected_df)


@patch("alphastats.llm.enrichment_analysis._get_functional_annotation_stringdb")
@patch("alphastats.llm.enrichment_analysis._map_short_representation_to_stringdb")
@patch("alphastats.llm.enrichment_analysis._shorten_representations")
@patch("streamlit.session_state", {})
def test_get_enrichment_data_string(mock_shorten, mock_map, mock_string):
    mock_shorten.side_effect = lambda x: x
    mock_map.side_effect = lambda x, _: x
    mock_string.return_value = pd.DataFrame(
        {
            "term": ["GO:0008150"],
            "description": ["biological_process"],
            "p_value": [0.01],
        }
    )

    difexpressed = ["gene1", "gene2"]
    background = ["gene1", "gene2", "gene3", "gene4"]
    organism_id = "9606"
    tool = "string"

    # When
    result = get_enrichment_data(difexpressed, organism_id, tool, background=background)

    # Assert that the mock functions were called with the correct arguments
    mock_shorten.assert_any_call(difexpressed)
    mock_shorten.assert_any_call(background)
    mock_map.assert_any_call(difexpressed, organism_id)
    mock_map.assert_any_call(background, organism_id)
    mock_string.assert_called_with(
        identifiers=difexpressed,
        background_identifiers=background,
        species_id=organism_id,
    )

    expected_df = pd.DataFrame(
        {
            "term": ["GO:0008150"],
            "description": ["biological_process"],
            "p_value": [0.01],
        }
    )

    pd.testing.assert_frame_equal(result, expected_df)


def test_get_enrichment_data_invalid_tool():
    difexpressed = ["gene1", "gene2"]
    tool = "invalid_tool"

    with pytest.raises(
        ValueError,
        match="Tool invalid_tool not supported. Must be either 'gprofiler' or 'string'.",
    ):
        get_enrichment_data(difexpressed, tool=tool)


@patch("alphastats.llm.enrichment_analysis._get_functional_annotation_gprofiler")
def test_get_enrichment_data_invalid_organism_gprofiler(mock_gprofiler):
    difexpressed = ["gene1", "gene2"]
    organism_id = "99999"  # Invalid organism ID for g:Profiler
    tool = "gprofiler"

    with pytest.raises(
        ValueError, match=f"Organism ID {organism_id} not supported by g:Profiler"
    ):
        get_enrichment_data(difexpressed, organism_id, tool, include_background=False)


@patch("streamlit.session_state", {})
def test_get_background_data_no_background_provided():
    with pytest.raises(
        ValueError,
        match="Background identifiers must be provided as additional argument if enrichment is not run from the GUI.",
    ):
        _get_background([])


@patch("alphastats.llm.enrichment_analysis._shorten_representations")
@patch("streamlit.session_state")
def test_get_background_with_streamlit_dataset(mock_session_state, mock_shorten):
    # Mock the dataset in Streamlit session state
    mock_dataset = Mock()
    mock_dataset.feature_to_repr_map.values.return_value = [
        "gene1:info1",
        "gene2:info2",
        "gene3:info3",
        "gene4:info4",
    ]
    mock_session_state.get.return_value = mock_dataset
    mock_shorten.side_effect = lambda x: [item.split(":")[0] for item in x]

    # When
    result = _get_background([])

    # Assert that the mock functions were called with the correct arguments
    mock_session_state.get.assert_called_with(StateKeys.DATASET, None)
    assert result == ["gene1", "gene2", "gene3", "gene4"]


@patch("alphastats.llm.enrichment_analysis.requests.post")
def test_wrap_exceptions_requests_post_success(mock_post):
    mock_response = Mock()
    mock_post.return_value = mock_response

    api_descriptor = "Test API"
    url = "http://example.com"
    data = {"key": "value"}
    timeout = 10
    result = _wrap_exceptions_requests_post(api_descriptor, url, timeout, data=data)

    # Assert that the mock was called with the correct arguments
    mock_post.assert_called_with(url=url, timeout=timeout, data=data)
    assert result == mock_response


@patch("alphastats.llm.enrichment_analysis.requests.post")
def test_wrap_exceptions_requests_post_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout

    api_descriptor = "Test API"
    url = "http://example.com"
    timeout = 10

    with pytest.raises(
        ValueError, match="Request to Test API timed out after 10 seconds"
    ):
        _wrap_exceptions_requests_post(api_descriptor, url, timeout)


@patch("alphastats.llm.enrichment_analysis.requests.post")
def test_wrap_exceptions_requests_post_request_exception(mock_post):
    mock_post.side_effect = requests.exceptions.RequestException("Connection error")

    api_descriptor = "Test API"
    url = "http://example.com"
    timeout = 10

    with pytest.raises(
        ValueError, match="Request to Test API failed: Connection error"
    ):
        _wrap_exceptions_requests_post(api_descriptor, url, timeout)
