from unittest.mock import MagicMock, patch

from alphastats.dataset.keys import Cols
from alphastats.gui.utils.analysis_helper import (
    gather_uniprot_data,
    get_regulated_features,
)
from alphastats.gui.utils.ui_helper import StateKeys
from alphastats.plots.plot_utils import PlotlyObject


def test_get_regulated_features_some():
    mock_plotly_object = MagicMock(spec=PlotlyObject)
    mock_plotly_object.res = {
        Cols.INDEX: ["feature1", "feature2", "feature3"],
        "label": ["label1", "", "label2"],
    }

    regulated_features = get_regulated_features(mock_plotly_object)
    assert regulated_features == ["feature1", "feature3"]


def test_get_regulated_features_none():
    mock_plotly_object = MagicMock(spec=PlotlyObject)
    mock_plotly_object.res = {
        Cols.INDEX: ["feature1", "feature2", "feature3"],
        "label": ["", "", ""],
    }

    regulated_features = get_regulated_features(mock_plotly_object)
    assert regulated_features == []


def test_get_regulated_features_all():
    mock_plotly_object = MagicMock(spec=PlotlyObject)
    mock_plotly_object.res = {
        Cols.INDEX: ["feature1", "feature2", "feature3"],
        "label": ["label1", "label2", "label3"],
    }

    regulated_features = get_regulated_features(mock_plotly_object)
    assert regulated_features == ["feature1", "feature2", "feature3"]


@patch("streamlit.session_state", new_callable=dict)
@patch("alphastats.gui.utils.analysis_helper.get_annotations_for_feature")
def test_gather_uniprot_data_empty(mock_get_info, mock_session_state):
    mock_session_state[StateKeys.ANNOTATION_STORE] = {}
    features = []

    gather_uniprot_data(features)
    mock_get_info.assert_not_called()


@patch("streamlit.session_state", new_callable=dict)
@patch("alphastats.gui.utils.analysis_helper.get_annotations_for_feature")
def test_gather_uniprot_data_lookupall(mock_get_info, mock_session_state):
    mock_session_state[StateKeys.ANNOTATION_STORE] = {}
    features = ["feature1", "feature2"]

    mock_get_info.side_effect = ["info1", "info2"]

    gather_uniprot_data(features)
    assert mock_session_state[StateKeys.ANNOTATION_STORE] == {
        "feature1": "info1",
        "feature2": "info2",
    }
    mock_get_info.assert_any_call("feature1")
    mock_get_info.assert_any_call("feature2")


@patch("streamlit.session_state", new_callable=dict)
@patch("alphastats.gui.utils.analysis_helper.get_annotations_for_feature")
def test_gather_uniprot_data_lookupsome(mock_get_info, mock_session_state):
    mock_session_state[StateKeys.ANNOTATION_STORE] = {"feature1": "existing_info"}
    features = ["feature1", "feature2"]

    mock_get_info.side_effect = ["info2"]

    gather_uniprot_data(features)
    assert mock_session_state[StateKeys.ANNOTATION_STORE] == {
        "feature1": "existing_info",
        "feature2": "info2",
    }
    mock_get_info.assert_called_once_with("feature2")
