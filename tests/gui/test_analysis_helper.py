from unittest.mock import MagicMock

from alphastats.dataset.keys import Cols
from alphastats.gui.utils.analysis_helper import get_regulated_features
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
