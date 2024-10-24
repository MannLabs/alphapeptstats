import pandas as pd

from alphastats.llm.llm_utils import (
    get_subgroups_for_each_group,
)


def test_get_subgroups_for_each_group_basic():
    """Test basic functionality with simple metadata."""
    # Create test metadata
    data = {
        "disease": ["cancer", "healthy", "cancer"],
        "treatment": ["drug_a", "placebo", "drug_b"],
    }
    metadata = pd.DataFrame(data)

    result = get_subgroups_for_each_group(metadata)

    expected = {
        "disease": ["cancer", "healthy"],
        "treatment": ["drug_a", "placebo", "drug_b"],
    }
    assert result == expected


def test_get_subgroups_for_each_group_empty():
    """Test with empty DataFrame."""
    metadata = pd.DataFrame()
    result = get_subgroups_for_each_group(metadata)
    assert result == {}


def test_get_subgroups_for_each_group_single_column():
    """Test with single column DataFrame."""
    data = {"condition": ["A", "B", "A"]}
    metadata = pd.DataFrame(data)

    result = get_subgroups_for_each_group(metadata)

    expected = {"condition": ["A", "B"]}
    assert result == expected


def test_get_subgroups_for_each_group_numeric_values():
    """Test with numeric values in DataFrame."""
    data = {"age_group": [20, 30, 20], "score": [1.5, 2.5, 1.5]}
    metadata = pd.DataFrame(data)

    result = get_subgroups_for_each_group(metadata)

    expected = {"age_group": ["20", "30"], "score": ["1.5", "2.5"]}
    assert result == expected
