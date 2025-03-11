from streamlit.testing.v1 import AppTest

from alphastats.gui.utils.state_keys import StateKeys

from .conftest import APP_FOLDER, create_dataset_alphapept

TESTED_PAGE = f"{APP_FOLDER}/pages/03_Data Overview.py"


def test_page_03_loads_without_input():
    """Test if the page loads without any input and inititalizes the session state with the correct values."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    assert not at.exception


def test_page_03_loads_with_input():
    """Test if the page loads with input and inititalizes the session state with the correct values."""
    at = AppTest(TESTED_PAGE, default_timeout=200)
    at.run()

    at.session_state[StateKeys.DATASET] = create_dataset_alphapept()
    at.run()

    assert not at.exception
