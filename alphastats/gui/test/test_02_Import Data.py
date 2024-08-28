from streamlit.testing.v1 import AppTest

def test_loadpage():
    at = AppTest("alphastats/gui/pages/02_Import Data.py", default_timeout=20)
    at.run()