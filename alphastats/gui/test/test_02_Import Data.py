from streamlit.testing.v1 import AppTest

def test_loadpage():
    at = AppTest("alphastats/gui/pages/02_Import Data.py", default_timeout=20)
    at.run()

    assert at.session_state.organism == 9606
    assert at.session_state.user_session_id == 'test session id'
    assert at.session_state.software == '<select>'
    assert at.session_state.gene_to_prot_id == {}
    assert at.session_state.loader == None

def test_sampledata():
    at = AppTest("alphastats/gui/pages/02_Import Data.py", default_timeout=20)
    at.run()

    at.button[0].click().run()

    assert at.session_state.metadata_columns == ['sample', 'disease', 'Drug therapy (procedure) (416608005)', 'Lipid-lowering therapy (134350008)']
    assert str(type(at.session_state.dataset)) == "<class 'alphastats.DataSet.DataSet'>"
    assert at.session_state.software == "<select>"
    assert str(type(at.session_state.distribution_plot)) == "<class 'plotly.graph_objs._figure.Figure'>"
    assert str(type(at.session_state.loader)) == "<class 'alphastats.loader.MaxQuantLoader.MaxQuantLoader'>"
    assert "plotting_options" in at.session_state
    assert "statistic_options" in at.session_state