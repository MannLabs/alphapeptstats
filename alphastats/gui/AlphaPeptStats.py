import streamlit as st

pg = st.navigation(
    [
        "pages/01_Home.py",
        "pages/02_Import Data.py",
        "pages/03_Data Overview.py",
        "pages/04_Preprocessing.py",
        "pages/05_Analysis.py",
        "pages/06_LLM.py",
        "pages/07_Results.py",
    ]
)
pg.run()
