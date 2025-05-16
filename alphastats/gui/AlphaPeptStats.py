import streamlit as st

from alphastats.gui.utils.ui_helper import has_llm_support

pages = [
    "pages/01_Home.py",
    "pages/02_Import Data.py",
    "pages/03_Data Overview.py",
    "pages/04_Preprocessing.py",
    "pages/05_Analysis.py",
]

if has_llm_support():
    pages.append("pages/06_LLM.py")

pages.append("pages/07_Results.py")

pg = st.navigation(pages)

pg.run()
