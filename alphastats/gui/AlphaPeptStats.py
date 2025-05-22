import streamlit as st

from alphastats.gui.utils.ui_helper import has_llm_support

# TODO check if numbers and white spaces can be dropped
pages = [
    "pages_/01_Home.py",
    "pages_/02_Import Data.py",
    "pages_/03_Data Overview.py",
    "pages_/04_Preprocessing.py",
    "pages_/05_Analysis.py",
]

if has_llm_support():
    pages.append("pages_/06_LLM.py")

pages.append("pages_/07_Results.py")

pages = [st.Page(page) for page in pages]

pg = st.navigation(pages)

pg.run()
