import streamlit as st

try:
    from alphastats.gui.utils.preprocessing_helper import main_preprocessing
    from alphastats.gui.utils.ui_helper import sidebar_info

except ModuleNotFoundError:
    from utils.preprocessing_helper import main_preprocessing
    from utils.ui_helper import sidebar_info


sidebar_info()

st.markdown("### Preprocessing")

main_preprocessing()