import os

import streamlit as st

from alphastats.gui.utils.llm_helper import llm_config
from alphastats.gui.utils.session_manager import STATE_SAVE_FOLDER_PATH, SessionManager
from alphastats.gui.utils.state_utils import (
    init_session_state,
)
from alphastats.gui.utils.ui_helper import (
    has_llm_support,
    img_to_bytes,
    sidebar_info,
)

st.set_page_config(layout="wide")

init_session_state()
sidebar_info()

img_center = """
<head>
<title> CSS object-position property </title>
<style>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 25%;
</style>
</head>
"""

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
icon = os.path.join(_this_directory, "../alphapeptstats_logo.png")

header_html = img_center + f"<img src='data:image/png;base64,{img_to_bytes(icon)}'>"

st.markdown(
    header_html,
    unsafe_allow_html=True,
)
st.markdown(
    """\n\n
An open-source Python package for the analysis of mass spectrometry based proteomics data
from the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/).
"""
)


if has_llm_support():
    st.markdown("### Configure LLM")
    llm_config()

st.markdown("""### Load previous session""")
saved_sessions = SessionManager.get_saved_sessions(STATE_SAVE_FOLDER_PATH)

if not saved_sessions:
    st.write(
        f"No saved sessions found in {STATE_SAVE_FOLDER_PATH}. Use the 'Save session' button onm the left to save a session."
    )
else:
    c1, _ = st.columns([0.25, 0.75])
    file_to_load = c1.selectbox(
        options=saved_sessions,
        label=f"Select a session to load (from {STATE_SAVE_FOLDER_PATH})",
    )

    if has_llm_support():
        c1.info(
            "Note that all LLM chats will be initialized with the one model configured above."
        )
    if st.button(
        "Load",
        help="Load the selected session. Note that this will overwrite the current session.",
    ):
        session_manager = SessionManager()
        loaded_file_path = session_manager.load(file_to_load, st.session_state)
        st.toast(f"Session state loaded from {loaded_file_path}", icon="✅")
        for warning in session_manager.warnings:
            st.warning(warning)


##
st.markdown(
    """### How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/alphapeptstats/stargazers) to boost
our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/alphapeptstats/issues)
or clone the repository and create a [pull request](https://github.com/MannLabs/alphapeptstats/pulls) with a new branch. For an even more
interactive participation, check out the [discussions](https://github.com/MannLabs/alphapeptstats/discussions) and the
[the Contributors License Agreement](misc/CLA.md).
"""
)

# TODO The emojis on the sidebar menu look ugly but currently there are no other options to add icon in streamlit multipage apps
# https://discuss.streamlit.io/t/icons-for-the-multi-app-page-menu-in-the-sidebar-other-than-emojis/27222
# https://icons.getbootstrap.com/
# https://medium.com/codex/create-a-multi-page-app-with-the-new-streamlit-option-menu-component-3e3edaf7e7ad
# https://lightrun.com/answers/streamlit-streamlit-set-multipage-app-emoji-in-stpage_config-not-filename
