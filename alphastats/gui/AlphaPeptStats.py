import streamlit as st
st.set_page_config(layout="wide")

try:
    from alphastats.gui.utils.ui_helper import sidebar_info, img_to_bytes
except ModuleNotFoundError:
    from utils.ui_helper import sidebar_info, img_to_bytes
import os
#from PIL import Image

# centering with streamlit is not really centered

from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

runtime = get_instance()
session_id = get_script_run_ctx().session_id
session_info = runtime._session_mgr.get_session_info(session_id)

user_session_id = session_id
st.session_state["user_session_id"] = user_session_id

img_center = """
<head>  
<title> CSS object-position property </title>  
<style> 
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 65%;
</style>  
</head>  
"""

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
icon = os.path.join(_this_directory, "alphapeptstats_logo.png")

header_html = img_center + "<img src='data:image/png;base64,{}'>".format(
    img_to_bytes(icon)
)

st.markdown(
    header_html,
    unsafe_allow_html=True,
)

##
st.markdown(
    """\n\n
An open-source Python package for the analysis of mass spectrometry based proteomics data 
from the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/).

### How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/alphapeptstats/stargazers) to boost 
our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/alphapeptstats/issues) 
or clone the repository and create a [pull request](https://github.com/MannLabs/alphapeptstats/pulls) with a new branch. For an even more 
interactive participation, check out the [discussions](https://github.com/MannLabs/alphapeptstats/discussions) and the 
[the Contributors License Agreement](misc/CLA.md).
"""
)

sidebar_info(show_logo=False)

# TODO The emojis on the sidebar menu look ugly but currently there are no other options to add icon in streamlit multipage apps
# https://discuss.streamlit.io/t/icons-for-the-multi-app-page-menu-in-the-sidebar-other-than-emojis/27222
# https://icons.getbootstrap.com/
# https://medium.com/codex/create-a-multi-page-app-with-the-new-streamlit-option-menu-component-3e3edaf7e7ad
#  https://lightrun.com/answers/streamlit-streamlit-set-multipage-app-emoji-in-stpage_config-not-filename
