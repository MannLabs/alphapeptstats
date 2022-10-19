import streamlit as st
from alphastats.gui.utils.ui_helper import sidebar_info, img_to_bytes

# centering with streamlit is not really centered
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

header_html = img_center + "<img src='data:image/png;base64,{}'>".format(
    img_to_bytes("alphastats_logo.png")
)

st.markdown(
    header_html, unsafe_allow_html=True,
)


# styl = f"""
#     <style>
#         .css-79elbk {{
#         background-image: url("alphastats_logo.png");
#         min-height: 100%;
#         position: absolute;
#         }}
#         .e1fqkh3o8 {{
#         background-image: url("alphastats_logo.png");
#         min-height: 100%;
#         position: absolute;
#         }}

#     </style>

#     """
# st.markdown(styl , unsafe_allow_html=True)

st.markdown(
    """\n\n
An open-source Python package for the analysis of mass spectrometry based proteomics data 
from the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/).

### How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/alphastats/stargazers) to boost 
our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/alphastats/issues) 
or clone the repository and create a [pull request](https://github.com/MannLabs/alphastats/pulls) with a new branch. For an even more 
interactive participation, check out the [discussions](https://github.com/MannLabs/alphastats/discussions) and the 
[the Contributors License Agreement](misc/CLA.md).
"""
)

sidebar_info(show_logo=False)

# TODO The emojis on the sidebar menu look ugly but currently there are no other options to add icon in streamlit multipage apps
# https://discuss.streamlit.io/t/icons-for-the-multi-app-page-menu-in-the-sidebar-other-than-emojis/27222
# https://icons.getbootstrap.com/
# https://medium.com/codex/create-a-multi-page-app-with-the-new-streamlit-option-menu-component-3e3edaf7e7ad
#  https://lightrun.com/answers/streamlit-streamlit-set-multipage-app-emoji-in-stpage_config-not-filename
