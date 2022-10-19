import streamlit as st
import pandas as pd
import base64


# TODO add logo above the options when issue is closed
# https://github.com/streamlit/streamlit/issues/4984


def sidebar_logo():
    img_center = """
    <head>  
    <title> CSS object-position property </title>  
    <style> 
    img {
       position: absolute;
       right: -7px;
       bottom: -710px;
       width: 45%;
    </style>  
    </head>  
    """
    header_html = img_center + "<img src='data:image/png;base64,{}'>".format(
        img_to_bytes("alphastats_logo_2.png")
    )
    st.sidebar.markdown(
        header_html, unsafe_allow_html=True,
    )


def sidebar_info(show_logo=True):
    display_sidebar_html_table()
    st.sidebar.markdown("\n\n")
    st.sidebar.info("[AlphaStats on GitHub](https://github.com/MannLabs/alphastats)")
    st.sidebar.markdown(
        """ <head><style type ='text/css' > 
    .footer{ position: fixed;     
        text-align: left;    
        bottom: 14px; 
        width: 100%;
    }  
    </style>
    </head>
    <body>
        <div class="footer">Mann Group, 2022</div>
    </body>""",
        unsafe_allow_html=True,
    )
    if show_logo:
        sidebar_logo()


def display_sidebar_html_table():

    if "dataset" not in st.session_state:
        return

    preprocessing_dict = st.session_state.dataset.preprocessing_info

    html_string = (
        "<style>.mytable th, td{ font-size:10px;font-family:Arial, Helvetica, sans-serif;color:#8C878D; border-color:#96D4D4;}</style>"
        "<table class='mytable'>"
    )

    for key, values in preprocessing_dict.items():
        html_string += "<tr><td>" + key + "</td><td>" + str(values) + "</td>" + "</tr>"

    html_string += "</table>"
    st.sidebar.markdown(html_string, unsafe_allow_html=True)


def img_to_bytes(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    # img_bytes = Path(img_path).read_bytes()
    # encoded = base64.b64encode(img_bytes).decode()
    return encoded_string.decode()
