import uuid

import streamlit as st
import pandas as pd
import base64
from alphastats import __version__

# TODO add logo above the options when issue is closed
# https://github.com/streamlit/streamlit/issues/4984


def sidebar_info():
    _display_sidebar_html_table()
    st.sidebar.markdown("\n\n")
    st.sidebar.markdown("AlphaPeptStats Version " + str(__version__))
    st.sidebar.info(
        "[AlphaPeptStats on GitHub](https://github.com/MannLabs/alphapeptstats)"
    )
    st.sidebar.info(
        "[Documentation](https://alphapeptstats.readthedocs.io/en/latest/index.html)"
    )

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
        <div class="footer">Mann Group, 2024</div>
    </body>""",
        unsafe_allow_html=True,
    )


def _display_sidebar_html_table():
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


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def empty_session_state():
    """
    remove all variables to avoid conflicts
    """
    for key in st.session_state.keys():
        del st.session_state[key]
    st.empty()


def init_session_state() -> None:
    """Initialize the session state."""

    st.session_state["user_session_id"] = str(uuid.uuid4())

    if "gene_to_prot_id" not in st.session_state:
        st.session_state["gene_to_prot_id"] = {}

    if "organism" not in st.session_state:
        st.session_state["organism"] = 9606  # human
