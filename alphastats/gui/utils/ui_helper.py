import base64

import pandas as pd
import streamlit as st

from alphastats import __version__
from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.session_manager import SessionManager
from alphastats.gui.utils.state_keys import StateKeys

# TODO add logo above the options when issue is closed
# https://github.com/streamlit/streamlit/issues/4984


def sidebar_info():
    st.sidebar.markdown("### Save session")
    session_name = st.sidebar.text_input(
        "Session name",
        max_chars=32,
        placeholder="(optional)",
        help="Optional name of the session to save. Needs to be alphanumeric.",
        value="",
    )
    if st.sidebar.button(
        "Save session",
        help="Saves the session to disk to be able to load it later. Note that if AlphaPeptStats is running in a hosted environment, the session might become visible to others.",
        disabled=session_name != "" and not session_name.isalnum(),
    ):
        saved_file_path = SessionManager().save(st.session_state, session_name)
        st.sidebar.success(f"Session saved to {saved_file_path}")
    st.sidebar.divider()

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
    if StateKeys.DATASET not in st.session_state:
        return

    preprocessing_dict = st.session_state[StateKeys.DATASET].preprocessing_info

    html_string = (
        "<style>.mytable th, td{ font-size:10px;font-family:Arial, Helvetica, sans-serif;color:#8C878D; border-color:#96D4D4;}</style>"
        "<table class='mytable'>"
    )

    for key, values in preprocessing_dict.items():
        html_string += "<tr><td>" + key + "</td><td>" + str(values) + "</td>" + "</tr>"

    html_string += "</table>"

    st.sidebar.markdown("### DateSet info")
    st.sidebar.markdown(html_string, unsafe_allow_html=True)


def img_to_bytes(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    # img_bytes = Path(img_path).read_bytes()
    # encoded = base64.b64encode(img_bytes).decode()
    return encoded_string.decode()


# @st.cache_data  # TODO check if caching is sensible here and if so, reimplement with dataset-hash
def _convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def show_button_download_df(
    df: pd.DataFrame, file_name: str, label="Download as .csv"
) -> None:
    """Show a button to download a dataframe as .csv."""
    csv = _convert_df_to_csv(df)

    st.download_button(
        label,
        csv,
        file_name + ".csv",
        "text/csv",
        key=f"download-csv-{file_name}",
    )


class AnalysisParameters(metaclass=ConstantsClass):
    TWOGROUP_GROUP1 = "group1"
    TWOGROUP_GROUP2 = "group2"
    DEA_TWOGROUPS_METHOD = "method"
    DEA_TWOGROUPS_FDR_METHOD = "fdr_method"
    TWOGROUP_COLUMN = "column"


class ResultParameters(metaclass=ConstantsClass):
    WIDTH = "width"
    HEIGHT = "height"
    SHOWLEGEND = "showlegend"
    QVALUE_CUTOFF = "qvalue_cutoff"
    LOG2FC_CUTOFF = "log2fc_cutoff"
    FLIP_XAXIS = "flip_xaxis"
    DRAW_LINES = "draw_lines"
    LABEL_SIGNIFICANT = "label_significant"
    RENDERER = "renderer"


def has_llm_support():
    """Check if the current environment has LLM support."""
    return False
