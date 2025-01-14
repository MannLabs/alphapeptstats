import base64
import uuid

import pandas as pd
import streamlit as st

from alphastats import __version__
from alphastats.dataset.keys import ConstantsClass
from alphastats.gui.utils.preprocessing_helper import PREPROCESSING_STEPS
from alphastats.llm.uniprot_utils import ExtractedUniprotFields

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


def empty_session_state():
    """
    remove all variables to avoid conflicts
    """
    for key in st.session_state:
        del st.session_state[key]
    st.empty()


class DefaultStates(metaclass=ConstantsClass):
    SELECTED_UNIPROT_FIELDS = [
        ExtractedUniprotFields.NAME,
        ExtractedUniprotFields.GENE,
        ExtractedUniprotFields.FUNCTIONCOMM,
    ]
    WORKFLOW = [
        PREPROCESSING_STEPS.REMOVE_CONTAMINATIONS,
        PREPROCESSING_STEPS.SUBSET,
        PREPROCESSING_STEPS.REPLACE_ZEROES,
        PREPROCESSING_STEPS.LOG2_TRANSFORM,
        PREPROCESSING_STEPS.DROP_UNMEASURED_FEATURES,
    ]


def init_session_state() -> None:
    """Initialize the session state if not done yet."""

    if StateKeys.USER_SESSION_ID not in st.session_state:
        st.session_state[StateKeys.USER_SESSION_ID] = str(uuid.uuid4())

    if StateKeys.ORGANISM not in st.session_state:
        st.session_state[StateKeys.ORGANISM] = 9606  # human

    if StateKeys.WORKFLOW not in st.session_state:
        st.session_state[StateKeys.WORKFLOW] = DefaultStates.WORKFLOW.copy()

    if StateKeys.ANALYSIS_LIST not in st.session_state:
        st.session_state[StateKeys.ANALYSIS_LIST] = []

    if StateKeys.LLM_INTEGRATION not in st.session_state:
        st.session_state[StateKeys.LLM_INTEGRATION] = {}

    if StateKeys.ANNOTATION_STORE not in st.session_state:
        st.session_state[StateKeys.ANNOTATION_STORE] = {}

    if StateKeys.SELECTED_UNIPROT_FIELDS not in st.session_state:
        st.session_state[StateKeys.SELECTED_UNIPROT_FIELDS] = (
            DefaultStates.SELECTED_UNIPROT_FIELDS.copy()
        )

    if StateKeys.MAX_TOKENS not in st.session_state:
        st.session_state[StateKeys.MAX_TOKENS] = 10000


class StateKeys(metaclass=ConstantsClass):
    USER_SESSION_ID = "user_session_id"
    DATASET = "dataset"

    WORKFLOW = "workflow"

    ANALYSIS_LIST = "analysis_list"

    # LLM
    OPENAI_API_KEY = "openai_api_key"  # pragma: allowlist secret
    MODEL_NAME = "model_name"
    LLM_INPUT = "llm_input"
    LLM_INTEGRATION = "llm_integration"
    ANNOTATION_STORE = "annotation_store"
    SELECTED_UNIPROT_FIELDS = "selected_uniprot_fields"
    MAX_TOKENS = "max_tokens"

    ORGANISM = "organism"  # TODO this is essentially a constant
