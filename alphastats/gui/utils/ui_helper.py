import streamlit as st
import pandas as pd
import base64
from alphastats import __version__

# TODO add logo above the options when issue is closed
# https://github.com/streamlit/streamlit/issues/4984


def sidebar_info(show_logo=True):
    display_sidebar_html_table()
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


class StateKeys:
    ## 02_Data Import
    # on 1st run
    ORGANISM = "organism"
    GENE_TO_PROT_ID = "gene_to_prot_id"
    USER_SESSION_ID = "user_session_id"
    LOADER = "loader"
    SOFTWARE = "software"
    # on sample run (function load_sample_data), removed on new session click
    DATASET = "dataset"  # functions upload_metadatafile
    PLOTTING_OPTIONS = "plotting_options"  # function load_options
    STATISTIC_OPTIONS = "statistic_options"  # function load_options
    DISTRIBUTION_PLOT = (
        "distribution_plot"  # function save_plot_sampledistribution_rawdata
    )
    METADATA_COLUMNS = (
        "metadata_columns"  # function create_metadata_file, upload_metadatafile
    )
    # on data upload
    INTENSITY_COLUMN = "intensity_column"
    INDEX_COLUMN = "index_column"
    # on metadata upload
    METADATAFILE = "metadatafile"
    SAMPLE_COLUMN = "sample_column"
