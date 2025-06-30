import pandas as pd
import streamlit as st

from alphastats.dataset.keys import Cols, Regulation
from alphastats.gui.utils.analysis import CUSTOM_ANALYSIS
from alphastats.gui.utils.analysis_helper import _save_analysis_to_session_state
from alphastats.gui.utils.session_manager import STATE_SAVE_FOLDER_PATH, SessionManager
from alphastats.gui.utils.state_utils import init_session_state
from alphastats.gui.utils.ui_helper import sidebar_info
from alphastats.gui.utils.upload_custom_analysis import (
    create_custom_result_component,
    parse_custom_analysis_file,
)

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()

st.markdown("## Upload Custom Analysis")

# Show existing saved sessions info
saved_sessions = SessionManager.get_saved_sessions(STATE_SAVE_FOLDER_PATH)
if saved_sessions:
    st.markdown("### Load a saved session")
    st.page_link(
        "pages_/01_Home.py", label="➔ Load a previous session on the main page.."
    )

st.markdown("### Upload Custom Analysis File")
upload_help_text = (
    "Upload a tab-separated file containing custom analysis results. "
    "The file should contain 'Significant', 'Difference', and 'Protein IDs' columns."
)
st.write(upload_help_text)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a custom analysis file",
    type=["txt", "tsv", "csv"],
    help=upload_help_text,
)

if uploaded_file is not None:
    st.markdown("##### File Preview")

    # Parse the file
    try:
        parsed_df = parse_custom_analysis_file(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        st.stop()

    # Show preview of the data
    st.write(f"File contains {len(parsed_df)} rows")
    st.dataframe(parsed_df.head(10), use_container_width=True)

    # Show summary statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total entries", len(parsed_df))
    with col2:
        st.metric(
            "Significant entries",
            len(parsed_df[parsed_df[Cols.SIGNIFICANT] != Regulation.NON_SIG]),
        )

    st.markdown("##### Analysis Parameters")

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        group1 = st.text_input(
            "Group 1",
            placeholder="e.g., Treatment (positive fold change values mean a protein is upregulated in this group)",
            key="custom_group1",
        )
    with param_col2:
        group2 = st.text_input(
            "Group 2",
            placeholder="e.g., Control (negative fold change values mean a protein is upregulated in this group)",
            key="custom_group2",
        )
    with param_col3:
        column = st.text_input(
            "Grouping column name", placeholder="e.g., condition", key="grouping_column"
        )

    st.markdown("##### Create Analysis Object")

    if st.button("Create Custom Analysis", type="primary"):
        if not group1 or not group2 or not column:
            st.error(
                "Please fill in all three parameters: Group 1, Group 2, and Column!"
            )
        elif group1 == group2:
            st.error("Group 1 and Group 2 must differ!")
        else:
            parameters = {"group1": group1, "group2": group2, "column": column}

            result_component, id_holder = create_custom_result_component(parsed_df)

            _save_analysis_to_session_state(
                analysis_results=result_component,
                method=CUSTOM_ANALYSIS,
                parameters=parameters,
                id_holder=id_holder,
            )

            st.toast("Custom analysis has been uploaded successfully!", icon="✅")
            st.page_link(
                "pages_/06_LLM.py", label="➔ Go to LLM page to analyze results.."
            )
            st.page_link(
                "pages_/07_Results.py", label="➔ Go to results page to view analysis.."
            )

else:
    st.info("Please upload a custom analysis file to continue.")

    st.markdown("##### Expected File Format")
    st.write("Your file should be tab-separated with at least these columns:")

    example_data = {
        "Significant": ["+", "", "+", ""],
        "Difference": [1.73, -0.83, 2.14, -0.28],
        "Majority Protein IDs": [
            "P12345;Q67890",
            "P23456",
            "P34567;Q78901;R90123",
            "P45678",
        ],
        "Gene names": ["Gene1", "Gene2", "Gene3", "Gene4"],
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)
