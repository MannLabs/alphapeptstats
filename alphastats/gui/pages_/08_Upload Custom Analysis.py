import pandas as pd
import streamlit as st
from dataset.keys import Cols, Regulation

from alphastats.gui.utils.analysis_helper import _save_analysis_to_session_state
from alphastats.gui.utils.result import ResultComponent
from alphastats.gui.utils.session_manager import STATE_SAVE_FOLDER_PATH, SessionManager
from alphastats.gui.utils.state_utils import init_session_state
from alphastats.gui.utils.ui_helper import sidebar_info


def parse_custom_analysis_file(uploaded_file) -> pd.DataFrame:
    """Parse uploaded custom analysis file and extract relevant columns."""
    try:
        # Read the uploaded file as tab-separated values
        df = pd.read_csv(uploaded_file, sep="\t")

        # Check if required columns exist
        required_columns = ["Significant", "Difference", "Protein IDs"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None

        # Extract only the required columns
        parsed_df = df[required_columns].copy()

        # Rename Protein IDs to index_
        parsed_df = parsed_df.rename(columns={"Protein IDs": "index_"})

        # Convert Significant column based on significance and difference direction
        # Handle both string values and NaN values
        significant_clean = parsed_df["Significant"].fillna("").astype(str)
        is_significant = significant_clean.map(
            {
                "+": True,
                "": False,
                " ": False,
                "nan": False,
            }
        ).fillna(False)

        # Create UP/DOWN based on significance and difference direction
        parsed_df[Cols.SIGNIFICANT] = "NON_SIG"  # Default value

        # Set UP for significant entries with positive difference
        up_mask = is_significant & (parsed_df["Difference"] > 0)
        parsed_df.loc[up_mask, Cols.SIGNIFICANT] = Regulation.UP

        # Set DOWN for significant entries with negative difference
        down_mask = is_significant & (parsed_df["Difference"] < 0)
        parsed_df.loc[down_mask, Cols.SIGNIFICANT] = Regulation.DOWN

        return parsed_df

    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None


def create_custom_result_component(parsed_df: pd.DataFrame) -> ResultComponent:
    """Create a simplified ResultComponent from parsed custom analysis data."""
    # Create basic ResultComponent with minimal required attributes
    result_component = ResultComponent(
        dataframe=parsed_df,
        preprocessing={},
        method={},
        feature_to_repr_map={},
        is_plottable=False,
    )

    # Set annotated_dataframe to the same data (pre-filled as requested)
    result_component.annotated_dataframe = parsed_df.copy()

    return result_component


def _finalize_custom_analysis_loading(
    result_component: ResultComponent, parameters: dict
) -> None:
    """Finalize the custom analysis loading process."""
    # Save to session state using existing infrastructure
    _save_analysis_to_session_state(
        analysis_results=result_component,
        method="Custom Analysis Upload",
        parameters=parameters,
    )

    st.toast("Custom analysis has been uploaded successfully!", icon="✅")
    st.page_link("pages_/06_LLM.py", label="➔ Go to LLM page to analyze results..")
    st.page_link(
        "pages_/07_Results.py", label="➔ Go to results page to view analysis.."
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
st.write(
    "Upload a tab-separated file containing custom analysis results. "
    "The file should contain 'Significant', 'Difference', and 'Protein IDs' columns."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a custom analysis file",
    type=["txt", "tsv", "csv"],
    help="Upload a tab-separated file (e.g., Fig4A_VolcanoStats.txt)",
)

if uploaded_file is not None:
    st.markdown("##### File Preview")

    # Parse the file
    parsed_df = parse_custom_analysis_file(uploaded_file)

    if parsed_df is not None:
        # Show preview of the data
        st.write(f"File contains {len(parsed_df)} rows")
        st.dataframe(parsed_df.head(10), use_container_width=True)

        # Show summary statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total entries", len(parsed_df))
        # with col2:
        #     significant_count = (
        #         parsed_df[Cols.SIGNIFICANT].sum()
        #         if Regulation.SIG in parsed_df.columns
        #         else 0
        #     )
        #     st.metric("Significant entries", significant_count)

        st.markdown("##### Analysis Parameters")

        # Input fields for group1, group2, and column
        param_col1, param_col2, param_col3 = st.columns(3)
        with param_col1:
            group1 = st.text_input(
                "Group 1", placeholder="e.g., Control", key="custom_group1"
            )
        with param_col2:
            group2 = st.text_input(
                "Group 2", placeholder="e.g., Treatment", key="custom_group2"
            )
        with param_col3:
            column = st.text_input(
                "Column", placeholder="e.g., condition", key="custom_column"
            )

        st.markdown("##### Create Analysis Object")

        if st.button("Create Custom Analysis", type="primary"):
            # Validate that all parameters are provided
            if not group1 or not group2 or not column:
                st.error(
                    "Please fill in all three parameters: Group 1, Group 2, and Column"
                )
            else:
                # Create parameters dictionary
                parameters = {"group1": group1, "group2": group2, "column": column}

                # Create ResultComponent
                result_component = create_custom_result_component(parsed_df)

                # Finalize loading with parameters
                _finalize_custom_analysis_loading(result_component, parameters)

else:
    st.info("Please upload a custom analysis file to continue.")

    # Show example of expected format
    st.markdown("##### Expected File Format")
    st.write("Your file should be tab-separated with at least these columns:")

    example_data = {
        "Significant": ["+", "", "+", ""],
        "Difference": [1.73, -0.83, 2.14, -0.28],
        "Protein IDs": ["P12345;Q67890", "P23456", "P34567;Q78901;R90123", "P45678"],
    }
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)
