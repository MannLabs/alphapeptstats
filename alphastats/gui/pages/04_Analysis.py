import streamlit as st

from alphastats.gui.utils.analysis import PlottingOptions, StatisticOptions
from alphastats.gui.utils.analysis_helper import (
    display_analysis_result_with_buttons,
    gather_parameters_and_do_analysis,
)
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    init_session_state,
    sidebar_info,
)

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()

st.markdown("## Analysis")

# set background to white so downloaded pngs dont have grey background
styl = """
    <style>
        .css-jc5rf5 {
            position: absolute;
            background: rgb(255, 255, 255);
            color: rgb(48, 46, 48);
            inset: 0px;
            overflow: hidden;
        }
    </style>
    """
st.markdown(styl, unsafe_allow_html=True)

# TODO put everything in the session state for a given parameter set?
# or is caching functionality the way to go here?

if StateKeys.DATASET not in st.session_state:
    st.info("Import data first.")
    st.stop()

# --- SELECTION -------------------------------------------------------
show_plot = False
show_df = False
analysis_result = None


c1, c2 = st.columns([0.33, 0.67])
with c1:
    plotting_options = PlottingOptions.get_values()
    statistic_options = StatisticOptions.get_values()
    analysis_method = st.selectbox(
        "Analysis",
        options=["<select>"]
        + ["------- plots ------------"]
        + plotting_options
        + ["------- statistics -------"]
        + statistic_options,
    )

    if analysis_method in plotting_options:
        analysis_result, analysis_object, parameters = (
            gather_parameters_and_do_analysis(analysis_method)
        )

    elif analysis_method in statistic_options:
        analysis_result, _, parameters = gather_parameters_and_do_analysis(
            analysis_method,
        )

with c2:
    if analysis_result is not None:
        display_analysis_result_with_buttons(
            analysis_result, analysis_method, parameters
        )


@st.fragment
def show_start_llm_button(analysis_method: str) -> None:
    """Show the button to start the LLM analysis."""

    msg = (
        "(this will overwrite the existing LLM analysis!)"
        if st.session_state.get(StateKeys.LLM_INTEGRATION, {}) != {}
        else ""
    )

    submitted = st.button(
        f"Analyse with LLM ... {msg}",
        disabled=(analysis_method != PlottingOptions.VOLCANO_PLOT),
        help="Interpret the current analysis with an LLM (available for 'Volcano Plot' only).",
    )
    if submitted:
        if StateKeys.LLM_INTEGRATION in st.session_state:
            del st.session_state[StateKeys.LLM_INTEGRATION]
        st.session_state[StateKeys.LLM_INPUT] = (analysis_object, parameters)

        st.toast("LLM analysis created!", icon="✅")
        st.page_link("pages/05_LLM.py", label="=> Go to LLM page..")


if analysis_result is not None:
    show_start_llm_button(analysis_method)
