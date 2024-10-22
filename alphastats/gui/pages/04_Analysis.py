import streamlit as st

from alphastats.gui.utils.analysis_helper import (
    display_df,
    display_plot,
    do_analysis,
)
from alphastats.gui.utils.options import get_plotting_options, get_statistic_options
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    convert_df,
    init_session_state,
    sidebar_info,
)

init_session_state()
sidebar_info()

st.markdown("### Analysis")

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


if StateKeys.DATASET not in st.session_state:
    st.info("Import Data first")
    st.stop()

# --- SELECTION -------------------------------------------------------
show_plot = False
show_df = False
analysis_result = None

c1, *_ = st.columns([1, 1, 1])
with c1:
    method = st.selectbox(
        "Analysis",
        options=["<select>"]
        + ["------- plots -------"]
        + list(get_plotting_options(st.session_state).keys())
        + ["------- statistics -------"]
        + list(get_statistic_options(st.session_state).keys()),
    )

    if method in (plotting_options := get_plotting_options(st.session_state)):
        analysis_result, analysis_object, parameters = do_analysis(
            method, options_dict=plotting_options
        )
        show_plot = analysis_result is not None

    elif method in (statistic_options := get_statistic_options(st.session_state)):
        analysis_result, *_ = do_analysis(
            method,
            options_dict=statistic_options,
        )
        show_df = analysis_result is not None


# --- SHOW PLOT -------------------------------------------------------
if show_plot:
    display_plot(method, analysis_result)

# --- SHOW STATISTICAL ANALYSIS -------------------------------------------------------
elif show_df:
    display_df(analysis_result)

    csv = convert_df(analysis_result)

    # TODO do we want to save statistical analysis to results page as well?
    st.download_button(
        "Download as .csv", csv, method + ".csv", "text/csv", key="download-csv"
    )


@st.fragment
def show_start_llm_button(method: str) -> None:
    """Show the button to start the LLM analysis."""

    msg = (
        "(Note this will overwrite the existing LLM analysis!)"
        if StateKeys.LLM_INTEGRATION in st.session_state
        else ""
    )

    submitted = st.button(
        f"Analyse with LLM ... {msg}",
        disabled=(method != "Volcano Plot"),
        help="Interpret the current analysis with an LLM (available for 'Volcano Plot' only).",
    )
    if submitted:
        if StateKeys.LLM_INTEGRATION in st.session_state:
            del st.session_state[StateKeys.LLM_INTEGRATION]
        st.session_state[StateKeys.LLM_INPUT] = (analysis_object, parameters)

        st.page_link("pages/05_LLM.py", label="=> Go to LLM page..")


if analysis_result is not None:
    show_start_llm_button(method)
