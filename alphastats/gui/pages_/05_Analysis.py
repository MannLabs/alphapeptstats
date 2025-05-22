import streamlit as st

from alphastats.gui.utils.analysis import (
    NewAnalysisOptions,
    PlottingOptions,
    StatisticOptions,
)
from alphastats.gui.utils.analysis_helper import (
    display_analysis_result_with_buttons,
    gather_parameters_and_do_analysis,
)
from alphastats.gui.utils.llm_helper import LLM_ENABLED_ANALYSIS
from alphastats.gui.utils.state_keys import (
    StateKeys,
)
from alphastats.gui.utils.state_utils import (
    init_session_state,
)
from alphastats.gui.utils.ui_helper import (
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

# TODO use caching functionality for all analysis (not: plot creation)

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
    new_options = NewAnalysisOptions.get_values()
    analysis_method = st.selectbox(
        "Analysis",
        options=["<select>"]
        + new_options
        + ["------- plots ------------"]
        + plotting_options
        + ["------- statistics -------"]
        + statistic_options,
        format_func=lambda x: x if x not in LLM_ENABLED_ANALYSIS else "💬 " + x,
    )

    if analysis_method in plotting_options or analysis_method in new_options:
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
