import streamlit as st

from alphastats.gui.utils.analysis_helper import (
    display_analysis_result_with_buttons,
)
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    init_session_state,
    sidebar_info,
)

init_session_state()
sidebar_info()

st.markdown("### Results")

if not st.session_state[StateKeys.ANALYSIS_LIST]:
    st.info("No analysis saved yet.")
    st.stop()

for count, saved_analysis in enumerate(st.session_state[StateKeys.ANALYSIS_LIST]):
    analysis_result = saved_analysis[0]
    method = saved_analysis[1]
    parameters = saved_analysis[2]

    st.markdown("\n\n\n")
    st.markdown(f"#### {method}")
    st.write(f"Parameters used for analysis: {parameters}")

    name = f"{method}_{count}"

    if st.button("x remove analysis", key=f"remove_{name}"):
        st.session_state[StateKeys.ANALYSIS_LIST].remove(saved_analysis)
        st.rerun()

    display_analysis_result_with_buttons(
        analysis_result,
        analysis_method=method,
        parameters=parameters,
        show_save_button=False,
        name=name,
    )
