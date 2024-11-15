import streamlit as st

from alphastats.gui.utils.analysis_helper import display_plot
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    init_session_state,
    sidebar_info,
)

init_session_state()
sidebar_info()

st.markdown("### Results")

if not st.session_state[StateKeys.PLOT_LIST]:
    st.info("No analysis saved yet.")
    st.stop()

for count, saved_item in enumerate(st.session_state[StateKeys.PLOT_LIST]):
    print("plot", type(saved_item), count)

    method = saved_item[0]
    plot = saved_item[1]
    parameters = saved_item[2]

    st.markdown("\n\n\n")
    st.markdown(f"#### {method}")
    st.write(f"Parameters used for analysis: {parameters}")

    if st.button("x remove analysis", key="remove" + method + str(count)):
        st.session_state[StateKeys.PLOT_LIST].remove(saved_item)
        st.rerun()

    display_plot(method + str(count), plot, show_save_button=False)
