import streamlit as st

from alphastats.gui.utils.analysis import PlottingOptions, StatisticOptions
from alphastats.gui.utils.analysis_helper import display_df, display_plot
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

for count, saved_item in enumerate(st.session_state[StateKeys.ANALYSIS_LIST]):
    print("plot", type(saved_item), count)

    plot = saved_item[0]
    method = saved_item[1]
    parameters = saved_item[2]

    st.markdown("\n\n\n")
    st.markdown(f"#### {method}")
    st.write(f"Parameters used for analysis: {parameters}")

    if st.button("x remove analysis", key="remove" + method + str(count)):
        st.session_state[StateKeys.ANALYSIS_LIST].remove(saved_item)
        st.rerun()

    name = f"{method}_{count}"
    if method in PlottingOptions.get_values():
        display_plot(method, plot, parameters, show_save_button=False, name=name)
    elif method in StatisticOptions.get_values():
        display_df(method, plot, parameters, show_save_button=False, name=name)
