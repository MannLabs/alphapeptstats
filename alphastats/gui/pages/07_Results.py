import streamlit as st

from alphastats.gui.utils.analysis_helper import (
    display_analysis_result_with_buttons,
)
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

st.markdown("## Results")

if not st.session_state[StateKeys.SAVED_ANALYSES]:
    st.info("No analysis saved yet.")
    st.stop()

for key, saved_analysis in st.session_state[StateKeys.SAVED_ANALYSES].items():
    analysis_result = saved_analysis["result"]
    method = saved_analysis["method"]
    parameters = saved_analysis["parameters"]
    number = saved_analysis["number"]

    st.markdown("\n\n\n")
    st.markdown(f"#### #{number}: {method} [{key}]")
    st.markdown(f"Parameters used for analysis: `{parameters}`")

    name = f"{method}_{number}"

    if st.button(f"❌ Remove analysis #{number}", key=f"remove_{name}"):
        del st.session_state[StateKeys.SAVED_ANALYSES][key]
        st.rerun()

    display_analysis_result_with_buttons(
        analysis_result,
        analysis_method=method,
        parameters=parameters,
        show_save_button=False,
        name=name,
        editable_annotation=False,
    )
