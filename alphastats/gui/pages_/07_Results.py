import streamlit as st

from alphastats.gui.utils.analysis_helper import (
    display_analysis_result_with_buttons,
)
from alphastats.gui.utils.llm_helper import show_llm_chat
from alphastats.gui.utils.state_keys import (
    LLMKeys,
    SavedAnalysisKeys,
    StateKeys,
)
from alphastats.gui.utils.state_utils import (
    init_session_state,
)
from alphastats.gui.utils.ui_helper import (
    has_llm_support,
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
    analysis_result = saved_analysis[SavedAnalysisKeys.RESULT]
    method = saved_analysis[SavedAnalysisKeys.METHOD]
    parameters = saved_analysis[SavedAnalysisKeys.PARAMETERS]
    number = saved_analysis[SavedAnalysisKeys.NUMBER]

    st.markdown("\n\n\n")
    st.markdown(f"### #{number}: {method} [{key}]")
    st.markdown(f"Parameters used for analysis: `{parameters}`")

    name = f"{method}_{number}"

    if st.button(
        f"❌ Remove analysis #{number}",
        key=f"remove_{name}",
        help="Also removes the associated LLM chat" if has_llm_support() else "",
    ):
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
    if has_llm_support():
        st.markdown("#### LLM Chat")
        if (
            llm_integration := st.session_state.get(StateKeys.LLM_CHATS, {})
            .get(key, {})
            .get(LLMKeys.LLM_INTEGRATION)
        ) is not None:
            with st.expander("LLM Chat (read-only)", expanded=False):
                show_llm_chat(llm_integration, key)
        else:
            st.write("No LLM chat available yet for this analysis.")

        # passing parameters is not possible yet https://github.com/streamlit/streamlit/issues/8112
        st.page_link("pages_/06_LLM.py", label="=> Create/Continue chat...")
