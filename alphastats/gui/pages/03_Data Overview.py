import streamlit as st

from alphastats.gui.utils.overview_helper import (
    display_matrix,
    get_intensity_distribution_processed,
    get_sample_histogram_matrix,
    get_intensity_distribution_unprocessed,
    display_loaded_dataset,
)
from alphastats.gui.utils.ui_helper import sidebar_info, init_session_state, StateKeys

init_session_state()
sidebar_info()

if "dataset" not in st.session_state:
    st.info("Import Data first")
    st.stop()

st.markdown("### DataSet Info")

display_loaded_dataset(st.session_state[StateKeys.DATASET])

st.markdown("## DataSet overview")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Intensity distribution raw data per sample**")
    st.plotly_chart(
        get_intensity_distribution_unprocessed().update_layout(plot_bgcolor="white"),
        use_container_width=True,
    )

with c2:
    st.markdown("**Intensity distribution data per sample used for analysis**")
    st.plotly_chart(
        get_intensity_distribution_processed().update_layout(plot_bgcolor="white"),
        use_container_width=True,
    )

st.plotly_chart(
    get_sample_histogram_matrix().update_layout(plot_bgcolor="white"),
    use_container_width=True,
)

display_matrix()
