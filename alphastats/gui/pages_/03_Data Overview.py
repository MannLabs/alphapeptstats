import streamlit as st

from alphastats.gui.utils.overview_helper import (
    display_loaded_dataset,
    display_matrix,
    get_intensity_distribution_processed,
    get_intensity_distribution_unprocessed,
    get_sample_histogram_matrix,
)
from alphastats.gui.utils.state_keys import StateKeys
from alphastats.gui.utils.state_utils import init_session_state
from alphastats.gui.utils.ui_helper import sidebar_info

st.set_page_config(layout="wide")
init_session_state()
sidebar_info()

st.markdown("## Data Overview")

if StateKeys.DATASET not in st.session_state:
    st.info("Import data first.")
    st.stop()

display_loaded_dataset(st.session_state[StateKeys.DATASET])

st.markdown("### Intensities")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Intensity distribution raw data per sample**")
    st.plotly_chart(
        get_intensity_distribution_unprocessed().update_layout(plot_bgcolor="white"),
        use_container_width=True,
        key="fig_intensity_distribution_unprocessed",
    )

with c2:
    st.markdown("**Intensity distribution data per sample used for analysis**")
    st.plotly_chart(
        get_intensity_distribution_processed().update_layout(plot_bgcolor="white"),
        use_container_width=True,
        key="fig_intensity_distribution_processed",
    )

st.plotly_chart(
    get_sample_histogram_matrix().update_layout(plot_bgcolor="white"),
    use_container_width=True,
    key="fig_sample_histogram_matrix",
)

display_matrix()
