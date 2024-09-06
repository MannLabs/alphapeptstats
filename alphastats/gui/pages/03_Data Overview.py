import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from alphastats.gui.utils.overview_helper import (
        display_matrix,
        get_intensity_distribution_processed,
        get_sample_histogram_matrix,
    )
    from alphastats.gui.utils.ui_helper import sidebar_info, StateKeys

except ModuleNotFoundError:
    from utils.overview_helper import (
        display_matrix,
        get_intensity_distribution_processed,
        get_sample_histogram_matrix,
    )
    from utils.ui_helper import sidebar_info, StateKeys

sidebar_info()

if StateKeys.DATASET in st.session_state:
    st.markdown("## DataSet overview")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Intensity distribution raw data per sample**")
        st.plotly_chart(
            st.session_state[StateKeys.DISTRIBUTION_PLOT].update_layout(plot_bgcolor="white"),
            use_container_width=True,
        )

    with c2:
        st.markdown("**Intensity distribution data per sample used for analysis**")
        st.plotly_chart(
            get_intensity_distribution_processed(
                user_session_id=st.session_state[StateKeys.USER_SESSION_ID]
            ).update_layout(plot_bgcolor="white"),
            use_container_width=True,
        )

    st.plotly_chart(
        get_sample_histogram_matrix(
            user_session_id=st.session_state[StateKeys.USER_SESSION_ID]
        ).update_layout(plot_bgcolor="white"),
        use_container_width=True,
    )

    display_matrix()


else:
    st.info("Import Data first")
