import streamlit as st

from alphastats.gui.utils.import_helper import (
    display_loaded_dataset,
)

try:
    from alphastats.gui.utils.overview_helper import (
        display_matrix,
        get_intensity_distribution_processed,
        get_sample_histogram_matrix,
    )
    from alphastats.gui.utils.ui_helper import sidebar_info

except ModuleNotFoundError:
    from utils.overview_helper import (
        display_matrix,
        get_intensity_distribution_processed,
        get_sample_histogram_matrix,
    )
    from utils.ui_helper import sidebar_info

sidebar_info()

if "dataset" in st.session_state:
    st.markdown("### DataSet Info")

    display_loaded_dataset(st.session_state["dataset"])

    st.markdown("## DataSet overview")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Intensity distribution raw data per sample**")
        st.plotly_chart(
            st.session_state["distribution_plot"].update_layout(plot_bgcolor="white"),
            use_container_width=True,
        )

    with c2:
        st.markdown("**Intensity distribution data per sample used for analysis**")
        st.plotly_chart(
            get_intensity_distribution_processed(
                user_session_id=st.session_state.user_session_id
            ).update_layout(plot_bgcolor="white"),
            use_container_width=True,
        )

    st.plotly_chart(
        get_sample_histogram_matrix(
            user_session_id=st.session_state.user_session_id
        ).update_layout(plot_bgcolor="white"),
        use_container_width=True,
    )

    display_matrix()


else:
    st.info("Import Data first")
