import streamlit as st
from gui.utils.options import get_plotting_options, get_statistic_options

from alphastats.gui.utils.analysis_helper import (
    display_df,
    display_figure,
    download_figure,
    download_preprocessing_info,
    get_analysis,
    save_plot_to_session_state,
)
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    convert_df,
    init_session_state,
    sidebar_info,
)


def select_analysis():
    """
    select box
    loads keys from option dicts
    """
    method = st.selectbox(
        "Analysis",
        options=list(get_plotting_options(st.session_state).keys())
        + list(get_statistic_options(st.session_state).keys()),
    )
    return method


init_session_state()
sidebar_info()

st.markdown("### Analysis")

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


if StateKeys.PLOT_LIST not in st.session_state:
    st.session_state[StateKeys.PLOT_LIST] = []


if StateKeys.DATASET in st.session_state:
    c1, c2 = st.columns((1, 2))

    plot_to_display = False
    df_to_display = False
    method_plot = None

    with c1:
        method = select_analysis()

        if method in (plotting_options := get_plotting_options(st.session_state)):
            analysis_result = get_analysis(method=method, options_dict=plotting_options)
            plot_to_display = True

        elif method in (statistic_options := get_statistic_options(st.session_state)):
            analysis_result = get_analysis(
                method=method,
                options_dict=statistic_options,
            )
            df_to_display = True

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

    with c2:
        # --- Plot -------------------------------------------------------

        if analysis_result is not None and method != "Clustermap" and plot_to_display:
            display_figure(analysis_result)

            save_plot_to_session_state(analysis_result, method)

            method_plot = [method, analysis_result]

        elif method == "Clustermap":
            st.write("Download Figure to see full size.")

            display_figure(analysis_result)

            save_plot_to_session_state(analysis_result, method)

        # --- STATISTICAL ANALYSIS -------------------------------------------------------

        elif analysis_result is not None and df_to_display:
            display_df(analysis_result)

            filename = method + ".csv"
            csv = convert_df(analysis_result)

    if analysis_result is not None and method != "Clustermap" and plot_to_display:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            download_figure(method_plot, format="pdf")

        with col2:
            download_figure(method_plot, format="svg")

        with col3:
            download_preprocessing_info(method_plot)

    elif analysis_result is not None and df_to_display and method_plot:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            download_figure(method_plot, format="pdf", plotting_library="seaborn")

        with col2:
            download_figure(method_plot, format="svg", plotting_library="seaborn")

        with col3:
            download_preprocessing_info(method_plot)

    elif analysis_result is not None and df_to_display:
        st.download_button(
            "Download as .csv", csv, filename, "text/csv", key="download-csv"
        )


else:
    st.info("Import Data first")
