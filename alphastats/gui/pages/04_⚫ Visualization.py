import streamlit as st
from alphastats.gui.utils.analysis_helper import get_analysis
from alphastats.gui.utils.ui_helper import sidebar_info


def display_plotly_figure(plot):

    st.plotly_chart(plot)


def save_plot_to_session_state(plot, method):
    st.session_state["plot_list"] += [(method, plot)]


def make_plot():
    method = st.selectbox(
        "Plot",
        options=[
            "PCA",
            "t-SNE",
            "Sampledistribution",
            "Intensity",
            "Volcano",
            "Clustermap",
            "Dendrogram",
        ],
        key="plot_method",
    )
    plot = get_analysis(method=method, options_dict=st.session_state.plotting_options)
    return plot


st.markdown("### Visualization")

sidebar_info()

if "plot_list" not in st.session_state:
    st.session_state["plot_list"] = []

if "dataset" in st.session_state:

    plot = make_plot()

    if plot is not None:
        display_plotly_figure(plot)
        save_plot_to_session_state(plot, st.session_state.plot_method)


else:
    st.info("Import Data first")
