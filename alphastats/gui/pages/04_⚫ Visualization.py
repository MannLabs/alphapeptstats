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
    plot = get_analysis(method=method, options_dict=options_dict)
    return plot


st.markdown("### Visualization")

sidebar_info()

if "plot_list" not in st.session_state:
    st.session_state["plot_list"] = []

if "dataset" in st.session_state:

    from alphastats.gui.utils.options import plotting_options as options_dict

    # if "plot_list" in st.session_state:
    #     show_previous_plots = st.button(label="Show previous plots")
    #     if show_previous_plots:
    #         for p in st.session_state.plot_list:
    #             display_plotly_figure(p)

    # if "n_rows" not in st.session_state:
    #     st.session_state.n_rows = 1
    # add = st.button(label="Add new plot")

    # if add:
    #     st.session_state.n_rows += 1
    #     st.experimental_rerun()

    # for i in range(st.session_state.n_rows):
    # add text inputs here
    plot = make_plot()

    if plot is not None:
        display_plotly_figure(plot)
        save_plot_to_session_state(plot, st.session_state.plot_method)


else:
    st.info("Import Data first")
