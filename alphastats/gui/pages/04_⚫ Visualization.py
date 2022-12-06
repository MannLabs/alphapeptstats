import streamlit as st
from alphastats.gui.utils.analysis_helper import get_analysis, load_options #, check_if_options_are_loaded
from alphastats.gui.utils.ui_helper import sidebar_info
import alphastats.gui.utils.analysis_helper

def check_if_options_are_loaded(f):
    # decorator to check for missing values
    def inner(*args, **kwargs):
        if hasattr(st.session_state, "plotting_options") is False:
            alphastats.gui.utils.analysis_helper.load_options()
            
        return f(*args, **kwargs)

    return inner

def display_plotly_figure(plot):
    try:
        st.plotly_chart(plot.update_layout(plot_bgcolor="white"))
    except:
        st.pyplot(plot)


def save_plot_to_session_state(plot, method):
    st.session_state["plot_list"] += [(method, plot)]


@check_if_options_are_loaded
def make_plot():
    method = st.selectbox(
        "Plot",
        options=list(st.session_state.plotting_options.keys()),
        key="plot_method",
    )
    plot = get_analysis(method=method, options_dict=st.session_state.plotting_options)
    return plot


st.markdown("### Visualization")

sidebar_info()


# set background to white so downloaded pngs dont have grey background
styl = f"""
    <style>
        .css-jc5rf5 {{
            position: absolute;
            background: rgb(255, 255, 255);
            color: rgb(48, 46, 48);
            inset: 0px;
            overflow: hidden;
        }}
    </style>
    """
st.markdown(styl, unsafe_allow_html=True)

if "plot_list" not in st.session_state:
    st.session_state["plot_list"] = []

if "dataset" in st.session_state:

    plot = make_plot()

    if plot is not None:
        display_plotly_figure(plot)
        save_plot_to_session_state(plot, st.session_state.plot_method)


else:
    st.info("Import Data first")
