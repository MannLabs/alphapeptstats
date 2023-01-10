import streamlit as st
from alphastats.gui.utils.analysis_helper import get_analysis, load_options #, check_if_options_are_loaded
from alphastats.gui.utils.ui_helper import sidebar_info
import alphastats.gui.utils.analysis_helper
import pandas as pd
import io 

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


def save_plotly(plot, format):
    # Create an in-memory buffer
    buffer = io.BytesIO()
    # Save the figure as a pdf to the buffer
    plot[1].write_image(file=buffer, format=format)
    st.download_button(
        label="Download as " + format, data=buffer, file_name=plot[0] + "." + format
    )


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


def download_preprocessing_info(plot):
    preprocesing_dict = plot[1].preprocessing
    df = pd.DataFrame(preprocesing_dict.items())
    filename = "plot" + plot[0] + "preprocessing_info.csv"
    csv = convert_df(df)
    st.download_button(
        "Download DataSet Info as .csv",
        csv,
        filename,
        "text/csv",
        key="preprocessing" ,
    )


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

        method_plot = [st.session_state.plot_method, plot]

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            save_plotly(method_plot, format="pdf")

        with col2:
            save_plotly(method_plot, format="svg")

        with col3:
            download_preprocessing_info(method_plot)

        


else:
    st.info("Import Data first")
