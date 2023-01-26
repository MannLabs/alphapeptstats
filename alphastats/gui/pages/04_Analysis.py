import streamlit as st
from alphastats.gui.utils.analysis_helper import (
    get_analysis,
    load_options,
)  # , check_if_options_are_loaded
from alphastats.gui.utils.ui_helper import sidebar_info
import alphastats.gui.utils.analysis_helper
import pandas as pd
import io


def check_if_options_are_loaded(f):
    """
    decorator to check whether analysis options are loaded
    """

    def inner(*args, **kwargs):

        if hasattr(st.session_state, "plotting_options") is False:
            alphastats.gui.utils.analysis_helper.load_options()

        return f(*args, **kwargs)

    return inner


def display_figure(plot):
    """
    display plotly or seaborn figure
    """
    try:
        st.plotly_chart(plot.update_layout(plot_bgcolor="white"))
    except:
        st.pyplot(plot)


def save_plot_to_session_state(plot, method):
    """
    save plot with method to session state to retrieve old results
    """
    st.session_state["plot_list"] += [(method, plot)]


def display_df(df):
    mask = df.applymap(type) != bool
    d = {True: "TRUE", False: "FALSE"}
    df = df.where(mask, df.replace(d))
    st.dataframe(df)


def download_figure(obj, format, plotting_library="plotly"):
    """
    download plotly figure
    """

    plot = obj[1]
    filename = obj[0] + "." + format

    buffer = io.BytesIO()

    if plotting_library == "plotly":
        # Save the figure as a pdf to the buffer
        plot.write_image(file=buffer, format=format)

    else:
        plot.savefig(buffer, format=format)

    st.download_button(label="Download as " + format, data=buffer, file_name=filename)


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
        key="preprocessing",
    )


@check_if_options_are_loaded
def select_analysis():
    """
    select box
    loads keys from option dicts
    """
    method = st.selectbox(
        "Analysis",
        options=list(st.session_state.plotting_options.keys())
        + list(st.session_state.statistic_options.keys()),
    )
    return method


st.markdown("### Analysis")

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

    method = select_analysis()

    # --- PLOT ----------------------------------------------------------------------------------------------------------

    if method in st.session_state.plotting_options.keys():

        analysis_result = get_analysis(
            method=method, options_dict=st.session_state.plotting_options
        )

        if analysis_result is not None and method != "Clustermap":
            display_figure(analysis_result)

            save_plot_to_session_state(analysis_result, method)

            method_plot = [method, analysis_result]

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                download_figure(method_plot, format="pdf")

            with col2:
                download_figure(method_plot, format="svg")

            with col3:
                download_preprocessing_info(method_plot)

        elif method == "Clustermap":

            st.write("Download Figure to see full size.")

            display_figure(analysis_result)

            save_plot_to_session_state(analysis_result, method)

            method_plot = [method, analysis_result]

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                download_figure(method_plot, format="pdf", plotting_library="seaborn")

            with col2:
                download_figure(method_plot, format="svg", plotting_library="seaborn")

            with col3:
                download_preprocessing_info(method_plot)

    # --- STATISTICAL ANALYSIS ------------------------------------------------------------------------------------------

    elif method in st.session_state.statistic_options.keys():

        analysis_result = get_analysis(
            method=method, options_dict=st.session_state.statistic_options
        )

        if analysis_result is not None:

            display_df(analysis_result)

            filename = method + ".csv"
            csv = convert_df(analysis_result)

            st.download_button(
                "Download as .csv", csv, filename, "text/csv", key="download-csv"
            )


else:
    st.info("Import Data first")
