import streamlit as st

import pandas as pd
import io


try:
    from alphastats.gui.utils.ui_helper import sidebar_info
    from alphastats.gui.utils.analysis_helper import (
    get_analysis,
)  

except ModuleNotFoundError:
    from utils.ui_helper import sidebar_info
    from utils.analysis_helper import (
    get_analysis,
)  

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


@st.cache_data
def convert_df(df, user_session_id = st.session_state.user_session_id):
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

    c1, c2 = st.columns((1,2))

    plot_to_display = False
    df_to_display = False

    with c1:

        method = select_analysis()

        if method in st.session_state.plotting_options.keys():

            analysis_result = get_analysis(
                method=method, options_dict=st.session_state.plotting_options
            )
            plot_to_display = True
        
        elif method in st.session_state.statistic_options.keys():

            analysis_result = get_analysis(
                method=method, options_dict=st.session_state.statistic_options
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

    
    elif analysis_result is not None and df_to_display:
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
