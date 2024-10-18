import pandas as pd
import streamlit as st

from alphastats.gui.utils.analysis_helper import display_plot
from alphastats.gui.utils.ui_helper import (
    StateKeys,
    convert_df,
    init_session_state,
    sidebar_info,
)


def display_plotly_figure(plot):
    st.plotly_chart(plot)


def download_preprocessing_info(plot, name, count):
    preprocesing_dict = plot.preprocessing
    df = pd.DataFrame(preprocesing_dict.items())
    filename = "plot" + name + "preprocessing_info.csv"
    csv = convert_df(df)
    print("preprocessing" + count)
    st.download_button(
        "Download DataSet Info as .csv",
        csv,
        filename,
        "text/csv",
        key="preprocessing" + count,
    )


init_session_state()
sidebar_info()

st.markdown("### Results")

if not st.session_state[StateKeys.PLOT_LIST]:
    st.info("No analysis saved yet.")
    st.stop()

for count, saved_item in enumerate(st.session_state[StateKeys.PLOT_LIST]):
    print("plot", type(saved_item), count)

    method = saved_item[0]
    plot = saved_item[1]

    if method == "ttest":
        plot = plot.plot

    st.markdown("\n\n")
    st.markdown(f"#### {method}")

    display_plot(method + str(count), plot, show_save_button=False)
