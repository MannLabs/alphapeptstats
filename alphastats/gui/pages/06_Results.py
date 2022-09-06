import streamlit as st
import pandas as pd
import kaleido
from alphastats.gui.utils.ui_helper import sidebar_info

def display_plotly_figure(plot):
    st.plotly_chart(plot)

def save_plotly_to_pdf(plot):
    plot[1].write_image(plot[0]+".pdf")

def save_plotly_to_svg(plot):
    plot[1].write_image(plot[0]+".svg")

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

def download_preprocessing_info(plot, count):
    preprocesing_dict = plot[1].preprocessing
    df = pd.DataFrame(preprocesing_dict.items())
    filename = "plot" + plot[0] + "preprocessing_info.csv"
    csv = convert_df(df)
    st.download_button(
        "Preprocessing as .csv",
            csv,
            filename,
            "text/csv",
            key= "preprocessing" + count
            )

st.markdown("### Results")

sidebar_info()

if "plot_list" in st.session_state:
    for count, plot in enumerate(st.session_state.plot_list):
        count = str(count)
        
        st.markdown("\n\n")
        st.write(plot[0])
        
        display_plotly_figure(plot[1])
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.button('Save as pdf', key = "save_to_pdf" + count)
            if "save_to_pdf" + count in st.session_state:
                pass
            #    save_plotly_to_pdf(plot)
       
        with col2:
            st.button('Save as svg', key = "save_to_svg" + count)
            if "save_to_svg" + count in st.session_state:
                pass
            #    save_plotly_to_svg(plot)
        
        with col3:
            download_preprocessing_info(plot, count)
            


else:
    st.info("No analysis performed yet.")