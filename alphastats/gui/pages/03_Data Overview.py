
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")

def get_normalization_imputation_text():
    normalization = st.session_state.dataset.preprocessing_info["Normalization"]
    imputation = st.session_state.dataset.preprocessing_info["Imputation"]
    text = "Normalization: " + str(normalization) + ", Imputation: " + str(imputation)
    return text

def plot_sampledistribution_rawdata():
    df = st.session_state.dataset.rawmat
    df = df.unstack().reset_index()
    df.rename(columns={"level_1": st.session_state.dataset.sample, 0: "Intensity"}, inplace=True)
    return px.violin(df, x=st.session_state.dataset.sample, y="Intensity")



def display_matrix():

    text = get_normalization_imputation_text()

    st.markdown("### DataFrame used for analysis")
    st.markdown(text)

    processed_df = pd.DataFrame(
            st.session_state.dataset.mat.values,
            index=st.session_state.dataset.mat.index.to_list(),
        ).head(10)
    
    st.dataframe(processed_df)
        
    csv = convert_df(processed_df)
    st.download_button(
                    "Download as .csv", csv, "analysis_matrix.csv", "text/csv", key="download-csv"
                )

    

if "dataset" in st.session_state:
    st.markdown("## DataSet overview")

    st.markdown("#### Intensity distribution raw data per sample")
    fig_raw = plot_sampledistribution_rawdata()
    st.plotly_chart(fig_raw.update_layout(plot_bgcolor="white"))

    st.markdown("#### Intensity distribution processed data per sample")
    fig_processed = st.session_state.dataset.plot_sampledistribution()
    st.plotly_chart(fig_processed.update_layout(plot_bgcolor="white"))
    
     #   st.markdown("### Intensity distribution processed data per sample")
    #if st.session_state.dataset.preprocessed:

    



    display_matrix()
    
    # Display Missing values / Imputed values
        


    



else:
    st.info("Import Data first")