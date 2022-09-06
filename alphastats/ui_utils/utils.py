
import pandas as pd
import logging
import streamlit as st

def read_uploaded_file_into_df(file):
    filename = file.name
    if filename.endswith(".xlsx"):
        df = pd.read_excel(file)
    elif filename.endswith(".txt") or filename.endswith(".tsv"):
        df = pd.read_csv(file, delimiter="\t")
    elif filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = None
        logging.warning(
                "WARNING: File could not be read. \nFile has to be a .xslx, .tsv, .csv or .txt file"
            )
        return
    return df

def get_unique_values_from_column(column):
    unique_values = st.session_state.dataset.metadata[column].unique().to_list()
    return unique_values

def load_analysis_options():
    return

def get_analysis_options_from_dict(plot):
    #extract plotting options from dict amd display as selectbox or checkbox
    # give selceted options to plotting function
    plotting_options = load_analysis_options()
    plot_dict = plotting_options.get(plot)
    
    if "settings" not in plot_dict.keys():
        return plot_dict["plotting_function"]
   
    settings_dict =plot_dict.get("settings")
    chosen_parameter_dict = {}
    
    for parameter in settings_dict:
        parameter_dict = settings_dict[parameter]
        if "options" in parameter_dict.keys():
            chosen_parameter = st.selectbox(
                parameter_dict.get("label"), 
                options=parameter_dict.get("options"),
                key = plot + parameter 
                )
        else:
            chosen_parameter = st.checkbox(parameter_dict.get("label"))
        chosen_parameter_dict[parameter] = chosen_parameter
    
    return plot_dict["function"](**chosen_parameter_dict)


def sidebar_info():
    st.sidebar.info("[AlphaStats on GitHub](https://github.com/MannLabs/alphastats)")
    st.sidebar.markdown(""" <head><style type ='text/css' > 
    .footer{ position: fixed;     
        text-align: left;    
        bottom: 20px; 
        width: 100%;
    }  
    </style>
    </head>
    <body>
        <div class="footer">Mann Group, 2022</div>
    </body>""", unsafe_allow_html=True)