import streamlit as st
import pandas as pd

def sidebar_info():
    display_sidebar_html_table()
    st.sidebar.markdown("\n\n")
    st.sidebar.info("[AlphaStats on GitHub](https://github.com/MannLabs/alphastats)")
    st.sidebar.markdown(
        """ <head><style type ='text/css' > 
    .footer{ position: fixed;     
        text-align: left;    
        bottom: 14px; 
        width: 100%;
    }  
    </style>
    </head>
    <body>
        <div class="footer">Mann Group, 2022</div>
    </body>""",
        unsafe_allow_html=True,
    )


def display_sidebar_html_table():
    
    if "dataset" not in st.session_state:
        return 

    preprocessing_dict = st.session_state.dataset.preprocessing_info
    
    html_string = ("<style>.mytable th, td{ font-size:10px;color:#8C878D; border-color:#96D4D4;}</style>" 
        "<table class='mytable'>")
    
    for key, values in preprocessing_dict.items():
        html_string += "<tr><td>" + key + "</td><td>" + str(values) + "</td>" + "</tr>"
    
    html_string += "</table>"
    st.sidebar.markdown(html_string, unsafe_allow_html=True)