import streamlit as st


def sidebar_info():
    st.sidebar.markdown(dataset_info_as_markdown())
    st.sidebar.info("[AlphaStats on GitHub](https://github.com/MannLabs/alphastats)")
    st.sidebar.markdown(
        """ <head><style type ='text/css' > 
    .footer{ position: fixed;     
        text-align: left;    
        bottom: 20px; 
        width: 100%;
    }  
    </style>
    </head>
    <body>
        <div class="footer">Mann Group, 2022</div>
    </body>""",
        unsafe_allow_html=True,
    )


def dataset_info_as_markdown():

    if "dataset" not in st.session_state:
        return ""

    preprocessing_dict = st.session_state.dataset.preprocessing_info
    markdown_string = "**My DataSet overview:**\n\n"
    for key, values in preprocessing_dict.items():
        markdown_string += "" + key + ": " + str(values) + " \n\n "

    return markdown_string
