import streamlit as st
from gui.main import user_interface
from streamlit_multipage import MultiPage


"""# AlphaStats\n  
An open-source Python package for the analysis of mass spectrometry based proteomics data 
from the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/).

## How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/alphastats/stargazers) to boost 
our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/alphastats/issues) 
or clone the repository and create a [pull request](https://github.com/MannLabs/alphastats/pulls) with a new branch. For an even more 
interactive participation, check out the [discussions](https://github.com/MannLabs/alphastats/discussions) and the 
[the Contributors License Agreement](misc/CLA.md)."""



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



sidebar_info()