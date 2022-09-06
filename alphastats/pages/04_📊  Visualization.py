import streamlit as st
from alphastats.AlphaStats import sidebar_info
from alphastats.ui_utils.utils import (
    get_unique_values_from_column
)

def load_plotting_options():
    plotting_options = {
            "Sampledistribution": {
                "settings": {
                "method": {
                    "options": ["violin", "box"], 
                    "label": "Plot layout"},
                "color": {
                    "options": [None] + st.session_state.metadata_columns,
                    "label": "Color according to"}
                },
                "plotting_function": st.session_state.dataset.plot_sampledistribution,
            },
            "Intensity": {
                "settings": {"id": {
                    "options": st.session_state.metadata_columns,
                    "label": "ProteinID/ProteinGroup",
                },
                "method": {
                    "options": ["violin", "box", "scatter"],
                    "label": "Plot layout",
                },
                "color": {
                    "options": [None] + st.session_state.metadata_columns,
                    "label": "Color according to",
                }},
                "plotting_function": st.session_state.dataset.plot_sampledistribution,
            },
            "PCA": {
                "settings": {
                    "group": {
                        "options": [None] + st.session_state.metadata_columns,
                        "label":  "Color according to"
                    },
                    "circle": {"label": "Circle"}
                },
                "plotting_function": st.session_state.dataset.plot_pca
            },
            "t-SNE": {
                "settings": {
                    "group": {
                        "options": [None] + st.session_state.metadata_columns,
                        "label":  "Color according to"
                    },
                    "Circle": {"label": "Circle"}
                },
                "plotting_function": st.session_state.dataset.plot_tsne
            },
            "Clustermap": {
                "plotting_function":st.session_state.dataset.plot_clustermap
            },
            "Dendrogram": {
                "plotting_function": st.session_state.dataset.plot_dendogram
            }
        }
    return plotting_options



def dataset_info_as_markdown():
    
    if "dataset" not in st.session_state:
        return ""
    
    preprocessing_dict = st.session_state.dataset.preprocessing_info
    markdown_string = ""
    for key, values in preprocessing_dict.items():
        markdown_string += "**" + key + ":** " + str(values) + " \n\n "
    
    return markdown_string


def display_plotly_figure(plot):
    st.plotly_chart(plot)

def get_plotting_options_from_dict(plot):
    #extract plotting options from dict amd display as selectbox or checkbox
    # give selceted options to plotting function
    plotting_options = load_plotting_options()
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
    
    return plot_dict["plotting_function"](**chosen_parameter_dict)

def plot_volcano():
    group = st.selectbox(
                "Grouping variable", 
                options=[None] + st.session_state.dataset.metadata.columns.to_list()
            )
    if group is not None:
        unique_values = get_unique_values_from_column(group)
        group1 = st.selectbox("Group 1", 
        options=["<select>"] + unique_values)
        group2 = st.selectbox("Group 2",
         options=["<select>"] + unique_values)
        method = st.selectbox(
                    "Differential Analysis using:",
                    options=["<select>", "anova", "wald", "ttest"],
                )
        if (
            group1 != "<select>"
                    and group2 != "<select>"
                    and method != "<select>"
                ):
                st.session_state.dataset.plot_volcano(
                        column=group, group1=group1, group2=group2, method=method
                    )


def choose_plotoptions(plot):
    if plot in load_plotting_options().keys():
        plotly_plot = get_plotting_options_from_dict(plot)
        display_plotly_figure(plotly_plot)

    elif plot == "Volcano":
        plotly_plot = plot_volcano()
        display_plotly_figure(plotly_plot)


def add_plot_widget():
    if "n_rows" not in st.session_state:
        st.session_state.n_rows = 1

    add = st.button(label="add")

    if add:
        st.session_state.n_rows += 1
        st.experimental_rerun()

    for i in range(st.session_state.n_rows):
        # add text inputs here
        plot = st.selectbox(
                "Plot",
                options=[
                    "PCA",
                    "t-SNE",
                    "Sampledistribution",
                    "Intensity",
                    "Volcano",
                    "Clustermap",
                    "Dendrogram",
                ], key = str(i)
            )  # Pass index as ke
        choose_plotoptions(plot)




st.markdown("### Visualization")
st.sidebar.markdown(dataset_info_as_markdown())
sidebar_info()

if "dataset" in st.session_state:
    if "plot_list" in st.session_state:
        for p in st.session_state.plot_list:
            display_plotly_figure(p)

    if "n_rows" not in st.session_state:
         st.session_state.n_rows = 1
    add = st.button(label="add")

    if add:
        st.session_state.n_rows += 1
        st.experimental_rerun()

    for i in range(st.session_state.n_rows):
        # add text inputs here
        plot = st.selectbox(
                "Plot",
                options=[
                    "PCA",
                    "t-SNE",
                    "Sampledistribution",
                    "Intensity",
                    "Volcano",
                    "Clustermap",
                    "Dendrogram",
                ], key = str(i)
            )  # Pass index as ke
        choose_plotoptions(plot)

else:
    st.info("Import Data first")

def show_dataset_overview():
    st.session_state.dataset.print
    

