from tracemalloc import Statistic
import streamlit as st
from alphastats.AlphaStats import sidebar_info


def load_statistic_options():
    statistic_options = {
            "Differential Expression Analysis - Wald-test": {
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
            "Tukey - Test": {
                "settings": {"protein_id": {
                    "options": st.session_state.mat.columns.to_list(),
                    "label": "ProteinID/ProteinGroup",
                },
                "group": {
                    "options": st.session_state.metadata_columns,
                    "label": "A metadata variable to calculate pairwise tukey",
                }},
                "plotting_function": st.session_state.dataset.calculate_tukey,
            },
            "t-test": {
                "settings":{ 

                }
            },
            "ANOVA": {
                "settings": {
                    "column": {
                        "options": st.session_state.metadata_columns,
                        "label":  "A variable from the metadata to calculate ANOVA",
                    },
                    "protein_ids": {
                        "options": ["all"] + st.session_state.dataset.mat.columns.to_list(),
                        "label":  "All ProteinIDs/or specific ProteinID to perform ANOVA"
                    },
                    "tukey": {"label": "Follow-up Tukey"}
                },
                "plotting_function": st.session_state.dataset.plot_pca
            },
            "ANCOVA": {
                "settings": {
                    "protein_id": {
                        "options": [None] + st.session_state.dataset.mat.columns.to_list(),
                        "label":  "Color according to"
                    },
                    "covar": {
                        "options": st.session_state.metadata_columns,
                        "label": "Name(s) of column(s) in metadata with the covariate."
                    },
                    "between": {
                        "options": st.session_state.metadata_columns,
                        "label": "Name of the column in the metadata with the between factor."
                    }
                },
                "plotting_function": st.session_state.dataset.plot_tsne
            }
        }
    return statistic_options

def get_unique_values_from_column(column):
    unique_values = st.session_state.dataset.metadata[column].unique().to_list()
    return unique_values

def dataset_info_as_markdown():
    
    if "dataset" not in st.session_state:
        return ""
    
    preprocessing_dict = st.session_state.dataset.preprocessing_info
    markdown_string = ""
    for key, values in preprocessing_dict.items():
        markdown_string += "**" + key + ":** " + str(values) + " \n\n "
    return markdown_string



def get_statistic_options_from_dict(plot):
    #extract plotting options from dict amd display as selectbox or checkbox
    # give selceted options to plotting function
    statistic_options = load_statistic_options()
    plot_dict = statistic_options.get(plot)
    
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

def calculate_ttest():
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

        if (
            group1 != "<select>"
                    and group2 != "<select>"
                ):
                st.session_state.dataset.calculate_ttest_fc(
                        column=group, group1=group1, group2=group2
                    )


def choose_statistic_options(statistic):

    if statistic in load_statistic_options().keys():
        statisitc_df = get_statistic_options_from_dict(statistic)
        display_df(statisitc_df)

    elif statistic == "ttest":
        plotly_plot = calculate_ttest()



st.markdown("### Statistical Analysis")
st.sidebar.markdown(dataset_info_as_markdown())
sidebar_info()

if "dataset" in st.session_state:
    if "n_rows" not in st.session_state:
         st.session_state.n_rows = 1
    add = st.button(label="add")

    if add:
        st.session_state.n_rows += 1
        st.experimental_rerun()

    for i in range(st.session_state.n_rows):
        # add text inputs here
        statistic = st.selectbox(
                "Statistical Analysis",
                options = load_statistic_options().keys() + ["ttest"], 
                key = "statistic" + str(i)
            )  # Pass index as ke
        choose_statistic_options(statistic)

else:
    st.info("Import Data first")

def show_dataset_overview():
    st.session_state.dataset.print
    

