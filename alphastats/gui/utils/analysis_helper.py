import pandas as pd
import logging
import streamlit as st
from datetime import datetime
from alphastats.plots.VolcanoPlot import VolcanoPlot


def check_if_options_are_loaded(f):
    # decorator to check for missing values
    def inner(*args, **kwargs):
        if hasattr(st.session_state, "plotting_options") is False:
            load_options()

        return f(*args, **kwargs)

    return inner


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
    unique_values = st.session_state.dataset.metadata[column].unique().tolist()
    return unique_values


def st_general(method_dict):

    chosen_parameter_dict = {}

    if "settings" in list(method_dict.keys()):

        settings_dict = method_dict.get("settings")

        for parameter in settings_dict:

            parameter_dict = settings_dict[parameter]

            if "options" in parameter_dict.keys():
                chosen_parameter = st.selectbox(
                    parameter_dict.get("label"), options=parameter_dict.get("options")
                )
            else:
                chosen_parameter = st.checkbox(parameter_dict.get("label"))

            chosen_parameter_dict[parameter] = chosen_parameter

    submitted = st.button("Submit")

    if submitted:
        with st.spinner("Calculating..."):
            return method_dict["function"](**chosen_parameter_dict)


@st.cache(persist=True, max_entries=10, allow_output_mutation=True)
def gui_volcano_plot_differential_expression_analysis(chosen_parameter_dict):
    """
    initalize volcano plot object with differential expression analysis results
    """
    volcano_plot = VolcanoPlot(
        dataset=st.session_state.dataset, 
        **chosen_parameter_dict, 
        plot = False
    )
    volcano_plot._perform_differential_expression_analysis()
    volcano_plot._add_hover_data_columns()
    return volcano_plot

def gui_volcano_plot():
    """
    Draw Volcano Plot using the VolcanoPlot class
    """
    chosen_parameter_dict = helper_compare_two_groups()
    method = st.selectbox(
        "Differential Analysis using:",
        options=["ttest", "anova", "wald", "sam"],
    )
    chosen_parameter_dict.update({"method": method})

    # TODO streamlit doesnt allow nested columns check for updates
    
    labels = st.checkbox("Add label")

    draw_line = st.checkbox("Draw line")

    alpha = st.number_input(
            label="alpha", min_value=0.001, max_value=0.050, value=0.050
        )

    min_fc = st.select_slider("Foldchange cutoff", range(0, 3), value=1)
 
    plotting_parameter_dict = {
            "labels": labels,
            "draw_line": draw_line,
            "alpha": alpha,
            "min_fc": min_fc,
        }

    if method == "sam":
        perm = st.number_input(
            label="Number of Permutations", min_value=1, max_value=1000, value=10
        )
        fdr = st.number_input(
            label="FDR cut off", min_value=0.005, max_value=0.1, value=0.050
        )
        chosen_parameter_dict.update({"perm": perm, "fdr": fdr})

   
    submitted = st.button("Submit")

    if submitted:
        volcano_plot = gui_volcano_plot_differential_expression_analysis(chosen_parameter_dict)
        volcano_plot._update(plotting_parameter_dict)
        volcano_plot._annotate_result_df()
        volcano_plot._plot()
        return volcano_plot.plot


def get_analysis_options_from_dict(method, options_dict):
    """
    extract plotting options from dict amd display as selectbox or 
    give selceted options to plotting function
    """

    method_dict = options_dict.get(method)

    if method == "t-SNE Plot":
        return st_tsne_options(method_dict)

    elif method == "Differential Expression Analysis - T-test":
        return st_calculate_ttest(method=method, options_dict=options_dict)

    elif method == "Differential Expression Analysis - Wald-test":
        return st_calculate_waldtest(method=method, options_dict=options_dict)

    elif method == "Volcano Plot":
        return gui_volcano_plot()

    elif method == "PCA Plot":
        return st_plot_pca(method_dict)

    elif method == "UMAP Plot":
        return st_plot_umap(method_dict)

    elif "settings" not in method_dict.keys():

        if st.session_state.dataset.mat.isna().values.any() == True:
            st.error(
                "Data contains missing values impute your data before plotting (Preprocessing - Imputation)."
            )
            return

        chosen_parameter_dict = {}
        return method_dict["function"](**chosen_parameter_dict)

    else:
        return st_general(method_dict=method_dict)


def st_plot_pca(method_dict):
    chosen_parameter_dict = helper_plot_dimensionality_reduction(
        method_dict=method_dict
    )

    submitted = st.button("Submit")

    if submitted:
        with st.spinner("Calculating..."):
            return method_dict["function"](**chosen_parameter_dict)


def st_plot_umap(method_dict):
    chosen_parameter_dict = helper_plot_dimensionality_reduction(
        method_dict=method_dict
    )

    submitted = st.button("Submit")

    if submitted:
        with st.spinner("Calculating..."):
            return method_dict["function"](**chosen_parameter_dict)


def st_calculate_ttest(method, options_dict):
    """
    perform ttest in streamlit
    """
    chosen_parameter_dict = helper_compare_two_groups()
    chosen_parameter_dict.update({"method": "ttest"})

    submitted = st.button("Submit")

    if submitted:
        with st.spinner("Calculating..."):
            return options_dict.get(method)["function"](**chosen_parameter_dict)


def st_calculate_waldtest(method, options_dict):
    chosen_parameter_dict = helper_compare_two_groups()
    chosen_parameter_dict.update({"method": "wald"})

    submitted = st.button("Submit")

    if submitted:
        with st.spinner("Calculating..."):
            return options_dict.get(method)["function"](**chosen_parameter_dict)


def helper_plot_dimensionality_reduction(method_dict):
    group = st.selectbox(
        method_dict["settings"]["group"].get("label"),
        options=method_dict["settings"]["group"].get("options"),
    )

    circle = False

    if group is not None:

        circle = st.checkbox("circle")

    chosen_parameter_dict = {
        "circle": circle,
        "group": group,
    }
    return chosen_parameter_dict


def helper_compare_two_groups():
    """
    Helper function to compare two groups for example
    Volcano Plot, Differetial Expression Analysis and t-test
    selectbox based on selected column
    """

    chosen_parameter_dict = {}
    group = st.selectbox(
        "Grouping variable",
        options=["< None >"] + st.session_state.dataset.metadata.columns.to_list(),
    )

    if group != "< None >":

        #col1, col2 = st.columns(2)

        unique_values = get_unique_values_from_column(group)

        #with col1:

        group1 = st.selectbox("Group 1", options=unique_values)

        #with col2:

        group2 = st.selectbox("Group 2", options=list(reversed(unique_values)))

        chosen_parameter_dict.update(
            {"column": group, "group1": group1, "group2": group2}
        )

        if group1 == group2:
            st.error(
                "Group 1 and Group 2 can not be the same please select different group."
            )

    else:

        #col1, col2 = st.columns(2)

        #with col1:

        group1 = st.multiselect(
                "Group 1 samples:",
                options=st.session_state.dataset.metadata["sample"].to_list(),
            )

        #with col2:

        group2 = st.multiselect(
                "Group 2 samples:",
                options=list(
                    reversed(st.session_state.dataset.metadata["sample"].to_list())
                ),
            )

        intersection_list = list(set(group1).intersection(set(group2)))

        if len(intersection_list) > 0:
            st.warning(
                "Group 1 and Group 2 contain same samples: " + str(intersection_list)
            )

        chosen_parameter_dict.update({"group1": group1, "group2": group2})

    return chosen_parameter_dict


def get_sample_names_from_software_file():
    """
    extract sample names from software
    """
    regex_find_intensity_columns = st.session_state.loader.intensity_column.replace(
        "[sample]", ".*"
    )

    df = st.session_state.loader.rawinput
    df = df.set_index(st.session_state.loader.index_column)
    df = df.filter(regex=(regex_find_intensity_columns), axis=1)
    # remove Intensity so only sample names remain
    substring_to_remove = regex_find_intensity_columns.replace(".*", "")
    df.columns = df.columns.str.replace(substring_to_remove, "")
    return df.columns.to_list()


def get_analysis(method, options_dict):

    if method in options_dict.keys():
        obj = get_analysis_options_from_dict(method, options_dict=options_dict)
        return obj


def st_tsne_options(method_dict):
    chosen_parameter_dict = helper_plot_dimensionality_reduction(
        method_dict=method_dict
    )

    n_iter = st.select_slider(
        "Maximum number of iterations for the optimization",
        range(250, 2001),
        value=1000,
    )
    perplexity = st.select_slider("Perplexity", range(5, 51), value=30)

    submitted = st.button("Submit")
    chosen_parameter_dict.update(
        {
            "n_iter": n_iter,
            "perplexity": perplexity,
        }
    )

    if submitted:
        with st.spinner("Calculating..."):
            return method_dict["function"](**chosen_parameter_dict)


def load_options():

    from alphastats.gui.utils.options import plotting_options, statistic_options

    st.session_state["plotting_options"] = plotting_options
    st.session_state["statistic_options"] = statistic_options
