from tkinter import E
import pandas as pd
import logging
import streamlit as st
from datetime import datetime


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


def check_software_file(df):
    # check if software files are in right format
    # can be fragile when different settings are used or software is updated
    software = st.session_state.software

    if software == "MaxQuant":
        expected_columns = ["Protein IDs", "Reverse", "Potential contaminant"]
        if (set(expected_columns).issubset(set(df.columns.to_list()))) == False:
            st.error(
                "This is not a valid MaxQuant file. Please check:"
                "http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable"
            )

    elif software == "AlphaPept":
        if "object" in df.iloc[:, 1:].dtypes.to_list():
            st.error("This is not a valid AlphaPept file.")

    elif software == "DIANN":
        expected_columns = [
            "PG.Q.value",
            "Global.PG.Q.value",
            "PTM.Q.value",
            "PTM.Site.Confidence",
            "PG.Quantity",
            "Protein.Group",
            "Protein.Ids",
            "Protein.Names",
            "Genes",
            "First.Protein.Description",
            "contamination_library",
        ]
        if (set(expected_columns).issubset(set(df.columns.to_list()))) == False:
            st.error("This is not a valid DIA-NN file.")

    elif software == "FragPipe":
        expected_columns = ["Protein Probability", "Indistinguishable Proteins"]
        if (set(expected_columns).issubset(set(df.columns.to_list()))) == False:
            st.error(
                "This is not a valid FragPipe file. Please check:"
                "https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv"
            )


def get_unique_values_from_column(column):
    unique_values = st.session_state.dataset.metadata[column].unique().tolist()
    return unique_values


def get_analysis_options_from_dict(method, options_dict):
    # extract plotting options from dict amd display as selectbox or checkbox
    # give selceted options to plotting function
    method_dict = options_dict.get(method)

    if method == "t-SNE":
        get_tsne_options(method_dict)

    if "settings" not in method_dict.keys():

        if "between_two_groups" in method_dict.keys():
            return helper_compare_two_groups(method=method, options_dict=options_dict)

        else:
            if st.session_state.dataset.mat.isna().values.any() == True:
                st.error(
                    "Data contains missing values impute your data before plotting (Preprocessing - Imputation)."
                )
                return
            return method_dict["function"]()

    settings_dict = method_dict.get("settings")
    chosen_parameter_dict = {}

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


def helper_compare_two_groups(method, options_dict):
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

        col1, col2 = st.columns(2)

        unique_values = get_unique_values_from_column(group)

        with col1:

            group1 = st.selectbox("Group 1", options=unique_values)

        with col2:

            group2 = st.selectbox("Group 2", options=unique_values)

        chosen_parameter_dict.update(
            {"column": group, "group1": group1, "group2": group2}
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            group1 = st.multiselect(
                "Group 1 samples:",
                options=st.session_state.dataset.metadata["sample"].to_list(),
            )

        with col2:

            group2 = st.multiselect(
                "Group 2 samples:",
                options=st.session_state.dataset.metadata["sample"].to_list(),
            )

        chosen_parameter_dict.update({"group1": group1, "group2": group2})

    if method == "Volcano":
        analysis_method = st.selectbox(
            "Differential Analysis using:", options=["anova", "wald", "ttest"],
        )
        chosen_parameter_dict.update({"method": analysis_method})

    elif method == "Differential Expression Analysis - T-test":
        chosen_parameter_dict.update({"method": "ttest"})

    elif method == "Differential Expression Analysis - Wald-test":
        chosen_parameter_dict.update({"method": "wald"})

    submitted = st.button("Submit")

    if submitted:
        with st.spinner("Calculating..."):
            return options_dict.get(method)["function"](**chosen_parameter_dict)


def get_sample_names_from_software_file():
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


def get_tsne_options(method_dict):

    group = st.selectbox(
        method_dict["settings"]["group"].get("label"),
        options=method_dict["settings"]["group"].get("options"),
        key=datetime.now().strftime("%H:%M:%S"),
    )

    circle = st.checkbox("circle", key=datetime.now().strftime("%H:%M:%S"))

    n_iter = st.select_slider(
        "Maximum number of iterations for the optimization",
        range(250, 2001),
        value=1000,
    )
    perplexity = st.select_slider("Perplexity", range(5, 51), value=30)

    submitted = st.button("Submit", key=datetime.now().strftime("%H:%M:%S"))
    chosen_parameter_dict = {
        "circle": circle,
        "n_iter": n_iter,
        "perplexity": perplexity,
        "group": group,
    }
    if submitted:
        with st.spinner("Calculating..."):
            return method_dict["function"](**chosen_parameter_dict)
