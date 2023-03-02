from curses import meta
import streamlit as st

import os

try:
    from alphastats.gui.utils.ui_helper import sidebar_info
    from alphastats.gui.utils.analysis_helper import *
    from alphastats.gui.utils.software_options import software_options
    from alphastats.loader.MaxQuantLoader import MaxQuantLoader
    from alphastats.DataSet import DataSet

except ModuleNotFoundError:
    from utils.ui_helper import sidebar_info
    from utils.analysis_helper import *
    from utils.software_options import software_options
    from alphastats import MaxQuantLoader
    from alphastats import DataSet



import pandas as pd
import plotly.express as px


def load_options():

    from alphastats.gui.utils.options import plotting_options, statistic_options

    st.session_state["plotting_options"] = plotting_options
    st.session_state["statistic_options"] = statistic_options


def check_software_file(df, software):
    """
    check if software files are in right format
    can be fragile when different settings are used or software is updated
    """

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
            "Protein.Group",
        ]

        if (set(expected_columns).issubset(set(df.columns.to_list()))) == False:
            st.error("This is not a valid DIA-NN file.")

    elif software == "Spectronaut":
        expected_columns = [
            "PG.ProteinGroups",
        ]

        if (set(expected_columns).issubset(set(df.columns.to_list()))) == False:
            st.error("This is not a valid Spectronaut file.")

    elif software == "FragPipe":
        expected_columns = ["Protein Probability", "Indistinguishable Proteins"]
        if (set(expected_columns).issubset(set(df.columns.to_list()))) == False:
            st.error(
                "This is not a valid FragPipe file. Please check:"
                "https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv"
            )


def print_software_import_info(software):
    import_file = software_options.get(software).get("import_file")
    string_output = f"Please upload {import_file} file from {software}."
    return string_output


def select_columns_for_loaders(software):
    """
    select intensity and index column depending on software
    will be saved in session state
    """
    st.write("\n\n")
    st.markdown("### 2. Select columns used for further analysis.")
    st.markdown("Select intensity columns for further analysis")

    st.selectbox(
        "Intensity Column",
        options=software_options.get(software).get("intensity_column"),
        key="intensity_column",
    )

    st.markdown("Select index column (with ProteinGroups) for further analysis")

    st.selectbox(
        "Index Column",
        options=software_options.get(software).get("index_column"),
        key="index_column",
    )


def load_proteomics_data(uploaded_file, intensity_column, index_column, software):
    """load software file into loader object from alphastats"""
    loader = software_options.get(software)["loader_function"](
        uploaded_file, intensity_column, index_column
    )
    return loader


def select_sample_column_metadata(df, software):
    samples_proteomics_data = get_sample_names_from_software_file()
    valid_sample_columns = []

    for col in df.columns.to_list():
        if bool(set(samples_proteomics_data) & set(df[col].to_list())):
            valid_sample_columns.append(col)

    if len(valid_sample_columns) == 0:
        st.error(
            f"Metadata does not match Proteomics data."
            f"Information for the samples: {samples_proteomics_data} is required."
        )

    st.write(
        f"Select column that contains sample IDs matching the sample names described"
        + f"in {software_options.get(software).get('import_file')}"
    )

    with st.form("sample_column"):
        st.selectbox("Sample Column", options=valid_sample_columns, key="sample_column")
        submitted = st.form_submit_button("Create DataSet")

    if submitted:
        return True


def upload_softwarefile(software):

    softwarefile = st.file_uploader(
        print_software_import_info(software=software),
        type=["csv", "tsv", "txt", "hdf"],
    )

    if softwarefile is not None:

        softwarefile_df = read_uploaded_file_into_df(softwarefile)
        # display head a protein data

        check_software_file(softwarefile_df, software)

        st.write(
            f"File successfully uploaded. Number of rows: {softwarefile_df.shape[0]}"
            f", Number of columns: {softwarefile_df.shape[1]}.\nPreview:"
        )
        st.dataframe(softwarefile_df.head(5))

        select_columns_for_loaders(software=software)

        if (
            "intensity_column" in st.session_state
            and "index_column" in st.session_state
        ):
            loader = load_proteomics_data(
                softwarefile_df,
                intensity_column=st.session_state.intensity_column,
                index_column=st.session_state.index_column,
                software=software,
            )
            st.session_state["loader"] = loader


def upload_metadatafile(software):

    st.write("\n\n")
    st.markdown("### 3. Upload corresponding metadata.")
    st.file_uploader(
        "Upload metadata file. with information about your samples",
        key="metadatafile",
    )

    if st.session_state.metadatafile is not None:

        metadatafile_df = read_uploaded_file_into_df(st.session_state.metadatafile)
        # display metadata
        st.write(
            f"File successfully uploaded. Number of rows: {metadatafile_df.shape[0]}"
            f", Number of columns: {metadatafile_df.shape[1]}. \nPreview:"
        )
        st.dataframe(metadatafile_df.head(5))
        # pick sample column

        if select_sample_column_metadata(metadatafile_df, software):
            # create dataset
            st.session_state["dataset"] = DataSet(
                loader=st.session_state.loader,
                metadata_path=metadatafile_df,
                sample_column=st.session_state.sample_column,
            )
            st.session_state["metadata_columns"] = metadatafile_df.columns.to_list()

            load_options()

            display_loaded_dataset()

    if st.button("Create a DataSet without metadata"):
        st.session_state["dataset"] = DataSet(loader=st.session_state.loader)
        st.session_state["metadata_columns"] = ["sample"]

        load_options()

        display_loaded_dataset()


def load_sample_data():
    _this_file = os.path.abspath(__file__)
    _this_directory = os.path.dirname(_this_file)
    filepath = os.path.join(_this_directory, "sample_data/proteinGroups.txt").replace("pages/","")
    metadatapath =  os.path.join(_this_directory, "sample_data/metadata.xlsx").replace("pages/","")
    
    loader = MaxQuantLoader(file=filepath)
    ds = DataSet(
        loader=loader, metadata_path=metadatapath, sample_column="sample"
    )
    
    ds.metadata = ds.metadata[
        [
            "sample",
            "disease",
            "Drug therapy (procedure) (416608005)",
            "Lipid-lowering therapy (134350008)",
        ]
    ]
    ds.preprocess(subset=True)
    st.session_state["loader"] = loader
    st.session_state["metadata_columns"] = ds.metadata.columns.to_list()
    st.session_state["dataset"] = ds

    load_options()


def import_data():

    software = st.selectbox(
        "Select your Proteomics Software",
        options=["<select>", "MaxQuant", "AlphaPept", "DIANN", "Fragpipe", "Spectronaut"],
    )

    session_state_empty = False

    if software != "<select>":
        # if
        # reset()
        upload_softwarefile(software=software)

    if "loader" in st.session_state:
        upload_metadatafile(software)


def display_loaded_dataset():

    st.info("Data was successfully imported")
    st.info("DataSet has been created")

    st.markdown(f"*Preview:* Raw data from {st.session_state.dataset.software}")
    st.dataframe(st.session_state.dataset.rawinput.head(5))

    st.markdown(f"*Preview:* Metadata")
    st.dataframe(st.session_state.dataset.metadata.head(5))

    st.markdown(f"*Preview:* Matrix")

    df = pd.DataFrame(
        st.session_state.dataset.mat.values,
        index=st.session_state.dataset.mat.index.to_list(),
    ).head(5)

    st.dataframe(df)


def save_plot_sampledistribution_rawdata():
    df = st.session_state.dataset.rawmat
    df = df.unstack().reset_index()
    df.rename(
        columns={"level_1": st.session_state.dataset.sample, 0: "Intensity"},
        inplace=True,
    )
    st.session_state["distribution_plot"] = px.violin(
        df, x=st.session_state.dataset.sample, y="Intensity"
    )


def empty_session_state():
    """
    remove all variables to avoid conflicts
    """
    for key in st.session_state.keys():
        del st.session_state[key]
    st.empty()

    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
    user_session_id = get_script_run_ctx().session_id
    st.session_state["user_session_id"] = user_session_id

sidebar_info()


if "dataset" not in st.session_state:
    st.markdown("### Import Proteomics Data")

    st.markdown(
        "Create a DataSet with the output of your proteomics software package and the corresponding metadata (optional). "
    )

    import_data()
    st.markdown("### Or Load sample Dataset")

if st.button("Load sample DataSet - PXD011839"):

    st.write(
        """

    ### Plasma proteome profiling discovers novel proteins associated with non-alcoholic fatty liver disease

    **Description**

    Non-alcoholic fatty liver disease (NAFLD) affects 25 percent of the population and can progress to cirrhosis, 
    where treatment options are limited. As the liver secrets most of the blood plasma proteins its diseases 
    should affect the plasma proteome. Plasma proteome profiling on 48 patients with cirrhosis or NAFLD with 
    normal glucose tolerance or diabetes, revealed 8 significantly changing (ALDOB, APOM, LGALS3BP, PIGR, VTN, 
    IGHD, FCGBP and AFM), two of which are already linked to liver disease. Polymeric immunoglobulin receptor (PIGR) 
    was significantly elevated in both cohorts with a 2.7-fold expression change in NAFLD and 4-fold change in 
    cirrhosis and was further validated in mouse models. Furthermore, a global correlation map of clinical and 
    proteomic data strongly associated DPP4, ANPEP, TGFBI, PIGR, and APOE to NAFLD and cirrhosis. DPP4 is a known 
    drug target in diabetes. ANPEP and TGFBI are of interest because of their potential role in extracellular matrix 
    remodeling in fibrosis.

    **Publication**

    Niu L, Geyer PE, Wewer Albrechtsen NJ, Gluud LL, Santos A, Doll S, Treit PV, Holst JJ, Knop FK, Vilsbøll T, Junker A, 
    Sachs S, Stemmer K, Müller TD, Tschöp MH, Hofmann SM, Mann M, Plasma proteome profiling discovers novel proteins 
    associated with non-alcoholic fatty liver disease. Mol Syst Biol, 15(3):e8793(2019)
    """
    )

    load_sample_data()

if "dataset" in st.session_state:
    st.info("DataSet has been imported")

    if "distribution_plot" not in st.session_state:
        save_plot_sampledistribution_rawdata()

    if st.button("New Session: Import new dataset"):

        empty_session_state()

        import_data()

    if "dataset" in st.session_state:

        display_loaded_dataset()
