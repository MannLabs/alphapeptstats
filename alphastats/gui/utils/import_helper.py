from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import os
import io

import plotly.express as px
from streamlit.runtime.uploaded_file_manager import UploadedFile

try:
    from alphastats.DataSet import DataSet
    from alphastats.gui.utils.analysis_helper import (
        get_sample_names_from_software_file,
        _read_file_to_df,
    )
    from alphastats.gui.utils.options import SOFTWARE_OPTIONS
    from alphastats.loader.MaxQuantLoader import MaxQuantLoader, BaseLoader
    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

except ModuleNotFoundError:
    from utils.analysis_helper import (
        get_sample_names_from_software_file,
        read_uploaded_file_into_df,
    )
    from utils.options import SOFTWARE_OPTIONS
    from alphastats import MaxQuantLoader, BaseLoader
    from alphastats import DataSet
    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx


def load_options():
    # TODO move import to top
    from alphastats.gui.utils.options import plotting_options, statistic_options

    st.session_state["plotting_options"] = plotting_options(st.session_state)
    st.session_state["statistic_options"] = statistic_options(st.session_state)


def load_proteomics_data(uploaded_file, intensity_column, index_column, software):
    """Load software file into loader object."""
    loader = SOFTWARE_OPTIONS.get(software)["loader_function"](
        uploaded_file, intensity_column, index_column
    )
    return loader


def load_softwarefile_df(software: str, softwarefile: UploadedFile) -> pd.DataFrame:
    """Load software file into pandas DataFrame.

    TODO rename: softwarefile -> data_file
    """
    softwarefile_df = _read_file_to_df(softwarefile)

    _check_softwarefile_df(softwarefile_df, software)

    st.write(
        f"File successfully uploaded. Number of rows: {softwarefile_df.shape[0]}"
        f", Number of columns: {softwarefile_df.shape[1]}."
    )

    st.write("Preview:")
    st.dataframe(softwarefile_df.head(5))

    return softwarefile_df


def show_metadata_file_uploader(loader: BaseLoader) -> Optional[pd.DataFrame]:
    """Show the 'upload metadata file' component and return the data."""
    st.write(
        "Download the template file and add additional information "
        + "to your samples as columns (e.g. 'disease group'). "
        + "Then upload the updated metadata file."
    )
    show_button_download_metadata_template_file(loader)

    metadatafile_upload = st.file_uploader(
        "Upload metadata file with information about your samples",
    )

    if metadatafile_upload is None:
        return None

    metadatafile_df = _read_file_to_df(metadatafile_upload)
    st.write(
        f"File successfully uploaded. Number of rows: {metadatafile_df.shape[0]}"
        f", Number of columns: {metadatafile_df.shape[1]}."
    )
    st.write("Preview:")
    st.dataframe(metadatafile_df.head(5))

    return metadatafile_df


def load_example_data():
    st.markdown("### Using Example Dataset")
    st.info("Example dataset and metadata loaded")
    st.write(
        """
    _Plasma proteome profiling discovers novel proteins associated with non-alcoholic fatty liver disease_

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
    _this_file = os.path.abspath(__file__)
    _this_directory = os.path.dirname(_this_file)
    _parent_directory = os.path.dirname(_this_directory)
    folder_to_load = os.path.join(_parent_directory, "sample_data")

    filepath = os.path.join(folder_to_load, "proteinGroups.txt")
    metadatapath = os.path.join(folder_to_load, "metadata.xlsx")

    loader = MaxQuantLoader(file=filepath)
    # TODO why is this done twice?
    dataset = DataSet(loader=loader, metadata_path=metadatapath, sample_column="sample")
    metadatapath = (
        os.path.join(_parent_directory, "sample_data", "metadata.xlsx")
        .replace("pages/", "")
        .replace("pages\\", "")
    )

    loader = MaxQuantLoader(file=filepath)
    dataset = DataSet(loader=loader, metadata_path=metadatapath, sample_column="sample")

    dataset.metadata = dataset.metadata[
        [
            "sample",
            "disease",
            "Drug therapy (procedure) (416608005)",
            "Lipid-lowering therapy (134350008)",
        ]
    ]
    dataset.preprocess(subset=True)
    metadata_columns = dataset.metadata.columns.to_list()
    return loader, metadata_columns, dataset


def display_loaded_dataset(dataset: DataSet) -> None:
    st.markdown(f"*Preview:* Raw data from {dataset.software}")
    st.dataframe(dataset.rawinput.head(5))

    st.markdown("*Preview:* Metadata")
    st.dataframe(dataset.metadata.head(5))

    st.markdown("*Preview:* Matrix")

    df = pd.DataFrame(
        dataset.mat.values,
        index=dataset.mat.index.to_list(),
    ).head(5)

    st.dataframe(df)


def save_plot_sampledistribution_rawdata(dataset: DataSet) -> None:
    df = dataset.rawmat
    df = df.unstack().reset_index()
    df.rename(
        columns={"level_1": dataset.sample, 0: "Intensity"},
        inplace=True,
    )
    st.session_state["distribution_plot"] = px.violin(
        df, x=dataset.sample, y="Intensity"
    )


def empty_session_state():
    """
    remove all variables to avoid conflicts
    """
    for key in st.session_state.keys():
        del st.session_state[key]
    st.empty()


def init_session_state() -> None:
    """Initialize the session state."""
    st.session_state["user_session_id"] = get_script_run_ctx().session_id

    if "gene_to_prot_id" not in st.session_state:
        st.session_state["gene_to_prot_id"] = {}

    if "organism" not in st.session_state:
        st.session_state["organism"] = 9606  # human


def _check_softwarefile_df(df: pd.DataFrame, software: str) -> None:
    """Check if the dataframe containing the software file is in right format.

    Can be fragile when different settings are used or software is updated.
    """

    if software == "MaxQuant":
        expected_columns = ["Protein IDs", "Reverse", "Potential contaminant"]
        if not set(expected_columns).issubset(set(df.columns.to_list())):
            raise ValueError(
                "This is not a valid MaxQuant file. Please check: "
                "http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable"
            )

    elif software == "AlphaPept":
        if "object" in df.iloc[:, 1:].dtypes.to_list():
            raise ValueError("This is not a valid AlphaPept file.")

    elif software == "DIANN":
        expected_columns = [
            "Protein.Group",
        ]

        if not set(expected_columns).issubset(set(df.columns.to_list())):
            raise ValueError("This is not a valid DIA-NN file.")

    elif software == "Spectronaut":
        expected_columns = [
            "PG.ProteinGroups",
        ]

        if not set(expected_columns).issubset(set(df.columns.to_list())):
            raise ValueError("This is not a valid Spectronaut file.")

    elif software == "FragPipe":
        expected_columns = ["Protein"]
        if not set(expected_columns).issubset(set(df.columns.to_list())):
            raise ValueError(
                "This is not a valid FragPipe file. Please check:"
                "https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv"
            )


def show_loader_columns_selection(
    software: str, softwarefile_df: Optional[pd.DataFrame] = None
) -> Tuple[str, str]:
    """
    select intensity and index column depending on software
    will be saved in session state
    """
    st.write("\n\n")
    st.markdown("##### 2. Select columns used for analysis")
    st.markdown("Select intensity columns for analysis")

    if software != "Other":
        intensity_column = st.selectbox(
            "Intensity Column",
            options=SOFTWARE_OPTIONS.get(software).get("intensity_column"),
        )

        st.markdown("Select index column (with ProteinGroups) for analysis")

        index_column = st.selectbox(
            "Index Column",
            options=SOFTWARE_OPTIONS.get(software).get("index_column"),
        )

    else:
        intensity_column = st.multiselect(
            "Intensity Columns",
            options=softwarefile_df.columns.to_list(),
        )  # TODO why is this a multiselect?

        st.markdown("Select index column (with ProteinGroups) for further analysis")

        index_column = st.selectbox(
            "Index Column",
            options=softwarefile_df.columns.to_list(),
        )

    return intensity_column, index_column


def show_select_sample_column_for_metadata(
    df: pd.DataFrame, software: str, loader: BaseLoader
) -> str:
    """Show the 'select sample column for metadata' component and return the value."""
    samples_proteomics_data = get_sample_names_from_software_file(loader)

    valid_sample_columns = [
        col
        for col in df.columns.to_list()
        if bool(set(samples_proteomics_data) & set(df[col].to_list()))
    ]

    if len(valid_sample_columns) == 0:
        raise ValueError(
            f"Metadata does not match Proteomics data."
            f"Information for the samples: {samples_proteomics_data} is required."
        )

    # TODO I get an ERROR: "described in Please upload proteinGroups.txt"
    st.write(
        "Select column that contains sample IDs matching the sample names described "
        + f"in {SOFTWARE_OPTIONS.get(software).get('import_file')}"
    )

    return st.selectbox("Sample Column", options=valid_sample_columns)


def show_button_download_metadata_template_file(loader: BaseLoader) -> None:
    """Show the 'download metadata template' button."""
    dataset = DataSet(loader=loader)
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        # Write each dataframe to a different worksheet.
        dataset.metadata.to_excel(writer, sheet_name="Sheet1", index=False)

    st.download_button(
        label="Download Excel template for metadata",
        data=buffer,
        file_name="metadata.xlsx",
        mime="application/vnd.ms-excel",
    )
