import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from alphastats.dataset.dataset import DataSet
from alphastats.dataset.keys import Cols
from alphastats.gui.utils.options import SOFTWARE_OPTIONS
from alphastats.loader.base_loader import BaseLoader
from alphastats.loader.maxquant_loader import MaxQuantLoader


def load_proteomics_data(uploaded_file, intensity_column, index_column, software):
    """Load software file into loader object."""
    loader = SOFTWARE_OPTIONS.get(software)["loader_function"](
        uploaded_file, intensity_column, index_column
    )
    return loader


def uploaded_file_to_df(
    uploaded_file: UploadedFile, software: str = None
) -> pd.DataFrame:
    """Load uploaded file into pandas DataFrame. If `software` is given, do some additional checks."""
    df = _read_file_to_df(uploaded_file)

    if software is not None:
        # assuming it's a softwarefile
        _check_softwarefile_df(df, software)

    st.write(
        f"File successfully uploaded. Number of rows: {df.shape[0]}"
        f", Number of columns: {df.shape[1]}."
    )

    st.write("Preview:")
    st.dataframe(df.head(5))

    return df


def _read_file_to_df(file: UploadedFile, decimal: str = ".") -> Optional[pd.DataFrame]:
    """Read file to DataFrame based on file extension.

    TODO rename: softwarefile -> data_file
    """

    extension = Path(file.name).suffix

    if extension == ".xlsx":
        return pd.read_excel(file)

    elif extension in [".txt", ".tsv"]:
        return pd.read_csv(file, delimiter="\t", decimal=decimal)

    elif extension == ".csv":
        return pd.read_csv(file, decimal=decimal)

    raise ValueError(
        f"Unknown file type '{extension}'. \nSupported types: .xslx, .tsv, .csv or .txt file"
    )


def load_example_data():
    st.markdown("### Using Example Dataset")
    st.toast("Example dataset loaded", icon="✅")
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
    folder_to_load = os.path.join(_parent_directory, "example_data")

    filepath = os.path.join(folder_to_load, "proteinGroups.txt")
    metadatapath = (
        os.path.join(_parent_directory, "example_data", "metadata.xlsx")
        .replace("pages_/", "")
        .replace("pages_\\", "")
    )

    loader = MaxQuantLoader(file=filepath)
    dataset = DataSet(
        loader=loader, metadata_path_or_df=metadatapath, sample_column="sample"
    )

    return dataset


def _check_softwarefile_df(df: pd.DataFrame, software: str) -> None:
    """Check if the dataframe containing the software file is in right format.

    Can be fragile when different settings are used or software is updated.
    """
    # TODO this needs to go to the loader

    if software == "MaxQuant":
        expected_columns = [
            "Protein IDs",
            "Reverse",
            "Potential contaminant",
            "Only identified by site",
        ]
        if not set(expected_columns).issubset(set(df.columns.to_list())):
            raise ValueError(
                "This is not a valid MaxQuant file. Please check: "
                "http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable"
                f" We check for these columns: {expected_columns}"
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


def get_sample_names_from_software_file(loader: BaseLoader) -> List[str]:
    """
    extract sample names from software
    """
    if isinstance(
        loader.intensity_column, str
    ):  # TODO duplicated logic in MaxQuantLoader
        regex_find_intensity_columns = loader.intensity_column.replace("[sample]", ".*")
        df = loader.rawinput
        df = df.set_index(loader.index_column)
        df = df.filter(regex=(regex_find_intensity_columns), axis=1)
        # remove Intensity so only sample names remain
        substring_to_remove = regex_find_intensity_columns.replace(".*", "")
        df.columns = df.columns.str.replace(substring_to_remove, "")
        sample_names = df.columns.to_list()

    else:
        sample_names = loader.intensity_column

    return sample_names


def show_button_download_metadata_template_file(loader: BaseLoader) -> None:
    """Show the 'download metadata template' button."""
    dataset = DataSet(loader=loader)
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        # map the internal column name to something else to avoid collisions
        dataset.metadata.rename(columns={Cols.SAMPLE: "sample"}).to_excel(
            writer, sheet_name="Sheet1", index=False
        )

    st.download_button(
        label="Download Excel template for metadata",
        data=buffer,
        file_name="metadata.xlsx",
        mime="application/vnd.ms-excel",
    )
