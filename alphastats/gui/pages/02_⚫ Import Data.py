import streamlit as st

from alphastats.DataSet import DataSet
from alphastats.gui.utils.ui_helper import sidebar_info
from alphastats.gui.utils.analysis_helper import (
    read_uploaded_file_into_df, 
check_software_file,
get_sample_names_from_software_file)
from alphastats.gui.utils.software_options import software_options


def print_software_import_info():
    import_file = software_options.get(st.session_state.software).get("import_file")
    string_output = (
        f"Please upload {import_file} file from {st.session_state.software}."
    )
    return string_output


def select_columns_for_loaders():
    """
    select intensity and index column depending on software
    will be saved in session state
    """
    st.write("\n\n")
    st.markdown("### 2. Select columns used for further analysis.")
    st.markdown("Select intensity columns for further analysis")

    st.selectbox(
        "Intensity Column",
        options=software_options.get(st.session_state.software).get("intensity_column"),
        key="intensity_column",
    )

    st.markdown("Select index column (with ProteinGroups) for further analysis")

    st.selectbox(
        "Index Column",
        options=software_options.get(st.session_state.software).get("index_column"),
        key="index_column",
    )


def load_proteomics_data(uploaded_file, intensity_column, index_column):
    """load software file into loader object from alphastats
    """
    loader = software_options.get(st.session_state.software)["loader_function"](
        uploaded_file, intensity_column, index_column
    )
    return loader


def select_sample_column_metadata(df):
    samples_proteomics_data = get_sample_names_from_software_file()
    valid_sample_columns = []

    for col in df.columns.to_list():
        if bool(set(samples_proteomics_data) & set(df[col].to_list())):
            valid_sample_columns.append(col)
    
    if len(valid_sample_columns):
        st.error(f"Metadata does not match Proteomics data." 
        f"Information for the samples: {samples_proteomics_data} is required.")


    st.write(
        f"Select column that contains sample IDs matching the sample names described"
        + f"in {software_options.get(st.session_state.software).get('import_file')}"
    )

    with st.form("sample_column"):
        st.selectbox("Sample Column", options=valid_sample_columns, key="sample_column")
        submitted = st.form_submit_button("Create DataSet")

    if submitted:
        return True


def display_file(df):
    st.dataframe(df.head(5))

def upload_softwarefile():

    st.file_uploader(print_software_import_info(), key="softwarefile")

    if st.session_state.softwarefile is not None:

        softwarefile_df = read_uploaded_file_into_df(st.session_state.softwarefile)
        # display head a protein data
        check_software_file(softwarefile_df)
        st.write(
            f"File successfully uploaded. Number of rows: {softwarefile_df.shape[0]} , Number of columns: {softwarefile_df.shape[1]}."
        )
        display_file(softwarefile_df)
        select_columns_for_loaders()

        if (
            "intensity_column" in st.session_state
            and "index_column" in st.session_state
        ):
            loader = load_proteomics_data(
                softwarefile_df,
                intensity_column=st.session_state.intensity_column,
                index_column=st.session_state.index_column,
            )
            st.session_state["loader"] = loader


def upload_metadatafile():

    st.write("\n\n")
    st.markdown("### 3. Upload corresponding metadata.")
    st.file_uploader(
        "Upload metadata file. with information about your samples", key="metadatafile",
    )

    if st.session_state.metadatafile is not None:

        metadatafile_df = read_uploaded_file_into_df(st.session_state.metadatafile)
        # display metadata
        st.write(
            f"File successfully uploaded. Number of rows: {metadatafile_df.shape[0]} , Number of columns: {metadatafile_df.shape[1]}."
        )
        display_file(metadatafile_df)
        # pick sample column

        if select_sample_column_metadata(metadatafile_df):
            # create dataset
            st.session_state["dataset"] = DataSet(
                loader=st.session_state.loader,
                metadata_path=metadatafile_df,
                sample_column=st.session_state.sample_column,
            )
            st.session_state["metadata_columns"] = metadatafile_df.columns.to_list()

    if st.button("Create a DataSet without metadata"):
        st.session_state["dataset"] = DataSet(loader=st.session_state.loader)


def import_data():

    st.markdown("### 1. Import Proteomics Data")

    st.selectbox(
        "Select your Proteomics Software",
        options=["<select>", "MaxQuant", "AlphaPept", "DIANN", "Fragpipe"],
        key="software",
    )

    if st.session_state.software != "<select>":
        upload_softwarefile()

    if "loader" in st.session_state:
        upload_metadatafile()


def display_loaded_dataset():
    
    st.info("Data was successfully imported")
    st.info("DataSet has been created")
    
    st.markdown(f"Raw data from {st.session_state.dataset.software}")
    display_file(st.session_state.dataset.rawdata)
    
    st.markdown(f"Metadata")
    display_file(st.session_state.dataset.metadata)
    
    st.markdown(f"Matrix")
    display_file(st.session_state.dataset.mat)


sidebar_info()

if "dataset" not in st.session_state:
    st.markdown(
        "Create a DataSet with the output of your proteomics software package and the corresponding metadata (optional). "
    )
    import_data()

elif st.button("Import new dataset"):
    del st.session_state["dataset"]
    import_data()

else:
    display_loaded_dataset()
