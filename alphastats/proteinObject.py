import re
import pandas as pd
import seaborn as sn
from data_cache import pandas_cache
import os
import warnings

"""
loading data into objet
- check for contaminations in peptides
- check for necessary columns in evidence file 
- check for outliers
"""



def filter_contaminations():
    pass



def check_experiment_type(data):
    # alphabase psm_reader
    # label free - column:  Idensity 
    # TMT columns: "Reporter intensity 0"...
    # SILAC
    pass


def save_fig():
    # check if directory for fig exists
    # else create directory to save plots (and report?)
    pass


def set_colors():
    # use seaborn
    pass


class proteinObject:
    """_summary_
    """
    # matrix with columns - proteins/peptides rows - samples
    # 
    # create object with metadata

    # give experiment type label-free, TMT, SILAC
    # detect experiment type print type
    def __init__(self, rawfile_path, metadata_path=None, intensity_column = None, software = None):
        """_summary_

        Parameters
        ----------
        evidence_file : _type_, optional
            _description_, by default None
        metadata : _type_, optional
            _description_, by default None
        """

        @pandas_cache
        def create_matrix(df, intensity_col = "LFQ intensity ", proteinID_col = "Protein IDs"):
            # LFQ intensity when label free but optional
            df = df.set_index(proteinID_col)
            df = df[df.columns[pd.Series(df.columns).str.startswith(intensity_col)]]
            # remove prefix
            df.columns = df.columns.str.lstrip(intensity_col)
            return df

        @pandas_cache
        def load_rawdata(file_path):
            df = pd.read_csv(file_path, delimiter = "\t")
            return df

        def filter_rawdata():
            pass

        @pandas_cache
        def load_metadata(file_path):
            file_extension = os.path.splitext(file_path).suffix

            if file_extension == ".xlsx":
                df = pd.read_excel(file_path)
            elif file_extension == ".txt" or file_extension == ".tsv":
                df = pd.read_csv(file_path, delimiter = "\t")
            elif file_extension == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = None
                warnings.warn("WARNING: Metadata could not be read. \nMetadata has to be a .xslx, .tsv, .csv or .txt file")

            if df:
                if all(elem in df["sample"].tolist()  for elem in self.mat.index.values) is False:
                    warnings.warn("WARNING: Sample names do not match protein data")
            return df


        
        # insert check whether columns are present
        # if not print missing columns
        # matrix with columns - proteins/peptides rows - samples
        self.rawfile = load_rawdata(rawfile_path)
        self.mat = create_matrix(self.rawfile, intensity_column = intensity_column)
        # df with metadata
        # check if metadata columns match rows in matrix
        self.metadata = None
        if metadata_path:
            self.metadata = load_metadata(metadata_path)
        self.software = software
        self.experiment_type = None
        self.data_format = None

    def summary(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # print summary
        pass

    def plot_pca(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        pass

    def plot_correlation_matrix(self, corr_method = "pearson", save_figure=False):
        """
        """
        corr_matrix = self.mat.transpose().corr(method=corr_method)
        plot = corr_matrix.style.background_gradient(cmap='coolwarm')
        if save_figure:
            save_fig(plot)
        return plot
    
    def plot_volcano():
        pass


"""
- Protein and Peptide dataset one class? What analysis can be performed on protein but not peptides and visa vera?
- Filtering of contaminations?
- How to deal with NAs
- aggregation of proteins into protein groups by means?
- do different versions of maxquant have different column labeling for evidence file?
    -yes 
- how to normalize silac data
- normalize data in _init_?
"""

"""
Plot Jaccard similarity?

"""