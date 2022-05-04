from random import sample
import re
import pandas as pd
import seaborn as sn
from data_cache import pandas_cache
import os
import warnings
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import scipy.stats

def check_param(par):
    pass
    

@pandas_cache
def create_matrix(df, intensity_col = None, proteinID_col = None):
    """Creates a matrix out of the MS Outputfile, with columns displaying samples and
    row the protein IDs.

    Parameters
    ----------
    df : df
        Pandas Dataframe of the MS Outputfile
    intensity_col : str, optional
        columns , by default "LFQ intensity "
    proteinID_col : str, optional
        column in Dataframe containg the Protein IDs, must be unique, by default "Protein IDs"

    Returns# 
    -------
    _type_
        _description_
    """
    df = df.set_index(proteinID_col)
    # check whether column is present print error 
    # the intensity column has a whitespace between intentisity and the peptide label - needs to be checked
    df = df[df.columns[pd.Series(df.columns).str.startswith(intensity_col)]]
    # remove prefix
    df.columns = df.columns.str.lstrip(intensity_col)

    #  - include Normalization
    #  - include Filtering
    return df


@pandas_cache
def load_rawdata(file_path):
    """Loads raw data into dataframe

    Parameters
    ----------
    file_path : str
        path to file

    Returns
    -------
    _type_
       Pandas Dataframe
    """
    df = pd.read_csv(file_path, delimiter = "\t")
    return df


class proteinObject:
    """_summary_
    """
    def __init__(self, rawfile_path: str, 
        metadata_path: str=None, 
        intensity_column =  "LFQ intensity ", 
        software = None, 
        sample_column = "sample",
        proteinID_column = "Protein IDs"):
        """Create a Protein Object containing the protein intensity and the corresponding metadata of the samples,
        ready for analyis 

        Parameters
        ----------
        rawfile_path : str
            path to Protein Intensity file
        metadata_path : str, optional
            path to metadata file (xlsx, csv or tsv), by default None
        intensity_column : str, optional
            , by default None
        software : str, optional
            _description_, by default None
        """

        def load_metadata(file_path, sample_column = None):
            """Load Metadata into Pandas Dataframe

            Parameters
            ----------
            file_path : str
                path to excel, .tsv or .csv - file
            sample_column : str, optional
                column in the file containing the sample names, the names of the samples should
                match sample labelling in the MS-Output file, by default "sample"

            Returns
            -------
            _type_
                Pandas Dataframe containing the metadata
            """
            file_extension = os.path.splitext(file_path)[1]

            if file_extension == ".xlsx":
                df = pd.read_excel(file_path)
            # find robust way to detect file format
            # else give file separation as variable
            elif file_extension == ".txt" or file_extension == ".tsv":
                df = pd.read_csv(file_path, delimiter = "\t")
            elif file_extension == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = None
                warnings.warn("WARNING: Metadata could not be read. \nMetadata has to be a .xslx, .tsv, .csv or .txt file")
                return
            
            df.columns = df.columns.str.replace(sample_column, 'sample')
            # check whether sample labeling matches protein data
            #  warnings.warn("WARNING: Sample names do not match sample labelling in protein data")
            return df
        
        self.rawdata = load_rawdata(rawfile_path)

        # include filtering before 
        self.mat = create_matrix(self.rawdata, intensity_col = intensity_column, proteinID_col= proteinID_column)

        self.metadata = None
        if metadata_path:
            self.metadata = load_metadata(metadata_path, sample_column= sample_column)
        self.software = software
        self.experiment_type = None
        self.data_format = None
        # update normalization when self.matrix is normalized, filtered
        self.normalization = None


    def summary(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # print summary
        pass


    def calculate_ttest_fc(self, column, group1, group2):
        """_summary_

        Args:
            column (_type_): _description_
            group1 (_type_): _description_
            group2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        # get samples names of two groupes
        group1_samples = self.metadata[self.metadata[column] == group1]["sample"].tolist()
        group2_samples = self.metadata[self.metadata[column] == group2]["sample"].tolist()
        # calculate fold change (if its is not logarithimic normalized)
        if self.normalization != "log":
            fc = self.mat[group1_samples].T.mean().values/self.mat[group2_samples].T.mean().values
    
        # calculate p-values 
        # output needs to be checked
        p_values = self.mat.apply(lambda row: scipy.stats.ttest_ind(self.mat[group1_samples].T.values.flatten(), self.mat[group2_samples].T.values.flatten())[1])
        df = pd.DataFrame()
        df["Protein IDs"] = p_values.index.tolist()
        df["fc"] = fc
        df["pvalue"] = p_values.values
        return df



    def plot_pca(self, group = None):
        """plot PCA

        Parameters
        ----------
        group : _type_, optional
            _description_, by default None
        """
        if group: 
            mat = self.mat[self.metadata["sample"].tolist()]
        else:
            mat = self.mat

        # needs to be checked with publications
        # depends on normalization whether NA can be replaced with 0  
        mat = mat.fillna(0)
        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
        components = pipeline.fit_transform(mat.transpose())
    
        if group:
            fig = px.scatter(components, x=0, y=1, color = self.metadata[group])
        else:
            fig = px.scatter(components, x=0, y=1)
        fig.show()
        return fig


    def plot_correlation_matrix(self, corr_method = "pearson", save_figure=False):
        """_summary_

        Parameters
        ----------
        corr_method : str, optional
            _description_, by default "pearson"
        save_figure : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        corr_matrix = self.mat.corr(method=corr_method)
        plot = px.imshow(corr_matrix)
        return plot
    

    def plot_volcano(self, column, group1, group2):
        df = self.calculate_ttest_fc(column, group1, group2)
        pass













