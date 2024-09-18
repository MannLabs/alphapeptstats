from typing import List, Union, Dict, Optional

import pandas as pd
import numpy as np
import logging
import warnings
import plotly
import scipy

from alphastats import BaseLoader


from alphastats.DataSet_Plot import Plot
from alphastats.DataSet_Preprocess import Preprocess
from alphastats.DataSet_Statistics import Statistics
from alphastats.utils import LoaderError, check_for_missing_values, ignore_warning
from alphastats.statistics.tukey_test import tukey_test
from alphastats.plots.ClusterMap import ClusterMap
from alphastats.plots.DimensionalityReduction import DimensionalityReduction
from alphastats.plots.IntensityPlot import IntensityPlot
from alphastats.plots.SampleHistogram import SampleHistogram
from alphastats.plots.VolcanoPlot import VolcanoPlot

plotly.io.templates["alphastats_colors"] = plotly.graph_objects.layout.Template(
    layout=plotly.graph_objects.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            "#009599",
            "#005358",
            "#772173",
            "#B65EAF",  # pink
            "#A73A00",
            "#6490C1",
            "#FF894F",
            "#2B5E8B",
            "#A87F32",
        ],
    )
)

plotly.io.templates.default = "simple_white+alphastats_colors"


class DataSet(Plot):
    """Analysis Object"""

    def __init__(
        self,
        loader: BaseLoader,
        metadata_path: Optional[str] = None,
        sample_column: Optional[str] = None,
    ):
        """Create DataSet

        Args:
            loader (_type_): loader of class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader, SpectronautLoader
            metadata_path (str, optional): path to metadata file. Defaults to None.
            sample_column (str, optional): column in metadata file indicating the sample IDs. Defaults to None.

        Attributes of a DataSet instance:
            DataSet().rawinput: Raw Protein data.
            DataSet().mat:      Processed data matrix with ProteinIDs/ProteinGroups as columns and samples as rows. All computations are performed on this matrix.
            DataSet().metadata: Metadata for the samples in the matrix. Metadata will be matched with DataSet().mat when needed (for instance Volcano Plot).

        """
        self._check_loader(loader=loader)

        self.rawinput: pd.DataFrame = loader.rawinput
        self.software: str = loader.software
        self.index_column: str = loader.index_column
        self.intensity_column: Union[str, list] = loader.intensity_column
        self.filter_columns: List[str] = loader.filter_columns
        self.evidence_df: pd.DataFrame = loader.evidence_df
        self.gene_names: str = loader.gene_names

        # include filtering before
        self._create_matrix()
        self._check_matrix_values()

        self.metadata: pd.DataFrame
        self.sample: str
        if metadata_path is not None:
            self.sample = sample_column
            self.metadata = self._load_metadata(file_path=metadata_path)
            self._remove_misc_samples_in_metadata()
        else:
            self.sample = "sample"
            self.metadata = pd.DataFrame({"sample": list(self.mat.index)})

        if loader == "Generic":
            intensity_column = loader._extract_sample_names(
                metadata=self.metadata, sample_column=self.sample
            )
            self.intensity_column = intensity_column

        # init preprocessing settings
        self.preprocessing_info: Dict = Preprocess.init_preprocessing_info(
            num_samples=self.mat.shape[0],
            num_protein_groups=self.mat.shape[1],
            intensity_column=self.intensity_column,
            filter_columns=self.filter_columns,
        )

        self.preprocessed = False
        self.preprocessed: bool = False

        print("DataSet has been created.")

    def _get_preprocess(self) -> Preprocess:
        """Return instance of the Preprocess object."""
        return Preprocess(
            self.filter_columns,
            self.rawinput,
            self.index_column,
            self.sample,
            self.metadata,
            self.preprocessing_info,
            self.mat,
        )

    def preprocess(
        self,
        log2_transform: bool = True,
        remove_contaminations: bool = False,
        subset: bool = False,
        data_completeness: float = 0,
        normalization: str = None,
        imputation: str = None,
        remove_samples: list = None,
        **kwargs,
    ) -> None:
        """A wrapper for Preprocess.preprocess(), see documentation there."""
        self.mat, self.metadata, self.preprocessing_info = (
            self._get_preprocess().preprocess(
                log2_transform,
                remove_contaminations,
                subset,
                data_completeness,
                normalization,
                imputation,
                remove_samples,
                **kwargs,
            )
        )
        self.preprocessed = True

    def reset_preprocessing(self):
        """Reset all preprocessing steps"""
        self._create_matrix()
        self.preprocessing_info = Preprocess.init_preprocessing_info(
            num_samples=self.mat.shape[0],
            num_protein_groups=self.mat.shape[1],
            intensity_column=self.intensity_column,
            filter_columns=self.filter_columns,
        )

        self.preprocessed = False
        # TODO fix bug: metadata is not reset/reloaded here
        print("All preprocessing steps are reset.")

    def batch_correction(self, batch: str) -> None:
        """A wrapper for Preprocess.batch_correction(), see documentation there."""
        self.mat = self._get_preprocess().batch_correction(batch)

    def _get_statistics(self) -> Statistics:
        """Return instance of the Statistics object."""
        return Statistics(
            mat=self.mat,
            metadata=self.metadata,
            index_column=self.index_column,
            sample=self.sample,
            preprocessing_info=self.preprocessing_info,
        )

    def diff_expression_analysis(
        self,
        group1: Union[str, list],
        group2: Union[str, list],
        column: str = None,
        method: str = "ttest",
        perm: int = 10,
        fdr: float = 0.05,
    ) -> pd.DataFrame:
        """A wrapper for the Statistics.diff_expression_analysis(), see documentation there."""
        return self._get_statistics().diff_expression_analysis(
            group1,
            group2,
            column,
            method,
            perm,
            fdr,
        )

    def tukey_test(self, protein_id: str, group: str) -> pd.DataFrame:
        """A wrapper for tukey_test.tukey_test(), see documentation there."""
        df = self.mat[[protein_id]].reset_index().rename(columns={"index": self.sample})
        df = df.merge(self.metadata, how="inner", on=[self.sample])

        return tukey_test(
            df,
            protein_id,
            group,
            self.index_column,
        )

    def anova(self, column: str, protein_ids="all", tukey: bool = True) -> pd.DataFrame:
        """A wrapper for Statistics.anova(), see documentation there."""
        return self._get_statistics().anova(column, protein_ids, tukey)

    def ancova(
        self, protein_id: str, covar: Union[str, list], between: str
    ) -> pd.DataFrame:
        """A wrapper for Statistics.ancova(), see documentation there."""
        return self._get_statistics().ancova(protein_id, covar, between)

    @check_for_missing_values
    def plot_pca(self, group: Optional[str] = None, circle: bool = False):
        """Plot Principal Component Analysis (PCA)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: PCA plot
        """
        dimensionality_reduction = DimensionalityReduction(
            mat=self.mat,
            metadata=self.metadata,
            sample=self.sample,
            preprocessing_info=self.preprocessing_info,
            group=group,
            circle=circle,
            method="pca",
        )
        return dimensionality_reduction.plot

    @check_for_missing_values
    def plot_tsne(
        self,
        group: Optional[str] = None,
        circle: bool = False,
        perplexity: int = 5,
        n_iter: int = 1000,
    ):
        """Plot t-distributed stochastic neighbor embedding (t-SNE)

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: t-SNE plot
        """
        dimensionality_reduction = DimensionalityReduction(
            mat=self.mat,
            metadata=self.metadata,
            sample=self.sample,
            preprocessing_info=self.preprocessing_info,
            group=group,
            method="tsne",
            circle=circle,
            perplexity=perplexity,
            n_iter=n_iter,
        )
        return dimensionality_reduction.plot

    @check_for_missing_values
    def plot_umap(self, group: Optional[str] = None, circle: bool = False):
        """Plot Uniform Manifold Approximation and Projection for Dimension Reduction

        Args:
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: UMAP plot
        """
        dimensionality_reduction = DimensionalityReduction(
            mat=self.mat,
            metadata=self.metadata,
            sample=self.sample,
            preprocessing_info=self.preprocessing_info,
            group=group,
            method="umap",
            circle=circle,
        )
        return dimensionality_reduction.plot

    @ignore_warning(RuntimeWarning)
    def plot_volcano(
        self,
        group1: Union[str, list],
        group2: Union[str, list],
        column: str = None,
        method: str = "ttest",
        labels: bool = False,
        min_fc: float = 1.0,
        alpha: float = 0.05,
        draw_line: bool = True,
        perm: int = 100,
        fdr: float = 0.05,
        # compare_preprocessing_modes: bool = False, # TODO reimplement
        color_list: list = [],
    ):
        """Plot Volcano Plot

        Args:
            column (str): column name in the metadata file with the two groups to compare
            group1 (str/list): name of group to compare needs to be present in column or list of sample names to compare
            group2 (str/list): name of group to compare needs to be present in column  or list of sample names to compare
            method (str): "anova", "wald", "ttest", "SAM" Defaul ttest.
            labels (bool): Add text labels to significant Proteins, Default False.
            alpha(float,optional): p-value cut off.
            min_fc (float): Minimum fold change.
            draw_line(boolean): whether to draw cut off lines.
            perm(float,optional): number of permutations when using SAM as method. Defaults to 100.
            fdr(float,optional): FDR cut off when using SAM as method. Defaults to 0.05.
            color_list (list): list with ProteinIDs that should be highlighted.
            compare_preprocessing_modes(bool): Will iterate through normalization and imputation modes and return a list of VolcanoPlots in different settings, Default False.


        Returns:
            plotly.graph_objects._figure.Figure: Volcano Plot
        """

        # TODO this needs to orchestrated from outside this method
        # if compare_preprocessing_modes:
        #     params_for_func = locals()
        #     results = self._compare_preprocessing_modes(
        #         func=VolcanoPlot, params_for_func=params_for_func
        #     )
        #     return results
        #
        # else:
        volcano_plot = VolcanoPlot(
            mat=self.mat,
            rawinput=self.rawinput,
            metadata=self.metadata,
            sample=self.sample,
            index_column=self.index_column,
            gene_names=self.gene_names,
            preprocessing_info=self.preprocessing_info,
            group1=group1,
            group2=group2,
            column=column,
            method=method,
            labels=labels,
            min_fc=min_fc,
            alpha=alpha,
            draw_line=draw_line,
            perm=perm,
            fdr=fdr,
            color_list=color_list,
        )

        return volcano_plot.plot

    def plot_intensity(
        self,
        protein_id: str,
        group: str = None,
        subgroups: list = None,
        method: str = "box",
        add_significance: bool = False,
        log_scale: bool = False,
        # compare_preprocessing_modes: bool = False, TODO reimplement
    ):
        """Plot Intensity of individual Protein/ProteinGroup

        Args:
            protein_id (str): ProteinGroup ID
            group (str, optional): A metadata column used for grouping. Defaults to None.
            subgroups (list, optional): Select variables from the group column. Defaults to None.
            method (str, optional):  Violinplot = "violin", Boxplot = "box", Scatterplot = "scatter" or "all". Defaults to "box".
            add_significance (bool, optional): add p-value bar, only possible when two groups are compared. Defaults False.
            log_scale (bool, optional): yaxis in logarithmic scale. Defaults to False.

        Returns:
            plotly.graph_objects._figure.Figure: Plotly Plot
        """
        # TODO this needs to orchestrated from outside this method
        # if compare_preprocessing_modes:
        #     params_for_func = locals()
        #     results = self._compare_preprocessing_modes(
        #         func=IntensityPlot, params_for_func=params_for_func
        #     )
        #     return results

        intensity_plot = IntensityPlot(
            mat=self.mat,
            metadata=self.metadata,
            sample=self.sample,
            intensity_column=self.intensity_column,
            preprocessing_info=self.preprocessing_info,
            protein_id=protein_id,
            group=group,
            subgroups=subgroups,
            method=method,
            add_significance=add_significance,
            log_scale=log_scale,
        )

        return intensity_plot.plot

    @ignore_warning(UserWarning)
    @check_for_missing_values
    def plot_clustermap(
        self,
        label_bar: str = None,
        only_significant: bool = False,
        group: str = None,
        subgroups: list = None,
    ):
        """Plot a matrix dataset as a hierarchically-clustered heatmap

        Args:
            label_bar (str, optional): column/variable name described in the metadata. Will be plotted as bar above the heatmap to see wheteher groups are clustering together. Defaults to None.. Defaults to None.
            only_significant (bool, optional): performs ANOVA and only signficantly different proteins will be clustered (p<0.05). Defaults to False.
            group (str, optional): group containing subgroups that should be clustered. Defaults to None.
            subgroups (list, optional): variables in group that should be plotted. Defaults to None.

        Returns:
             ClusterGrid: Clustermap
        """

        clustermap = ClusterMap(
            mat=self.mat,
            metadata=self.metadata,
            sample=self.sample,
            index_column=self.index_column,
            preprocessing_info=self.preprocessing_info,
            label_bar=label_bar,
            only_significant=only_significant,
            group=group,
            subgroups=subgroups,
        )
        return clustermap.plot

    def plot_samplehistograms(self):
        """Plots the Denisty distribution of each sample

        Returns:
            plotly: Plotly Graph Object
        """
        return SampleHistogram(mat=self.mat).plot()

    def _get_plot(self) -> Plot:
        """Get instance of the Plot object."""
        return Plot(
            self.mat,
            self.rawmat,
            self.metadata,
            self.sample,
            self.preprocessing_info,
        )

    def plot_correlation_matrix(self, method: str = "pearson"):  # TODO unused
        return self._get_plot().plot_correlation_matrix(method)

    def plot_sampledistribution(
        self,
        method: str = "violin",
        color: str = None,
        log_scale: bool = False,
        use_raw: bool = False,
    ):
        return self._get_plot().plot_sampledistribution(
            method, color, log_scale, use_raw
        )

    def plot_dendrogram(
        self, linkagefun=lambda x: scipy.cluster.hierarchy.linkage(x, "complete")
    ):
        return self._get_plot().plot_dendrogram(linkagefun)

    def _check_loader(self, loader):
        """Checks if the Loader is from class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader

        Args:
            loader : loader
        """
        if not isinstance(loader, BaseLoader):
            raise LoaderError(
                "loader must be a subclass of BaseLoader, "
                f"got {loader.__class__.__name__}"
            )

        if not isinstance(loader.rawinput, pd.DataFrame) or loader.rawinput.empty:
            raise ValueError(
                "Error in rawinput, consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader, SpectronautLoader"
            )

        if not isinstance(loader.index_column, str):
            raise ValueError(
                "Invalid index_column: consider reloading your data with: AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader, SpectronautLoader"
            )

    def _check_matrix_values(self):
        if np.isinf(self.mat).values.sum() > 0:
            logging.warning("Data contains infinite values.")

    def _remove_misc_samples_in_metadata(self):
        samples_matrix = self.mat.index.to_list()
        samples_metadata = self.metadata[self.sample].to_list()
        misc_samples = list(set(samples_metadata) - set(samples_matrix))
        if len(misc_samples) > 0:
            self.metadata = self.metadata[
                ~self.metadata[self.sample].isin(misc_samples)
            ]
            logging.warning(
                f"{misc_samples} are not described in the protein data and"
                "are removed from the metadata."
            )

    def _create_matrix(self):
        """
        Creates a matrix of the Outputfile, with columns displaying features (Proteins) and
        rows the samples.
        """

        df = self.rawinput
        df = df.set_index(self.index_column)

        if isinstance(self.intensity_column, str):
            regex_find_intensity_columns = self.intensity_column.replace(
                "[sample]", ".*"
            )
            df = df.filter(regex=(regex_find_intensity_columns), axis=1)
            # remove Intensity so only sample names remain
            substring_to_remove = regex_find_intensity_columns.replace(".*", "")
            df.columns = df.columns.str.replace(substring_to_remove, "")

        else:
            df = df[self.intensity_column]

        # transpose dataframe
        mat = df.transpose()
        mat.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.rawmat = mat

        # remove proteins with only zero  # TODO this is re-done in preprocessing
        mat_no_zeros = mat.loc[:, (mat != 0).any(axis=0)]
        self.mat = mat_no_zeros.astype(float)

    def _load_metadata(
        self, file_path: Union[pd.DataFrame, str]
    ) -> Optional[pd.DataFrame]:
        """Load metadata either xlsx, txt, csv or txt file

        Args:
            file_path: path to metadata file or metadata DataFrame  # TODO disentangle this
        """
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        elif file_path.endswith(".xlsx"):
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="openpyxl",
                # message=r"/extension is not supported and will be removed/",  # this somehow does not work here?
            )
            df = pd.read_excel(file_path)
            # find robust way to detect file format
            # else give file separation as variable
        elif file_path.endswith(".txt") or file_path.endswith(".tsv"):
            df = pd.read_csv(file_path, delimiter="\t")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            logging.warning(
                "WARNING: Metadata could not be read. \nMetadata has to be a .xslx, .tsv, .csv or .txt file"
            )
            return None

        if df is not None and self.sample not in df.columns:
            logging.error(f"sample_column: {self.sample} not found in {file_path}")

        # check whether sample labeling matches protein data
        #  warnings.warn("WARNING: Sample names do not match sample labelling in protein data")
        df.columns = df.columns.astype(str)
        return df
