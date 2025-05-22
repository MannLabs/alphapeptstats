import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly
import scipy

from alphastats.dataset.factory import DataSetFactory
from alphastats.dataset.harmonizer import DataHarmonizer
from alphastats.dataset.keys import Cols
from alphastats.dataset.plotting import Plot
from alphastats.dataset.preprocessing import Preprocess
from alphastats.dataset.statistics import Statistics
from alphastats.dataset.utils import (
    LoaderError,
    check_for_missing_values,
    ignore_warning,
)
from alphastats.loader.base_loader import BaseLoader
from alphastats.plots.clustermap import ClusterMap
from alphastats.plots.dimensionality_reduction import DimensionalityReduction
from alphastats.plots.intensity_plot import IntensityPlot
from alphastats.plots.sample_histogram import SampleHistogram
from alphastats.plots.volcano_plot import VolcanoPlot
from alphastats.statistics.tukey_test import tukey_test

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


class DataSet:
    """The main object of AlphaPeptStats, providing an interface to all relevant functionality and data."""

    def __init__(
        self,
        loader: BaseLoader,
        metadata_path_or_df: Optional[Union[str, pd.DataFrame]] = None,
        sample_column: Optional[str] = None,
    ):
        """Create DataSet

        Args:
            loader (_type_): loader of class AlphaPeptLoader, MaxQuantLoader, DIANNLoader, FragPipeLoader, SpectronautLoader
            metadata_path_or_df (str or pd.DataFrame, optional): path to metadata file or an actual df. Defaults to None.
            sample_column (str, optional): column in metadata file indicating the sample IDs. Defaults to None.

        Attributes of a DataSet instance:
            DataSet().rawinput: Raw Protein data.
            DataSet().mat:      Processed data matrix with ProteinIDs/ProteinGroups as columns and samples as rows. All computations are performed on this matrix.
            DataSet().metadata: Metadata for the samples in the matrix. Metadata will be matched with DataSet().mat when needed (for instance Volcano Plot).

        """
        self._check_loader(loader=loader)

        self._data_harmonizer = DataHarmonizer(
            loader, sample_column
        )  # TODO should be moved to the loaders

        # fill data from loader
        self.rawinput: pd.DataFrame = self._data_harmonizer.get_harmonized_rawinput(
            loader.rawinput
        )
        self.filter_columns: List[str] = loader.filter_columns
        self.software: str = loader.software
        self._intensity_column: Union[str, list] = (
            loader._extract_sample_names(
                metadata=self.metadata, sample_column=sample_column
            )
            if loader
            == "Generic"  # TODO is this ever the case? not rather instanceof(loader, GenericLoader)?
            else loader.intensity_column
        )

        # self.evidence_df: pd.DataFrame = loader.evidence_df  # TODO unused

        # TODO: Add a store for dea results here

        self._dataset_factory = DataSetFactory(
            rawinput=self.rawinput,
            intensity_column=self._intensity_column,
            metadata_path_or_df=metadata_path_or_df,
            data_harmonizer=self._data_harmonizer,
        )

        rawmat, mat, metadata, preprocessing_info = self._get_init_dataset()
        self.rawmat: pd.DataFrame = rawmat
        self.mat: pd.DataFrame = mat
        self.metadata: pd.DataFrame = metadata
        self.preprocessing_info: Dict = preprocessing_info

        # TODO: Make these public attributes
        (
            self._gene_to_features_map,
            self._protein_to_features_map,
            self._feature_to_repr_map,
        ) = self._create_id_dicts()

        print("DataSet has been created.")

    def _get_init_dataset(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Get the initial data structure for the DataSet."""
        rawmat, mat = self._dataset_factory.create_matrix_from_rawinput()

        metadata = self._dataset_factory.create_metadata(mat)

        preprocessing_info = Preprocess.init_preprocessing_info(
            num_samples=mat.shape[0],
            num_protein_groups=mat.shape[1],
            intensity_column=self._intensity_column,
            filter_columns=self.filter_columns,
        )

        return rawmat, mat, metadata, preprocessing_info

    @staticmethod
    def _check_loader(loader):
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

    def _create_id_dicts(self, sep: str = ";") -> Tuple[dict, dict, dict]:
        """
        Create mappings from gene and protein to feature, and from feature to representation.
        Features are the entities measured in each sample, usually protein groups represented by semicolon separated protein ids.
        This is to maintain the many-to-many relationships between the three entities feature, protein and gene.

        This method processes the raw input data to generate three dictionaries:
        1. gene_to_features_map: Maps each gene to a list of features.
        2. protein_to_features_map: Maps each protein to a list of features.
        3. feature_to_repr_map: Maps each feature to its representation string.

        Args:
            sep (str): The separator used to split gene and protein identifiers. Default is ";".

        Returns:
            Tuple[dict, dict, dict]: A tuple containing three dictionaries:
            - gene_to_features_map (dict): A dictionary mapping genes to features.
            - protein_to_features_map (dict): A dictionary mapping proteins to features.
            - feature_to_repr_map (dict): A dictionary mapping features to their representation strings.
        """

        features = set(self.mat.columns.to_list())
        gene_to_features_map = defaultdict(list)
        protein_to_features_map = defaultdict(list)
        feature_to_repr_map = {}

        for proteins, feature in zip(
            self.rawinput[Cols.INDEX], self.rawinput[Cols.INDEX]
        ):
            if feature not in features:
                continue
            # TODO: Shorten list if too many ids e.g. to id1;...(19) if 20 ids are present
            feature_to_repr_map[feature] = "ids:" + proteins
            for protein in proteins.split(sep):
                protein_to_features_map[protein].append(feature)

        if Cols.GENE_NAMES in self.rawinput.columns:
            for genes, feature in zip(
                self.rawinput[Cols.GENE_NAMES], self.rawinput[Cols.INDEX]
            ):
                if feature not in features:
                    continue
                if isinstance(genes, str):
                    for gene in genes.split(sep):
                        gene_to_features_map[gene].append(feature)
                    feature_to_repr_map[feature] = genes

        return gene_to_features_map, protein_to_features_map, feature_to_repr_map

    def _get_preprocess(self) -> Preprocess:
        """Return instance of the Preprocess object."""
        return Preprocess(
            self.filter_columns,
            self.rawinput,
            self.metadata,
            self.preprocessing_info,
            self.mat,
        )

    def preprocess(
        self,
        log2_transform: bool = False,
        remove_contaminations: bool = False,
        subset: bool = False,
        replace_zeroes: bool = False,
        data_completeness: float = 0,
        normalization: str = None,
        imputation: str = None,
        remove_samples: list = None,
        drop_unmeasured_features: bool = False,
        **kwargs,
    ) -> None:
        """A wrapper for Preprocess.preprocess(), see documentation there."""
        self.mat, self.metadata, self.preprocessing_info = (
            self._get_preprocess().preprocess(
                log2_transform,
                remove_contaminations,
                subset,
                replace_zeroes,
                data_completeness,
                normalization,
                imputation,
                remove_samples,
                drop_unmeasured_features,
                **kwargs,
            )
        )
        (
            self._gene_to_features_map,
            self._protein_to_features_map,
            self._feature_to_repr_map,
        ) = self._create_id_dicts()

    def reset_preprocessing(self):
        """Reset all preprocessing steps"""
        (
            self.rawmat,
            self.mat,
            self.metadata,
            self.preprocessing_info,
        ) = self._get_init_dataset()
        (
            self._gene_to_features_map,
            self._protein_to_features_map,
            self._feature_to_repr_map,
        ) = self._create_id_dicts()

    def batch_correction(self, batch: str) -> None:
        """A wrapper for Preprocess.batch_correction(), see documentation there."""
        self.mat = self._get_preprocess().batch_correction(batch)

    def _get_statistics(self) -> Statistics:
        """Return instance of the Statistics object."""
        return Statistics(
            mat=self.mat,
            metadata=self.metadata,
            preprocessing_info=self.preprocessing_info,
        )

    # TODO: Add function get_differential_expression_analysis() which will handle the dea store and run diff_expression_analysis() if necessary
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
        # TODO: This method is the one which will be called if a dea result is not yet in the store.
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
        df = self.mat[[protein_id]].reset_index().rename(columns={"index": Cols.SAMPLE})
        df = df.merge(self.metadata, how="inner", on=[Cols.SAMPLE])

        return tukey_test(
            df,
            protein_id,
            group,
        )

    def anova(self, column: str, protein_ids="all", tukey: bool = True) -> pd.DataFrame:
        """A wrapper for Statistics.anova(), see documentation there."""
        return self._get_statistics().anova(column, protein_ids, tukey)

    def ancova(
        self, protein_id: str, covar: Union[str, list], between: str
    ) -> pd.DataFrame:
        """A wrapper for Statistics.ancova(), see documentation there."""
        return self._get_statistics().ancova(protein_id, covar, between)

    def multicova_analysis(
        self,
        covariates: list,
        n_permutations: int = 3,
        fdr: float = 0.05,
        s0: float = 0.05,
        subset: dict = None,
    ) -> Tuple[pd.DataFrame, list]:
        """A wrapper for Statistics.multicova_analysis(), see documentation there."""
        return self._get_statistics().multicova_analysis(
            covariates, n_permutations, fdr, s0, subset
        )

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
            preprocessing_info=self.preprocessing_info,
            group=group,
            method="umap",
            circle=circle,
        )
        return dimensionality_reduction.plot

    def perform_dimensionality_reduction(
        self, method: str, group: Optional[str] = None, circle: bool = False
    ):
        """Generic wrapper for dimensionality reduction methods to be used by LLM.

        Args:
            method (str): "pca", "tsne", "umap"
            group (str, optional): column in metadata that should be used for coloring. Defaults to None.
            circle (bool, optional): draw circle around each group. Defaults to False.
        """

        result = {
            "pca": self.plot_pca,
            "tsne": self.plot_tsne,
            "umap": self.plot_umap,
        }.get(method)
        if result is None:
            raise ValueError(f"Invalid method: {method}")

        return result(group=group, circle=circle)

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
        color_list: list = None,
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
            #compare_preprocessing_modes(bool): Will iterate through normalization and imputation modes and return a list of VolcanoPlots in different settings, Default False.


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
        if color_list is None:
            color_list = []
        volcano_plot = VolcanoPlot(
            mat=self.mat,
            rawinput=self.rawinput,
            metadata=self.metadata,
            preprocessing_info=self.preprocessing_info,
            feature_to_repr_map=self._feature_to_repr_map,
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

    def _get_feature_ids_from_search_string(self, string: str) -> List[str]:
        """Get the feature id from a string representing a feature.

        Goes through id mapping dictionaries and finds the completest match.

        Parameters
        ----------
        string : str
            The string representating the feature."""

        if string in self._feature_to_repr_map:
            return [string]
        representation_keys = [
            feature
            for feature, representation in self._feature_to_repr_map.items()
            if representation == string
        ]
        if representation_keys:
            return representation_keys
        if string in self._protein_to_features_map:
            return self._protein_to_features_map[string]
        if string in self._gene_to_features_map:
            return self._gene_to_features_map[string]
        raise ValueError(f"Feature {string} is not in the (processed) data.")

    def _get_multiple_feature_ids_from_strings(self, features: List) -> List:
        """Get the feature ids from a list of strings representing features.

        Parameters
        ----------
        features : list
            A list of strings representing the features."""

        unmapped_features = []
        protein_ids = []
        for feature in features:
            try:
                for protein_id in self._get_feature_ids_from_search_string(feature):
                    protein_ids.append(protein_id)
            except ValueError:
                unmapped_features.append(feature)
        if unmapped_features:
            warnings.warn(
                f"Could not find the following features: {', '.join(unmapped_features)}"
            )
        if not protein_ids:
            raise ValueError("No valid features provided.")

        return protein_ids

    def plot_intensity(
        self,
        *,
        feature: str,
        group: str = None,
        subgroups: list = None,
        method: str = "box",  # TODO rename
        add_significance: bool = False,
        log_scale: bool = False,
        # compare_preprocessing_modes: bool = False, TODO reimplement
    ):
        """Plot Intensity of individual Protein/ProteinGroup

        Args:
            feature (str): ProteinGroup ID, gene name or feature representation, or comma-separated list thereof.
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

        if "," in feature:
            features = [substring.strip() for substring in feature.split(",")]
        else:
            features = [feature]
        protein_id = self._get_multiple_feature_ids_from_strings(features)

        intensity_plot = IntensityPlot(
            mat=self.mat,
            metadata=self.metadata,
            intensity_column=self._intensity_column,
            preprocessing_info=self.preprocessing_info,
            protein_id=protein_id,
            feature_to_repr_map=self._feature_to_repr_map,
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
            preprocessing_info=self.preprocessing_info,
            label_bar=label_bar,
            only_significant=only_significant,
            group=group,
            subgroups=subgroups,
        )
        return clustermap.plot

    def plot_samplehistograms(self):
        """Plots the density distribution of each sample

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
            self.preprocessing_info,
        )

    def plot_correlation_matrix(self, method: str = "pearson"):  # TODO unused
        """A wrapper for Plot.plot_correlation_matrix(), see documentation there."""
        return self._get_plot().plot_correlation_matrix(method)

    def plot_sampledistribution(
        self,
        method: str = "violin",
        color: str = None,  # TODO rename to group
        log_scale: bool = False,
        use_raw: bool = False,
    ):
        """A wrapper for Plot.plot_sampledistribution(), see documentation there."""
        return self._get_plot().plot_sampledistribution(
            method, color, log_scale, use_raw
        )

    def plot_dendrogram(
        self, linkagefun=lambda x: scipy.cluster.hierarchy.linkage(x, "complete")
    ):
        """A wrapper for Plot.plot_dendrogram(), see documentation there."""
        return self._get_plot().plot_dendrogram(linkagefun)
