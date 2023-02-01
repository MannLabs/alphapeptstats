from alphastats.plots.PlotUtils import PlotUtils, plotly_object
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import pandas as pd

# make own alphastats theme
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


class DimensionalityReduction(PlotUtils):
    def __init__(self, dataset, group, method, circle, **kwargs) -> None:
        self.dataset = dataset
        self.method = method
        self.circle = circle
        self.group = group
        self.plot = None

        sample_names, group_color = self._prepare_df()

        if self.method == "pca":
            self._pca()

        elif self.method == "umap":
            self._umap()

        elif self.method == "tsne":
            self._tsne(**kwargs)

        self._plot(sample_names=sample_names, group_color=group_color)

    def _add_circles_to_scatterplot(self, fig):
        # called by _plot_dimensionality_reduction()
        # convert figure to dict and extract information
        fig_dict = fig.to_plotly_json().get("data")
        for group in fig_dict:
            # get coordinates for the group
            x_vector = group.get("x")
            y_vector = group.get("y")
            # get color of the group to color circle in the same color
            group_color = group.get("marker").get("color")
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=min(x_vector),
                y0=min(y_vector),
                x1=max(x_vector),
                y1=max(y_vector),
                opacity=0.2,
                fillcolor=group_color,
                line_color=group_color,
            )
        return fig

    def _prepare_df(self):

        if self.group:
            mat = self.dataset._subset()
            self.dataset.metadata[self.group] = self.dataset.metadata[self.group].apply(
                str
            )
            group_color = self.dataset.metadata[self.group]
            sample_names = self.dataset.metadata[self.dataset.sample].to_list()

        else:
            mat = self.dataset.mat
            group_color = self.group
            sample_names = mat.reset_index(level=0)["index"].to_list()

        mat = mat.fillna(0)
        self.prepared_df = mat

        return sample_names, group_color

    def _pca(self):
        pca = sklearn.decomposition.PCA(n_components=2)
        self.components = pca.fit_transform(self.prepared_df)
        self.labels = {
            "0": "PC 1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100),
            "1": "PC 2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100),
        }

    def _tsne(self, **kwargs):
        tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, **kwargs)
        self.components = tsne.fit_transform(self.prepared_df)
        self.labels = {
            "0": "Dimension 1",
            "1": "Dimension 2",
        }

    def _umap(self):

        # TODO umap import is reallly buggy 
        try:
            import umap.umap_ as umap
        except ModuleNotFoundError:
            import umap
        
        umap_2d = umap.UMAP(n_components=2, init="random", random_state=0)
        self.components = umap_2d.fit_transform(self.prepared_df)
        self.labels = {
            "0": "",
            "1": "",
        }

    def _plot(self, sample_names, group_color):
        components = pd.DataFrame(self.components)
        components[self.dataset.sample] = sample_names

        fig = px.scatter(
            components,
            x=0,
            y=1,
            labels=self.labels,
            color=group_color,
            hover_data=[components[self.dataset.sample]],
        )

        # rename hover_data_0 to sample
        fig_dict = fig.to_plotly_json()
        data = fig_dict.get("data")

        for count, d in enumerate(data):
            hover = d.get("hovertemplate").replace("hover_data_0", "sample")
            fig_dict["data"][count]["hovertemplate"] = hover
        fig = go.Figure(fig_dict)

        #  save plotting data in figure object
        fig = plotly_object(fig)
        fig = self._update_figure_attributes(
            figure_object=fig,
            plotting_data=pd.DataFrame(components),
            method=self.method,
            preprocessing_info=self.dataset.preprocessing_info
        )

        # draw circles around plotted groups
        if self.circle is True and self.group is not None:
            fig = self._add_circles_to_scatterplot(fig)

        if self.group:
            fig.update_layout(legend_title_text=self.group)

        self.plot = fig
