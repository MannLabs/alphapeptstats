Functions
=================

Preprocessing
---------------

* All preprocessing steps can be done with: :py:meth:`~alphastats.DataSet_Preprocess.Preprocess.preprocess`
* The information about the preprocessing steps can be accessed any time using :py:meth:`~alphastats.DataSet_Preprocess.Preprocess.preprocess_print_info`


Figures
----------

To generate interactive plots, AlphaStats uses the graphing library `Plotly <https://plotly.com/python/>`_
and all plotting methods will return a plotly object.
The plotly graphs returned by AlphaStats can be customized.
A description on how do customize your plots can be found `here <https://maegul.gitbooks.io/resguides-plotly/content/content/plotting_locally_and_offline/python/methods_for_updating_the_figure_or_graph_objects.html>`_


**Plot Intensity**

* Plot Intensity for indiviual Protein/ProteinGroup :py:meth:`~alphastats.DataSet_Plot.Plot.plot_intensity`
* Plot Intensity distribution for each sample  :py:meth:`~alphastats.DataSet_Plot.Plot.plot_sampledistribution`


**Dimensionality reduction plots:**

* Principal Component Analysis (PCA): :py:meth:`~DataSet_Plot.Plot.plot_pca`
* t-SNE: :py:meth:`~alphastats.DataSet_Plot.Plot.plot_tsne`
* UMAP :py:meth:`~alphastats.DataSet_Plot.Plot.plot_umap`

**Plot Distance between samples**

* Plot correlation matrix :py:meth:`~plot_correlation_matrix`

* Plot Dendrogram :py:meth:`~Plot.plot_dendrogram`

* Plot Clustermap :py:meth:`alphastats.DataSet_Plot.Plot.plot_clustermap`

**Volcano Plot**

To estimate the differential expression between two groups, the function plot_volcano() either performs a t-test, an ANOVA
or a Wald-test using the package `diffxpy <https://github.com/theislab/diffxpy>`_ .

* Volcano Plot :py:meth:`~DataSet.plot_volcano`

The results of the statistical analysis for the volcano plot will be saved within the plot and can be accessed:

.. code-block:: python

    plot = DataSet.plot_volcano(column = "disease", group1 = "healthy", group2 = "cancer")
    plot.plotting_data

**Save Figures**

The plots will return a plotly object, thus you can use write_image() from plotly.
More details on how to save plotly figures you can find `here <https://plotly.com/python/static-image-export/>`_.

.. code-block::python

    plot = DataSet.plot_volcano(column = "disease", group1 = "healthy", group2 = "cancer")
    plot.write_image("images/volcano_plot.svg")


Statistical Analysis
----------------------

* Perform Differential Expression Analysis a Wald test or t-test `diffxpy <https://github.com/theislab/diffxpy>`_.  :py:meth:`~alphastats.DataSet_Statistics.Statistics.diff_expression_analysis`
* ANOVA  :py:meth:`~alphastats.DataSet_Statistics.Statistics.anova`
* ANCOVA  :py:meth:`~alphastats.DataSet_Statistics.Statistics.ancova`
* Tukey - test :py:meth:`~alphastats.DataSet_Statistics.Statistics.tukey_test`



GO Analysis
----------------------
The GO Analysis uses the API from `aGOtool <https://agotool.org/>`_.

* Characterize foreground without performing a statistical test: :py:meth:`~alphastats.DataSet_Pathway.Enrichment.go_characterize_foreground`
* Gene Ontology Enrichment Analysis with abundance correction: :py:meth:`~alphastats.DataSet_Pathway.Enrichment.go_abundance_correction`
* Gene Ontology Enrichment Analysis without abundance correction: :py:meth:`~alphastats.DataSet_Pathway.Enrichment.go_compare_samples`
* Gene Ontology Enrichement Analysis using a Background from UniProt Reference Proteomes: :py:meth:`~alphastats.DataSet_Pathway.Enrichment.go_genome`

**Visualization of GO Analysis results**

All GO-analysis functions will return a DataFrame with the results.

* Plot Scatterplot with -log10(p-value) on x-axis and effect size on y-axis. `df.plot_scatter()`
* Plot p-values as Barplot `df.plot_bar`
