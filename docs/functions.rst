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

* Principal Component Analysis (PCA): :py:meth:`~alphastats.DataSet_Plot.Plot.plot_pca`
* t-SNE: :py:meth:`~alphastats.DataSet_Plot.Plot.plot_tsne`

**Plot Distance between samples**

* Plot correlation matrix :py:meth:`~plot_correlation_matrix`
* Plot Dendogram :py:meth:`~Plot.plot_dendogram`
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

.. code-block:: 
    
    plot = DataSet.plot_volcano(column = "disease", group1 = "healthy", group2 = "cancer")
    plot.write_image("images/volcano_plot.svg")


Statistical Analysis
----------------------

* Perform Differential Expression Analysis with a Wald test using `diffxpy <https://github.com/theislab/diffxpy>`_.  :py:meth:`~alphastats.DataSet_Statistics.Statistics.perform_diff_expression_analysis`
* ANOVA  :py:meth:`~alphastats.DataSet_Statistics.Statistics.anova`
* ANCOVA  :py:meth:`~alphastats.DataSet_Statistics.Statistics.ancova`
* Tukey - test :py:meth:`~alphastats.DataSet_Statistics.Statistics.calculate_tukey`
* T-test :py:meth:`~alphastats.DataSet_Statistics.Statistics.calculate_ttest_fc`


Misc 
------

Get an overview over your dataset

* :py:meth:`~alphastats.DataSet.overview`

* :py:meth:`~alphastats.DataSet_Preprocess.Preprocess.preprocess_print_info`


