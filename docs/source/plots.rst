Figures
=================

To generate interactive plots, AlphaStats uses the graphing library `Plotly <https://plotly.com/python/>`_ 
and all plotting methods will return a plotly object. 
The plotly graphs returned by AlphaStats can be customized.
A description how you can customize your plots can be found `here <https://maegul.gitbooks.io/resguides-plotly/content/content/plotting_locally_and_offline/python/methods_for_updating_the_figure_or_graph_objects.html>`_


**Plot Intensity**

* Plot Intensity for indiviual Protein/ProteinGroup :py:meth:`~alphastats.DataSet_Plot.Plot.plot_intensity`
* Plot Intensity distribution for each sample  :py:meth:`~alphastats.DataSet_Plot.Plot.plot_sampledistribution`


**Dimensionality reduction plots:**

* Principal Component Analysis (PCA): :py:meth:`~alphastats.DataSet_Plot.Plot.plot_pca`
* t-SNE: :py:meth:`~alphastats.DataSet_Plot.Plot.plot_tsne`

**Plot Distance between samples**

* Plot correlation matrix :py:meth:`~alphastats.DataSet_Plot.Plot.plot_correlation_matrix`
* Plot Dendogram :py:meth:`~alphastats.DataSet_Plot.Plot.plot_dendogram`