# Changelog

# 0.6.5
* FIX coloring of plot_sampledistribution issue #229
* ADD nbformat requirement issue #230

# 0.6.4
* FIX windows installer

# 0.6.3
* ENH download metadata template in the GUI
* ENH multicova analysis
* ENH filter data completeness `dataset.preprocess(data_completeness=0.7)` 
* ADD `GenericLoader` for not supported data formats

# 0.6.2
* FIX preprocessing with VST floats and inf
* FIX plotly display

# 0.6.1
* FIX data loading

# 0.6.0
* ADD mzTAB support
* ENH color Volcano Plot data points using list of protein names `color_list=your_protein_list`


# 0.5.4
* FIX altair version - binning of streamlit version

# 0.5.3
* FIX FragPipe loading issue

# 0.5.2
* FIX FragPipe import #173

# 0.5.1
* ENH increase limit uplaod size in GUI

# 0.5.0
* ADD plot Sample Distribution Histogram
* ADD paired-ttest option 
* ENH add option to remove samples in GUI
* ADD pyComBat Batch correction Behdenna A, Haziza J, Azencot CA and Nordor A. (2020) pyComBat, a Python tool for batch effects correction in high-throughput molecular data using empirical Bayes methods. bioRxiv doi: 10.1101/2020.03.17.995431
* remove iteration utilities, remove C++ dependency
* ENH upgrade to pandas 2.0.0

# 0.4.5
* FIX loading of Data on Windows

# 0.4.4
* FIX one click installer

# 0.4.3
* FIX loading dataset with columns
* ADD log2-transformation

# 0.4.2
* ADD option compare_preprocessing_modes
* update to streamlit 1.19 with new caching functions

# 0.4.1
* FIX bug when drawing FDR lines
* ADD functionionality `dataset.reset_preprocessing()` to reset all preprocessing steps

# 0.4.0
* SAM implementation from Isabel (MultiCova) 
* Volcano Plot with permutation based FDR line
* Bug fix when reseting matrix and saving info

# 0.3.0
* One Click Installer for macOS, Linux and Windows
* Spectronaut support

# 0.2.8
* enhance performance of plotting VolanoPlots in the GUI
* refactoring of plotting classes
* BUG fix when importing new dataset in GUI

## 0.2.7
* Fix warning when loading DIA-NN GUI

## 0.2.6
* Bug fix when importing data in GUI
* version binning

## 0.2.5
* DataSet overview tab
* allow download of data matrix in GUI

## 0.2.4
* added kaleido dependency
* GUI add button to reset preprocessing
* GUI combine visualization and analysis
* Display seaborn clustermap in GUI
* GUI fixes

## 0.2.3
* fix umap import
* version report in GUI
* Bug fix GUI

## 0.2.2
* rename AlphaStats to AlphaPeptStats
* GUI bug fix

## 0.2.1
* fix requirements
* function renaming perform_diff_expr_analysis > perform_diff_expr_analysis, calculate_tukey -> tukey_test

## 0.2.0
* Gene Ontology Enrichment Analysis using a GO tool

## 0.1.2
* rawdata renamed to rawinput
* GUI bug fix

## 0.1.1
* Graphical User Interface
* Volcano Plot can be plotted with cut off lines, add labels to significant enriched proteins
* plot-intensity with t-test between two groups
* Differential Expression analysis + Volcano Plot can be calcultated/plotted with list of sample names
* UMAP

## 0.0.6
* remove dependency for Microsoft Visual Studio C++ (dash_bio) (Windows PCs)
* clustermap with seaborn, adding bar labels is possible
* save plotting data, preprocessing, method in figure object (accessible using figure.plotting_data, figure.method, figure.preprocessing)















