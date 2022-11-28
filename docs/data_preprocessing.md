# Data preprocessing

Preprocessing of the data is substantial for the results of the downstream analysis.
Data preprocessing available in AlphaStats covers: the removal of contaminants, normalization, imputation and subsetting and removal of samples.

```python
DataSet.preprocess(
        remove_contaminations,
        subset,
        normalization,
        imputation,
        remove_samples
)
```

## Removal of contaminations

- Should I remove contaminations/ `remove_contaminations=True`? Generally speaking - Yes (source)

Various proteomics software annotates contaminants differently or not at all. MaxQuant describes contaminations or ProteinGroups that should be removed in `Only identified by site`, `Reverse`, `Potential contaminant`. Likewise MaxQuant, AlphaPept flags spurious proteins as `Reverse`. 

In addition, AlphaStats identifies contaminations based on the contamination library created by [Frankenfield et al. 2022](https://www.biorxiv.org/content/10.1101/2022.04.27.489766v2.full).


## Normalization

Depending on the software and the settings, data could already have been normalized before loading into AlphaStats.

AlphaStats has the following Normalization methods implemented:

 - **Z-Score Normalization (Standardization)**: Centers the protein intensity of each sample, meaning scaling the variance to 1.
 
 - **Quantile Normalization**: Aims to correct technical bias by adjusting the distribution of protein intensities for each sample. This normalization method is suitable when it is assumed that only a small portion of the protein expression varies among certain conditions, while the majority of the proteome remains stable ([Dubois et al., 2022](https://doi.org/10.1016/j.biosystems.2022.104661) ).
 
 - **Linear Normalization** 
 - **Variance Stabilization Transformation**


> **Note**
> It has been shown that normalizing the data first and then imputing the data performs better, than the other way around 
([Karpievitch et al. 2012](https://doi.org/10.1186/1471-2105-13-S16-S5)). This preprocessing order is also acquired in
AlphaStats (unless preprocessing is done in several steps).


## Imputation

Especially, missing values are challenging when it comes to analyzing proteomic mass spectrometry data. Missing values can either be missing completely at random (MCAR), due to technical limitations or missing not at random (MNAR) meaning that the abundance is below the detection limit of the platform or completely absent.
 
To deal with missing values, AlphaStats has the following methods implemented:

- **k-nearest neighbors (kNN)**: Missing values are imputed using the mean protein intensity of k-nearest neighbors

- **Random Forest**: Applies the machine learning algorithm random forest and predicts the values of the target variable using specific known target variables as the outcome and other variables as predictors.

- **median**: Replaces missing values using the median of each protein.

- **mean**: Replaces missing values using the mean of each protein.

Overall, random forest-based imputation for mass spectrometry data has shown a high performance among several studies compared to other imputations methods([Kokla et al. 2019](https://doi.org/10.1186/s12859-019-3110-0), [Jin et al. 2021](https://doi.org/10.1038/s41598-021-81279-4)). However, when applying random forest imputation to your dataset, you have to expect a long run time.



## Subset Data

In case the proteomics data contains more samples than the metadata, the proteomics data can be filtered based on the samples present in the metadata using `DataSet.preprocess(subset=True)`.

## Remove Samples

If you want to remove samples from your `DataSet`, outliers for instance you can give a list of sample names `DataSet.preprocess(remove_samples=["sample1", "sample3"])`.


