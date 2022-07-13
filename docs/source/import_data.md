# Data import

Currently AlphaStats allows the analysis of four quantitative proteomics software packages: AlphaPept, DIA-NN, FragPipe and MaxQuant. As the output of these softwares differs signficantly data needs to be loaded in customized loaders.

## Importing data
In order to import data from these softwares the columns describing intensity and Protein/ProteinGroups have to specified when loading. Each loader has a default for the `intensity_column`and the `index_column`, however the column naming can vary depending on the version of the software and personalized settings.

As we are dealing with wide data, a column represents the intensity for one sample. Thus the `intensity_column` must be specified as follow: For MaxQuant `"LFQ intensity [sample]"` or for AlphaPept `"[sample]_LFQ"` (this is already set as default).

## Additional modifications by AlphaStats
When importing the data, AlphaStats will itendify potential contaminations based on a contaminant library, reated by [Frankenfield et al. 2022](https://www.biorxiv.org/content/10.1101/2022.04.27.489766v2.full). This information will be added as extra column to the imported data and can either be ignored or used for filtering in the preprocessing step.

## Alphapept
[Alphapept](https://github.com/MannLabs/alphapept) output can either be imported as `results_proteins.csv` or `results.hdf`. 

Find more details about the file format [here](https://mannlabs.github.io/alphapept/file_formats.html).

```python
from alphastats import AlphaPeptLoader
alphapept_data = AlphaPeptLoader(file="testfiles/alphapept_results_proteins.csv")
```

## MaxQuant
[MaxQuant](https://www.maxquant.org/) generates multiple files as output. For the downstream anaylsis the `proteinGroups.txt` file, containing the aggregated protein intensities is sufficient. 

Find more details about the file format [here](http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable)

```python
from alphastats import MaxQuantPeptLoader
maxquant_data = MaxQuantLoader(file="testfiles/maxquant_proteinGroups.txt")
```

## DIA-NN
For the analysis of [DIA-NN](https://github.com/vdemichev/DiaNN) output use `report_final.pg_matrix.tsv`. Versions before 1.7. are not supported.

Find more details about the file format [here](https://github.com/vdemichev/DiaNN#output).

```python
from alphastats import DIANNLoader
diann_data = DIANNLoader(file="testfiles/diann_report_final.pg_matrix.tsv")
```

## FragPipe

Find more details about the file format [here](https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv).

```python
from alphastats import FragPipeLoader
fragpipe_data = FragPipeLoader(file="testfiles/fragpipe_combined_proteins.tsv")
```


