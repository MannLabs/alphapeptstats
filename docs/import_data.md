# Data import


Currently, AlphaStats allows the analysis of five quantitative proteomics software packages: AlphaPept, DIA-NN, FragPipe, MaxQuant and Spectronaut. As the output of these software differs significantly data needs to be loaded in customized loaders.

Imported proteomics data and metadata can be combined in a DataSet, which will be used for the downstream analysis.

```python
import alphastats

maxquant_data = alphastats.MaxQuantLoader(
    file="testfiles/maxquant_proteinGroups.txt"
)

dataset = alphastats.dataset.dataset.DataSet(
    loader = maxquant_data,
    metadata_path_or_df="../testfiles/maxquant/metadata.xlsx",
    sample_column="sample"
)
```


## Importing data from a Proteomics software
To import data from these software the columns describing intensity and Protein/ProteinGroups have to be specified when loading. Each loader has a default for the `intensity_column`and the `index_column`, however, the column naming can vary depending on the version of the software and personalized settings.

As we are dealing with wide data, a column represents the intensity for one sample. Thus the `intensity_column` must be specified as follow: For MaxQuant `"LFQ intensity [sample]"` or for AlphaPept `"[sample]_LFQ"` (this is already set as default).

Upon data import, the proteomics data gets processed in an internal format.

### Additional modifications by AlphaStats

When importing the data, AlphaStats will identify potential contaminations based on a contaminant library, created by [Frankenfield et al. 2022](https://www.biorxiv.org/content/10.1101/2022.04.27.489766v2.full). This information will be added as an extra column to the imported data and can either be ignored or used for filtering in the preprocessing step.


### AlphaPept
[Alphapept](https://github.com/MannLabs/alphapept) output can either be imported as `results_proteins.csv` or `results.hdf`.

**Intensity types**
AlphaPept either described the raw intensity or the free quantification (LFQ) intensity. By default AlphaStats uses the LFQ-Intensity for the downstream analysis.

Further, AlphaStats will identify "Reverse" - Proteins.

Find more details about the file format [here](https://mannlabs.github.io/alphapept/file_formats.html).

```python
import alphastats
alphapept_data = alphastats.AlphaPeptLoader(file="testfiles/alphapept_results_proteins.csv")
```

### MaxQuant
[MaxQuant](https://www.maxquant.org/) generates multiple files as output. For the downstream analysis the `proteinGroups.txt` file, containing the aggregated protein intensities is sufficient.

**Intensity types**
MaxQuant annotates different intensity types: raw intensity, label - free quantification (LFQ) intensity and intensity-based absolute quantification (iBAQ) intensity. The default settings are "LFQ intensity [sample]".

Find more details about the file format [here](http://www.coxdocs.org/doku.php?id=maxquant:table:proteingrouptable)

```python
import alphastats
maxquant_data = alphastats.MaxQuantLoader(file="testfiles/maxquant_proteinGroups.txt")
```

### DIA-NN
For the analysis of [DIA-NN](https://github.com/vdemichev/DiaNN) output use `report_final.pg_matrix.tsv`. Versions before 1.7. are not supported.


Find more details about the file format [here](https://github.com/vdemichev/DiaNN#output).

```python
import alphastats
diann_data = alphastats.DIANNLoader(file="testfiles/diann_report_final.pg_matrix.tsv")
```

### FragPipe

Find more details about the file format [here](https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv).

```python
import alphastats
fragpipe_data = alphastats.FragPipeLoader(file="testfiles/fragpipe_combined_proteins.tsv")
```

### Spectronaut

Find more details about the file format [here](https://biognosys.com/content/uploads/2022/12/Spectronaut17_UserManual.pdf).


As default alphastats will use "PG.ProteinGroups" and "PG.Quantity" for the analysis. For an ananlysis on a peptide level the "F.PeakArea" and the peptide sequences ("PEP.StrippedSequence") can be used.

```python
import alphastats
spectronaut_data = alphastats.SpectronautLoader(
    file="testfiles/spectronaut/results.tsv",
    intensity_column = "F.PeakArea",
    index_column = "PEP.StrippedSequence"
    )
```

### mzTab

Find more details about the file format [here](https://www.psidev.info/mztab).

```python
import alphastats
mztab_data = alphastats.mzTabLoader(
    file="testfiles/mztab/test.mztab"
    )
```


## Preparing metadata

To compare samples across various conditions in the downstream analysis, a metadata file in form of a table (excel, csv, tsv) is required. This file should contain a column with the sample IDs (raw file names) matching the sample names annotated in the output file of your proteomics software. Further, information can be provided like disease and various clinical parameters. Examples of metadata files can be found in the [testfiles-folder](https://github.com/MannLabs/alphastats/tree/main/testfiles).


## Creating a DataSet

The whole downstream analysis can be performed on the alphastats.dataset.dataset.DataSet. To create the DataSet you need to provide the loader object as well as the metadata.

```python
import alphastats

maxquant_data = alphastats.MaxQuantLoader(
    file="testfiles/maxquant_proteinGroups.txt"
)

dataset = alphastats.dataset.dataset.DataSet(
    loader = maxquant_data,
    metadata_path_or_df="../testfiles/maxquant/metadata.xlsx",
    sample_column="sample"
)
```
