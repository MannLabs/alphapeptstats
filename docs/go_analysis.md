#  GO Analysis

Gene Ontology (GO) Enrichement Analysis are widely used to analysis omics-data sets to identify significant enriched GO terms. AlphaStats utilizes [aGOtool](https://agotool.org/) for enrichement analysis. aGOtool is especially tailored for Mass Spectrometry (MS) based proteomics data. This tool considers the fact that post-translational modifications (PTMs) are more likely to be detected on high abundant proteins than on low abundant proteins. The functional enrichement is performed for GO (molecular function, biological process, cellular component), UniProt keywords, KEGG pathways, PubMed publications, Reactome, Wiki Pathways, Interpro domains, PFAM domains, Brenda Tissues and Diseases.
The bias correction by aGOtool aims for an increased specificity, fewer significantly enriched but more biologically meaningful and accurate enrichement terms [Schölz et al. 2015](https://doi.org/10.1038/nmeth.3621).


The implementation of aGOtool in AlphaStats will allow you to perform following analysis:
- **Abundance Correction**: Compares two samples (for example healthy vs. controls). As foreground all positively associated proteins of the foreground are used. For the background postively associated proteins and their intensity of the background are used.
- **Characterize Foreground**: Display functional annotations of your Protein(s) of interest without performing a statistical test.
- **Compare Samples**: GO Enrichement ANalysis without abundance correction.
- **Genome**: GO Enrichement Analysis using a Background from UniProt Reference Proteomes.

All functions will return a pandas DataFrame with the results.

### Requirements

A GO Enrichement Analysis using Proteomics data is usally performed on a list of proteins with specific PTMs. Currently, AlphaStats offers the option to load the *evidence.txt* file from *MaxQuant*. This file will be used to extract proteins with PTMs when performing a GO analysis.
In case there is no information about PTMs available, a list of upregulated proteins in form of UniProt protein accession numbers can be passed to the functions.

More details about the GO Analysis can be found here:
 - [aGOtool](https://agotool.org/)
 - Publication: [Schölz et al. 2015](https://doi.org/10.1038/nmeth.3621)



