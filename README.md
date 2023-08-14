[![PyPI version](https://badge.fury.io/py/alphastats.svg)](https://badge.fury.io/py/alphastats)
[![codecov](https://codecov.io/gh/MannLabs/alphastats/branch/main/graph/badge.svg?token=HY4A0KKLRI)](https://codecov.io/gh/MannLabs/alphastats)
[![Downloads](https://static.pepy.tech/badge/alphastats)](https://pepy.tech/project/alphastats)
[![Downloads](https://static.pepy.tech/badge/alphastats/week)](https://pepy.tech/project/alphastats)
[![CI](https://github.com/MannLabs/alphapeptstats/actions/workflows/python-package.yml/badge.svg)](https://github.com/MannLabs/alphapeptstats/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/alphapeptstats/badge/?version=latest)](https://alphapeptstats.readthedocs.io/en/latest/?badge=latest)


<div align = center>
<img src="https://github.com/MannLabs/alphapeptstats/blob/main/misc/alphastats_workflow.png?raw=true" width="771.4" height="389.2">
</div>


<div align = center>
<br>
<br>

[<kbd> <br> Documentation <br> </kbd>][link]

</div>

<br>
<br>

[link]:https://alphapeptstats.readthedocs.io/en/main/

<div align = center>
<br>
<br>

[<kbd> <br> Streamlit WebApp <br> </kbd>][link_streamlit]

</div>

<br>
<br>

[link_streamlit]:https://mannlabs-alphapeptstats-alphastatsguialphapeptstats-qyzgwd.streamlit.app/

An open-source Python package for downstream mass spectrometry downstream data analysis from the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/).


* [**Citation**](#citation)
* [**Installation**](#installation)
* [**Troubleshooting**](#troubleshooting)
* [**License**](#license)
* [**How to contribute**](#how-to-contribute)
* [**Changelog**](#changelog)

---
## Citation
Publication: [AlphaPeptStats: an open-source Python package for automated and scalable statistical analysis of mass spectrometry-based proteomics](https://doi.org/10.1093/bioinformatics/btad461)
> **Citation:** <br>
> Krismer, E., Bludau, I.,  Strauss M. & Mann M. (2023). AlphaPeptStats: an open-source Python package for automated and scalable statistical analysis of mass spectrometry-based proteomics. Bioinformatics 
> https://doi.org/10.1093/bioinformatics/btad461

---
## Installation

AlphaPeptStats can be used as 
 * python library (pip-installation), or 
 * Graphical User Interface (either pip-installation or one-click installer). 
 
Further we provide a Dockerimage for the GUI.

### Pip Installation

AlphaStats can be installed in an existing Python 3.8/3.9/3.10 environment with a single `bash` command. 

```bash
pip install alphastats
```

In case you want to use the Graphical User Interface, use following command in the command line:
 
```bash
alphastats gui
```

AlphaStats can be imported as a Python package into any Python script or notebook with the command `import alphastats`.
A brief [Jupyter notebook tutorial](nbs/getting_started.ipynb) on how to use the API is also present in the [nbs folder](nbs).


### One Click Installer

One click Installer for MacOS, Windows and Linux can be found [here](https://github.com/MannLabs/alphapeptstats/releases).


### Docker Image

We provide two Dockerfiles, one for the library and one for the Graphical User Interface.
The Image can be pulled from Dockerhub

```bash
docker pull elenakrismer/alphapeptstats_streamlit
```

---
## GUI
![](https://github.com/MannLabs/alphapeptstats/blob/main/misc/volcano.gif)

---
## Troubleshooting

In case of issues, check out the following:

* [Issues](https://github.com/MannLabs/alphapeptstats/issues): Try a few different search terms to find out if a similar problem has been encountered before

---
## License

AlphaStats was developed by the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/) and is freely available with an [Apache License](LICENSE.txt). External Python packages (available in the [requirements](requirements) folder) have their own licenses, which can be consulted on their respective websites.

---
## How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/alphapeptstats/stargazers) to boost our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/alphapeptstats/issues) or clone the repository and create a [pull request](https://github.com/MannLabs/alphapeptstats/pulls) with a new branch. For an even more interactive participation, check out the [discussions](https://github.com/MannLabs/alphapeptstats/discussions) and the [the Contributors License Agreement](misc/CLA.md).

---
## Changelog

See the [HISTORY.md](HISTORY.md) for a full overview of the changes made in each version.


---
## FAQ

### How can I resolve the Microsoft visual error message when installing: error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"?
Please, find a description on how to update required tools [here](https://github.com/MannLabs/alphapeptstats/issues/158).

## How to resolve ERROR:: Could not find a local HDF5 installation. on Mac M1?

Before installing AlphaPeptStats you might need to install pytables first:

````
conda install -c anaconda pytables
````
