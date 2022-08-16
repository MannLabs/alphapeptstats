[![codecov](https://codecov.io/gh/MannLabs/alphastats/branch/main/graph/badge.svg?token=HY4A0KKLRI)](https://codecov.io/gh/MannLabs/alphastats)

![Pip installation](https://github.com/MannLabs/alphastats/workflows/Default%20installation%20and%20tests/badge.svg)

# AlphaStats
An open-source Python package of the AlphaStats ecosystem from the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann). To enable all hyperlinks in this document, please view it at [GitHub](https://github.com/MannLabs/alphastats).

* [**About**](#about)
* [**License**](#license)
* [**Installation**](#installation)
  * [**Pip installer**](#pip)
* [**Usage**](#usage)
  * [**Python and jupyter notebooks**](#python-and-jupyter-notebooks)
* [**Troubleshooting**](#troubleshooting)
* [**Citations**](#citations)
* [**How to contribute**](#how-to-contribute)
* [**Changelog**](#changelog)

---
## About
An open-source Python package for downstream mass spectrometry data analysis from the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann).

---
## License

AlphaStats was developed by the [Mann Group, Novo Nordisk Foundation Center for Protein Research](https://www.cpr.ku.dk/research/proteomics/mann/) and is freely available with an [Apache License](LICENSE.txt). External Python packages (available in the [requirements](requirements) folder) have their own licenses, which can be consulted on their respective websites.

---
## Installation

### Pip

AlphaStats can be installed in an existing Python 3.8 environment with a single `bash` command. *This `bash` command can also be run directly from within a Jupyter notebook by prepending it with a `!`*:

```bash
pip install alphastats
```

Next, download the AlphaStats repository from GitHub either directly or with a `git` command. This creates a new alphastats subfolder in your current directory.

```bash
git clone https://github.com/MannLabs/alphastats.git
```

For any Python package, it is highly recommended to use a separate [conda virtual environment](https://docs.conda.io/en/latest/), as otherwise *dependancy conflicts can occur with already existing packages*.

```bash
conda create --name alphastats python=3.8 -y
conda activate alphastats
```

Finally, AlphaStats and all its [dependancies](requirements) need to be installed. To take advantage of all features and allow development (with the `-e` flag), this is best done by also installing the [development dependencies](requirements/requirements_development.txt) instead of only the [core dependencies](requirements/requirements.txt):

```bash
pip install -e "./alphastats[development]"
```

---
## Usage


### Python and Jupyter notebooks

AlphaStats can be imported as a Python package into any Python script or notebook with the command `import alphastats`.

A brief [Jupyter notebook tutorial](nbs/workflow_mq.ipynb) on how to use the API is also present in the [nbs folder](nbs).

---
## Troubleshooting

In case of issues, check out the following:

* [Issues](https://github.com/MannLabs/alphastats/issues): Try a few different search terms to find out if a similar problem has been encountered before
* [Discussions](https://github.com/MannLabs/alphastats/discussions): Check if your problem or feature requests has been discussed before.

---
## Citations

There are currently no plans to draft a manuscript.

---
## How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/alphastats/stargazers) to boost our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/alphastats/issues) or clone the repository and create a [pull request](https://github.com/MannLabs/alphastats/pulls) with a new branch. For an even more interactive participation, check out the [discussions](https://github.com/MannLabs/alphastats/discussions) and the [the Contributors License Agreement](misc/CLA.md).

---
## Changelog

See the [HISTORY.md](HISTORY.md) for a full overview of the changes made in each version.
