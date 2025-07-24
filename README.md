[![PyPI version](https://badge.fury.io/py/alphastats.svg)](https://badge.fury.io/py/alphastats)
[![codecov](https://codecov.io/gh/MannLabs/alphastats/branch/main/graph/badge.svg?token=HY4A0KKLRI)](https://codecov.io/gh/MannLabs/alphastats)
[![Downloads](https://static.pepy.tech/badge/alphastats)](https://pepy.tech/project/alphastats)
[![Downloads](https://static.pepy.tech/badge/alphastats/week)](https://pepy.tech/project/alphastats)
[![CI](https://github.com/MannLabs/alphapeptstats/actions/workflows/python-package.yml/badge.svg)](https://github.com/MannLabs/alphapeptstats/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/alphapeptstats/badge/?version=latest)](https://alphapeptstats.readthedocs.io/en/latest/?badge=latest)


An open-source Python package for downstream mass spectrometry downstream data analysis from the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/).

![](https://github.com/MannLabs/alphapeptstats/blob/main/misc/volcano.gif)

=> [Run the app right now in your browser](https://mannlabs-alphapeptstats-alphastatsguialphapeptstats-qyzgwd.streamlit.app/)



---
## Installation

AlphaPeptStats can be used as
 * python library (pip-installation), or
 * Graphical User Interface (either pip-installation or one-click installer), or
 * as a Docker container.


### One Click Installer

One click Installer for MacOS, Windows and Linux can be found [here](https://github.com/MannLabs/alphapeptstats/releases).

#### Windows
Download the latest `alphastats-X.Y.Z-windows-amd64.exe` build and double click it to install. If you receive a warning during installation click *Run anyway*.
Important note: always install AlphaPeptStats into a new folder, as the installer will not properly overwrite existing installations.

#### Linux
Download the latest `alphastats-X.Y.Z-linux-x64.deb` build and install it via `dpkg -i alphastats-X.Y.Z-linux-x64.deb`.

#### MacOS
Download the latest build suitable for your chip architecture
(can be looked up by clicking on the Apple Symbol > *About this Mac* > *Chip* ("M1", "M2", "M3" -> `arm64`, "Intel" -> `x64`),
`alphastats-X.Y.Z-macos-darwin-arm64.pkg` or `alphastats-X.Y.Z-macos-darwin-x64.pkg`. Open the parent folder of the downloaded file in Finder,
right-click and select *open*. If you receive a warning during installation click *Open*.

In newer MacOS versions, additional steps are required to enable installation of unverified software.
This is indicated by a dialog telling you `“alphastats. ... .pkg” Not Opened`.
1. Close this dialog by clicking `Done`.
2. Choose `Apple menu` > `System Settings`, then `Privacy & Security` in the sidebar. (You may need to scroll down.)
3. Go to `Security`, locate the line "alphadia.pkg was blocked to protect your Mac" then click `Open Anyway`.
4. In the dialog windows, click `Open Anyway`.

### Pip Installation

AlphaStats can be installed in an existing Python >=3.9 environment with a single `bash` command.

```bash
pip install alphastats
```

In case you want to use the Graphical User Interface, use following command in the command line:

```bash
alphastats gui
```
If you get an `AxiosError: Request failed with status code 403'` when uploading files, try running `DISABLE_XSRF=1 alphastats gui`.

If you receive an error like `library 'hdf5' not found`, your computer is missing the HDF5 library. Install it via your favorite package manager or use `conda create --name alphastats python=3.9 hdf5`.
Alternatively, use ```conda install -c anaconda pytables```.

AlphaStats can be imported as a Python package into any Python script or notebook with the command `import alphastats`.
A brief [Jupyter notebook tutorial](nbs/getting_started.ipynb) on how to use the API is also present in the [nbs folder](nbs).


### Docker version
The containerized version can be used to run alphapeptstats without any installation (apart from Docker)

### 1. Setting up Docker
Install the latest version of docker (https://docs.docker.com/engine/install/).

### 2. Start the container
```bash
PORT=8501
SESSIONS_PATH=./sessions
docker run -p $PORT:8501 -v $SESSIONS_PATH:/app/sessions mannlabs/alphastats:latest
```
After initial download of the container, alphapeptstats will start running on [http://localhost:$PORT](http://localhost:8501).
Note: this will create a directory `$SESSIONS_PATH` where sessions will be stored.

## API Documentation
AlphaPeptStats provides an extensive API [documentation](https://alphapeptstats.readthedocs.io/en/main/).


---
## Troubleshooting

In case of issues, check out the following:

* [Issues](https://github.com/MannLabs/alphapeptstats/issues): Try a few different search terms to find out if a similar problem has been encountered before

### Common problems

#### How to resolve " error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools" " ?
Please, find a description on how to update required tools [here](https://github.com/MannLabs/alphapeptstats/issues/158).

#### How to resolve "ERROR: Could not find a local HDF5 installation" on Mac Silicon (M1/M2/M3)?

Before installing AlphaPeptStats you might need to install pytables first:

````
conda install -c anaconda pytables
````


---
## License

AlphaStats was developed by the [Mann Group at the University of Copenhagen](https://www.cpr.ku.dk/research/proteomics/mann/) and is
now maintenained by the [Mann Group at the MPI Biochemistry](https://www.biochem.mpg.de/mann). It is
freely available with an [Apache License](LICENSE.txt). External Python packages have their own
licenses, which can be consulted on their respective websites.

---
## How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/alphapeptstats/stargazers) to boost our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/alphapeptstats/issues) or clone the repository and create a [pull request](https://github.com/MannLabs/alphapeptstats/pulls) with a new branch. For an even more interactive participation, check out the [discussions](https://github.com/MannLabs/alphapeptstats/discussions) and the [Contributors License Agreement](misc/CLA.md).


### Notes for developers

#### Tagging of changes
In order to have release notes automatically generated, changes need to be tagged with labels.
The following labels are used (should be safe-explanatory):
`breaking-change`, `bug`, `enhancement`.

#### Release a new version
This package uses a shared release process defined in the
[alphashared](https://github.com/MannLabs/alphashared) repository. Please see the instructions
[there](https://github.com/MannLabs/alphashared/blob/reusable-release-workflow/.github/workflows/README.md#release-a-new-version).

#### pre-commit hooks
It is highly recommended to use the provided pre-commit hooks, as the CI pipeline enforces all checks therein to
pass in order to merge a branch.

The hooks need to be installed once by
```bash
pip install -r requirements_dev.txt
pre-commit install
```
You can run the checks yourself using:
```bash
pre-commit run --all-files
```

##### The `detect-secrets` hook fails
This is because you added some code that was identified as a potential secret.
1. Run `detect-secrets scan --exclude-files testfiles --exclude-lines '"(hash|id|image/\w+)":.*' > .secrets.baseline`
(check `.pre-commit-config.yaml` for the exact parameters)
2. Run `detect-secrets audit .secrets.baseline` and check if the detected 'secret' is actually a secret
3. Commit the latest version of `.secrets.baseline`



---
## Changelog

See the [GitHub Release Notes](https://github.com/MannLabs/alphapeptstats/releases) for changes from version 0.6.8 on,
[HISTORY.md](HISTORY.md) for older versions.


---
## Citation
Publication: [AlphaPeptStats: an open-source Python package for automated and scalable statistical analysis of mass spectrometry-based proteomics](https://doi.org/10.1093/bioinformatics/btad461)
> **Citation:** <br>
> Krismer, E., Bludau, I.,  Strauss M. & Mann M. (2023). AlphaPeptStats: an open-source Python package for automated and scalable statistical analysis of mass spectrometry-based proteomics. Bioinformatics
> https://doi.org/10.1093/bioinformatics/btad461
