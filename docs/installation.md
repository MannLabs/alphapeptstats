## Installation

### Pip

For any Python package, it is highly recommended to use a separate conda virtual environment, as otherwise, dependency conflicts can occur with already existing packages.

```bash
conda create --name alphastats python=3.8
conda activate alphastats
```

AlphaStats can be installed in an existing Python 3.8 environment with a single `bash` command.
```bash
pip install alphastats
```

#### MacOS M1
On M1 Mac the installation initally might fail due to a missing local HDF5 installation. To solve this issue install pytables manually:

```bash
conda install -c anaconda pytables
```
