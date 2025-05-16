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

If you receive an error like `library 'hdf5' not found`, your computer is missing the HDF5 library. Install it via your favorite package manager or use `conda create --name alphastats python=3.9 hdf5`.
Alternatively, use ```conda install -c anaconda pytables```.
