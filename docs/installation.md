## Installation

### Pip

For any Python package, it is highly recommended to use a separate conda virtual environment, as otherwise dependancy conflicts can occur with already existing packages.

```bash
conda create --name alphastats
conda activate alphastats
```

AlphaStats can be installed in an existing Python 3.8 environment with a single `bash` command. *This `bash` command can also be run directly from within a Jupyter notebook by prepending it with a `!`*:

```bash
pip install alphastats
```