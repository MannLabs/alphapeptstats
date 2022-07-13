#!python
#from pkgutil import ImpImporter
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from .loader.DIANNLoader import DIANNLoader
from .loader.FragPipeLoader import *
from alphastats.loader.MaxQuantLoader import MaxQuantLoader 
from .DataSet import *

__project__ = "alphastats"
__version__ = "0.0.1"
__license__ = "Apache"
__description__ = "An open-source Python package for Mass Spectrometry Analysis"
__author__ = "Mann Labs"
__author_email__ = "opensource@alphastats.com"
__github__ = "https://github.com/MannLabs/alphastats"
__keywords__ = [
    "bioinformatics",
    "software",
    "mass spectometry",
]
__python_version__ = ">=3.8,<3.10"
__classifiers__ = [
    "Development Status :: 1 - Planning",
    # "Development Status :: 2 - Pre-Alpha",
    # "Development Status :: 3 - Alpha",
    # "Development Status :: 4 - Beta",
    # "Development Status :: 5 - Production/Stable",
    # "Development Status :: 6 - Mature",
    # "Development Status :: 7 - Inactive"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__console_scripts__ = [
    "alphastats=alphastats.cli:run",
]
__urls__ = {
    "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
    "GitHub": __github__,
    # "ReadTheDocs": None,
    # "PyPi": None,
    # "Scientific paper": None,
}
__extra_requirements__ = {
    "development": "requirements_development.txt",
}
