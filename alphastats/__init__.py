__project__ = "alphastats"
__version__ = "0.4.2"
__license__ = "Apache"
__description__ = "An open-source Python package for Mass Spectrometry Analysis"
__author__ = "Mann Labs"
__author_email__ = "elena.krismer@hotmail.com"
__github__ = "https://github.com/MannLabs/alphapeptstats"
__keywords__ = [
    "bioinformatics",
    "software",
    "mass spectometry",
]
__python_version__ = ">=3.8,<4"
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
__urls__ = {
    "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
    "GitHub": __github__,
    "ReadTheDocs": "https://mannlabs.github.io/alphapeptstats/",
    "PyPi": "https://pypi.org/project/alphastats/"
    # "Scientific paper": None,
}
__console_scripts__ = [
    "alphastats=alphastats.cli:run",
]
__extra_requirements__ = {
    "development": "requirements_development.txt",
}

from .loader.AlphaPeptLoader import *
from .loader.DIANNLoader import *
from .loader.FragPipeLoader import *
from .loader.MaxQuantLoader import *
from .DataSet import *
from .cli import *
import alphastats.gui
