# builtin
import setuptools
import re
import os

# local
# info has to be imported individually or sidepacakges will already be installed
# from alphastats import __project__, __version__, __license__, __description__,__author__,__author_email__
# from alphastats import __github__, __keywords__, __python_version__, __classifiers__, __urls__, __extra_requirements__, __console_scripts__


def get_long_description():
    with open("README.md", "r") as readme_file:
        long_description = readme_file.read()
    return long_description


def get_requirements():
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    return required
    

def create_pip_wheel():
    requirements = get_requirements()
    setuptools.setup(
        name="alphastats",
        version="0.4.2",
        license="Apache",
        description="An open-source Python package for Mass Spectrometry Analysis",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Mann Labs",
        author_email="elena.krismer@hotmail.com",
        url="https://github.com/MannLabs/alphastats",
        project_urls={
            "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
            "GitHub": "https://github.com/MannLabs/alphapeptstats",
            "ReadTheDocs": "https://mannlabs.github.io/alphapeptstats/",
            "PyPi": "https://pypi.org/project/alphastats/"
            # "Scientific paper": None,
        },
        keywords=["bioinformatics", "software", "mass spectometry",],
        classifiers=[
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
        ],
        packages=[
            "alphastats", 
            "alphastats.plots", 
            "alphastats.multicova",
            "alphastats.gui", 
            "alphastats.data", 
            "alphastats.gui.pages",
            "alphastats.gui",
            "alphastats.gui.sample_data",
            "alphastats.gui.utils",
            "alphastats.loader"
        ],
        include_package_data=True,
        entry_points={"console_scripts": "alphastats=alphastats.gui.gui:run",},
        install_requires=requirements,
        python_requires=">=3.8,<4",
    )


if __name__ == "__main__":
    create_pip_wheel()
