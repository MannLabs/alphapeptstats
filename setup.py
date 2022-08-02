#!python

# builtin
import setuptools
import re
import os
import configparser
config = configparser.ConfigParser()
config.read('settings.ini')
package = config["DEFAULT"]


def get_long_description():
    with open("README.md", "r") as readme_file:
        long_description = readme_file.read()
    return long_description


def create_pip_wheel():
    setuptools.setup(
        name=package["lib_name"],
        version=package["version"],
        license=package["license"],
        description=package["description"],
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author=package["author"],
        author_email=package["author_email"],
        url=package["git_url"],
        keywords=package["keywords"],
        packages=[package["lib_name"]],
        include_package_data=True,
        package_data={package["lib_name"]: ["data/contaminations.txt"]},
        entry_points={"console_scripts": package["console_scripts"],},
        install_requires=[
            "pandas", 
            "sklearn",
            "data_cache",
            "dash_bio",
            "plotly",
            "iteration_utilities",
            "openpyxl",
            "sklearn_pandas",
            "pingouin", 
            "pywin32==225; sys_platform=='win32'"
        ],
        python_requires=">=3.8",
    )


if __name__ == "__main__":
    create_pip_wheel()
