#!/bin/bash
set -e -u

# Build the installer for Linux.
# This script must be run from the root of the repository.
# Prerequisites: wheel has been build, e.g. using build_wheel.sh

rm -rf dist_pyinstaller build_pyinstaller

# Setting up the local package
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "dist/alphastats-0.6.9-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphastats.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
