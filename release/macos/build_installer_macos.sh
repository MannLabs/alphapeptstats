#!/bin/bash
set -e -u

# Build the installer for MacOS.
# This script must be run from the root of the repository.

rm -rf dist
rm -rf build

# Creating the wheel
python setup.py sdist bdist_wheel
pip install "dist/alphastats-0.6.7-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphastats.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
