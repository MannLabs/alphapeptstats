#!/bin/bash
set -e -u

# Build the installer for MacOS.
# This script must be run from the root of the repository.
# Prerequisites: wheel has been build, e.g. using build_wheel.sh

rm -rf dist_pyinstaller build_pyinstaller

WHL_NAME=$(cd dist && ls ./*.whl && cd ..)
pip install "dist/${WHL_NAME}"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphastats.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
