# Build the installer for Windows.
# This script must be run from the root of the repository.
# Prerequisites: wheel has been build, e.g. using build_wheel.sh

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./build_pyinstaller
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist_pyinstaller

# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "dist/alphastats-0.6.9-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphastats.spec  --distpath dist_pyinstaller --workpath build_pyinstaller -y
