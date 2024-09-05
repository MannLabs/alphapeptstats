#!bash
# TODO remove with old release workflow

# Initial cleanup
rm -rf dist
rm -rf build
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alphapeptstats_installer python=3.10 -y
conda activate alphapeptstats_installer

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/linux
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "../../dist/alphastats-0.6.7-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==5.8
pyinstaller ../pyinstaller/alphastats.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../omiclearn/data/*.fasta dist/omiclearn/data
# WARNING: this probably does not work!!!!

# Wrapping the pyinstaller folder in a .deb package
mkdir -p dist/alphapeptstats_gui_installer_linux/usr/local/bin/alphastats
mv dist/alphapeptstats dist/alphapeptstats_gui_installer_linux/usr/local/bin/alphastats
mkdir dist/alphapeptstats_gui_installer_linux/DEBIAN
cp control dist/alphapeptstats_gui_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/alphapeptstats_gui_installer_linux/
