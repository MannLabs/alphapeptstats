#!bash
# TODO remove with old release workflow

# Initial cleanup
rm -rf dist
rm -rf build
FILE=alphapeptstats.pkg
if test -f "$FILE"; then
  rm alphapeptstats.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alphapeptstatsinstaller python=3.10 -y
conda activate alphapeptstatsinstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/macos
pip install "../../dist/alphastats-0.6.7-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==5.8
pyinstaller ../pyinstaller/alphastats.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../omiclearn/data/*.fasta dist/omiclearn/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/alphastats/Contents/Resources
cp ../logos/alphapeptstats_logo.icns dist/alphastats/Contents/Resources
mv dist/alphastats_gui dist/alphastats/Contents/MacOS
cp Info.plist dist/alphastats/Contents
cp alphastats_terminal dist/alphastats/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/alphapeptstats_logo.png Resources/alphapeptstats_logo.png
chmod 777 scripts/*

pkgbuild --root dist/alphastats --identifier de.mpg.biochem.alphastats.app --version 0.6.7--install-location /Applications/alphapeptstats.app --scripts scripts alphapeptstats.pkg
productbuild --distribution distribution.xml --resources Resources --package-path alphapeptstats.pkg dist/alphastats_gui_installer_macos.pkg
