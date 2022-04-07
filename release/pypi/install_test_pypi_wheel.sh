conda create -n alphastats_pip_test python=3.8 -y
conda activate alphastats_pip_test
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "alphastats[stable]"
alphastats
conda deactivate
