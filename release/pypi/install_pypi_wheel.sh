conda create -n alphastats_pip_test python=3.8 -y
conda activate alphastats_pip_test
pip install "alphastats[stable]"
alphastats
conda deactivate
