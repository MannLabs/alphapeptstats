conda create -n alphastats python=3.8 -y
conda activate alphastats
pip install -e '../.[stable,development-stable]'
alphastats
conda deactivate
