conda create -n alphastats python=3.8 -y
conda activate alphastats
pip install -e '../.[development]'
alphastats
conda deactivate
