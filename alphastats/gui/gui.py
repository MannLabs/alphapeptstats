import os


def run():
    file_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(file_path))
    os.system("python -m streamlit run AlphaPeptStats.py")
