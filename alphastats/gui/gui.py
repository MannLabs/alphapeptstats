import os
import sys
from streamlit.web import cli as stcli

def run():
    file_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(file_path))
    os.system("python -m streamlit run AlphaPeptStats.py --global.developmentMode=false --server.maxUploadSize 500")
    _this_file = os.path.abspath(__file__)
    _this_directory = os.path.dirname(_this_file)

    file_path = os.path.join(_this_directory, 'AlphaPeptStats.py')

    HOME = os.path.expanduser("~")
    ST_PATH = os.path.join(HOME, ".streamlit")

    for folder in [ST_PATH]:
        if not os.path.isdir(folder):
            os.mkdir(folder)


    print(f'Starting AlphaPeptStats from {file_path}')

    args = ["streamlit", "run", file_path, "--global.developmentMode=false", "--browser.gatherUsageStats=False", "--server.maxUploadSize 500"]

    sys.argv = args

    sys.exit(stcli.main())
