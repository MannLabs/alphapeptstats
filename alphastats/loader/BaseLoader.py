import pandas as pd

class BaseLoader():
    """Parent object of other Loaders
    """
    def __init__(self, file_path: str, sep= "\t"):
        """_summary_

        Args:
            file_path (str): path to file 
            sep (str, optional): file separation. Defaults to "\t".
        """

        self.rawdata = pd.read_csv(file_path, sep = sep)
        