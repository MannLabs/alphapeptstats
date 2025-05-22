import logging
import unittest

from alphastats.dataset.dataset import DataSet
from alphastats.loader.maxquant_loader import MaxQuantLoader

logger = logging.getLogger(__name__)


class TestGPT(unittest.TestCase):
    def setUp(self):
        self.loader = MaxQuantLoader(file="testfiles/maxquant/proteinGroups.txt")
        self.metadata_path = "testfiles/maxquant/metadata.xlsx"
        self.obj = DataSet(
            loader=self.loader,
            metadata_path_or_df=self.metadata_path,
            sample_column="sample",
        )
        # expected dimensions of matrix
        self.matrix_dim = (312, 2596)
        self.matrix_dim_filtered = (312, 2397)
        self.comparison_column = "disease"


if __name__ == "__main__":
    unittest.main()
