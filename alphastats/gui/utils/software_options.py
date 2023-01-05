from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.SpectronautLoader import SpectronautLoader

software_options = {
    "MaxQuant": {
        "import_file": "proteinGroups.txt",
        "intensity_column": ["LFQ intensity [sample]"],
        "index_column": ["Protein IDs"],
        "loader_function": MaxQuantLoader,
    },
    "AlphaPept": {
        "import_file": "results_proteins.csv or results.hdf",
        "intensity_column": ["[sample]_LFQ"],
        "index_column": ["Unnamed: 0"],
        "loader_function": AlphaPeptLoader,
    },
    "DIANN": {
        "import_file": "report.pg_matrix.tsv",
        "intensity_column": ["[sample]"],
        "index_column": ["Protein.Group"],
        "loader_function": DIANNLoader,
    },
    "FragPipe": {
        "import_file": "combined_protein.tsv",
        "intensity_column": ["[sample] MaxLFQ Intensity "],
        "index_column": ["Protein"],
        "loader_function": FragPipeLoader,
    },
    "Spectronaut": {
        "import_file": "spectronaut.tsv",
        "intensity_column": ["PG.Quantity", "F.PeakArea", "PG.Log2Quantity"],
        "index_column": ["PG.ProteinGroups", "PEP.StrippedSequence"],
        "loader_function": SpectronautLoader,
    },
}
