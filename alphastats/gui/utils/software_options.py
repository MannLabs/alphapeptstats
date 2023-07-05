from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.SpectronautLoader import SpectronautLoader
from alphastats.loader.GenericLoader import GenericLoader
from alphastats.loader.mzTabLoader import mzTabLoader

software_options = {
    "MaxQuant": {
        "import_file": "Please upload proteinGroups.txt",
        "intensity_column": ["LFQ intensity [sample]", "Intensity [sample]"],
        "index_column": ["Protein IDs", "Gene names", "Majority protein IDs", "Protein names"],
        "loader_function": MaxQuantLoader,
    },
    "AlphaPept": {
        "import_file": "Please upload results_proteins.csv or results.hdf",
        "intensity_column": ["[sample]_LFQ"],
        "index_column": ["Unnamed: 0"],
        "loader_function": AlphaPeptLoader,
    },
    "DIANN": {
        "import_file": "Please upload report.pg_matrix.tsv",
        "intensity_column": ["[sample]"],
        "index_column": ["Protein.Group"],
        "loader_function": DIANNLoader,
    },
    "FragPipe": {
        "import_file": "Please upload combined_protein.tsv file from FragPipe.",
        "intensity_column": ["[sample] MaxLFQ Intensity", "[sample] Intensity"],
        "index_column": ["Protein", "Protein ID"],
        "loader_function": FragPipeLoader,
    },
    "Spectronaut": {
        "import_file": "Please upload spectronaut.tsv",
        "intensity_column": ["PG.Quantity", "F.PeakArea", "PG.Log2Quantity"],
        "index_column": ["PG.ProteinGroups", "PEP.StrippedSequence"],
        "loader_function": SpectronautLoader,
    },
     "Other": {
        "import_file": "Please upload your proteomics file.",
        "intensity_column": [],
        "index_column": [],
        "loader_function": GenericLoader,
    },
     "mzTab": {
        "import_file": "Please upload your .mzTab file with quantitative proteomics data.",
        "intensity_column": ["protein_abundance_[sample]"],
        "index_column": ["accession"],
        "loader_function": mzTabLoader,
    },
}
