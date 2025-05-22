from alphastats.loader.alphapept_loader import AlphaPeptLoader
from alphastats.loader.diann_loader import DIANNLoader
from alphastats.loader.fragpipe_loader import FragPipeLoader
from alphastats.loader.generic_loader import GenericLoader
from alphastats.loader.maxquant_loader import MaxQuantLoader
from alphastats.loader.mztab_loader import mzTabLoader
from alphastats.loader.spectronaut_loader import SpectronautLoader

SOFTWARE_OPTIONS = {
    "MaxQuant": {
        "import_file": "Please upload proteinGroups.txt",
        "intensity_column": ["LFQ intensity [sample]", "Intensity [sample]"],
        "index_column": [
            "Protein IDs",
            "Gene names",
            "Majority protein IDs",
            "Protein names",
        ],
        "loader_function": MaxQuantLoader,
    },
    "AlphaPept": {
        "import_file": "Please upload results_proteins.csv or results.hdf",
        "intensity_column": ["[sample]_LFQ"],
        "index_column": ["Unnamed: 0"],
        "loader_function": AlphaPeptLoader,  # TODO loader_function -> loader_class or loader
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
