from alphastats import SpectronautLoader
from alphastats.gui.utils.ui_helper import StateKeys
from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.GenericLoader import GenericLoader
from alphastats.loader.MaxQuantLoader import MaxQuantLoader
from alphastats.loader.mzTabLoader import mzTabLoader


def get_plotting_options(state):
    dataset = state[StateKeys.DATASET]
    plotting_options = {
        "Sampledistribution Plot": {
            "settings": {
                "method": {"options": ["violin", "box"], "label": "Plot layout"},
                "color": {
                    "options": [None] + state[StateKeys.METADATA_COLUMNS],
                    "label": "Color according to",
                },
            },
            "function": dataset.plot_sampledistribution,
        },
        "Intensity Plot": {
            "settings": {
                "protein_id": {
                    "options": dataset.mat.columns.to_list(),
                    "label": "ProteinID/ProteinGroup",
                },
                "method": {
                    "options": ["violin", "box", "scatter"],
                    "label": "Plot layout",
                },
                "group": {
                    "options": [None] + state[StateKeys.METADATA_COLUMNS],
                    "label": "Color according to",
                },
            },
            "function": dataset.plot_intensity,
        },
        "PCA Plot": {
            "settings": {
                "group": {
                    "options": [None] + state[StateKeys.METADATA_COLUMNS],
                    "label": "Color according to",
                },
                "circle": {"label": "Circle"},
            },
            "function": dataset.plot_pca,
        },
        "UMAP Plot": {
            "settings": {
                "group": {
                    "options": [None] + state[StateKeys.METADATA_COLUMNS],
                    "label": "Color according to",
                },
                "circle": {"label": "Circle"},
            },
            "function": dataset.plot_umap,
        },
        "t-SNE Plot": {
            "settings": {
                "group": {
                    "options": [None] + state[StateKeys.METADATA_COLUMNS],
                    "label": "Color according to",
                },
                "circle": {"label": "Circle"},
            },
            "function": dataset.plot_tsne,
        },
        "Volcano Plot": {
            "between_two_groups": True,  # TODO: between_two_groups is never used anywhere
            "function": dataset.plot_volcano,
        },
        "Clustermap": {"function": dataset.plot_clustermap},
        # "Dendrogram": {"function": state[StateKeys.DATASET].plot_dendrogram},  # TODO why commented?
    }
    return plotting_options


def get_statistic_options(state):
    dataset = state[StateKeys.DATASET]
    statistic_options = {
        "Differential Expression Analysis - T-test": {
            "between_two_groups": True,
            "function": dataset.diff_expression_analysis,
        },
        "Differential Expression Analysis - Wald-test": {
            "between_two_groups": True,
            "function": dataset.diff_expression_analysis,
        },
        "Tukey - Test": {
            "settings": {
                "protein_id": {
                    "options": dataset.mat.columns.to_list(),
                    "label": "ProteinID/ProteinGroup",
                },
                "group": {
                    "options": state[StateKeys.METADATA_COLUMNS],
                    "label": "A metadata variable to calculate pairwise tukey",
                },
            },
            "function": dataset.tukey_test,
        },
        "ANOVA": {
            "settings": {
                "column": {
                    "options": state[StateKeys.METADATA_COLUMNS],
                    "label": "A variable from the metadata to calculate ANOVA",
                },
                "protein_ids": {
                    "options": ["all"] + dataset.mat.columns.to_list(),
                    "label": "All ProteinIDs/or specific ProteinID to perform ANOVA",
                },
                "tukey": {"label": "Follow-up Tukey"},
            },
            "function": dataset.anova,
        },
        "ANCOVA": {
            "settings": {
                "protein_id": {
                    "options": [None] + dataset.mat.columns.to_list(),
                    "label": "Color according to",
                },
                "covar": {
                    "options": state[StateKeys.METADATA_COLUMNS],
                    "label": "Name(s) of column(s) in metadata with the covariate.",
                },
                "between": {
                    "options": state[StateKeys.METADATA_COLUMNS],
                    "label": "Name of the column in the metadata with the between factor.",
                },
            },
            "function": dataset.ancova,
        },
    }
    return statistic_options


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
