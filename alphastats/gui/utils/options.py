import streamlit as st


plotting_options = {
    "Sampledistribution": {
        "settings": {
            "method": {"options": ["violin", "box"], "label": "Plot layout"},
            "color": {
                "options": [None] + st.session_state.metadata_columns,
                "label": "Color according to",
            },
        },
        "function": st.session_state.dataset.plot_sampledistribution,
    },
    "Intensity": {
        "settings": {
            "protein_id": {
                "options": st.session_state.dataset.mat.columns.to_list(),
                "label": "ProteinID/ProteinGroup",
            },
            "method": {
                "options": ["violin", "box", "scatter"],
                "label": "Plot layout",
            },
            "group": {
                "options": [None] + st.session_state.metadata_columns,
                "label": "Color according to",
            },
        },
        "function": st.session_state.dataset.plot_intensity,
    },
    "PCA": {
        "settings": {
            "group": {
                "options": [None] + st.session_state.metadata_columns,
                "label": "Color according to",
            },
            "circle": {"label": "Circle"},
        },
        "function": st.session_state.dataset.plot_pca,
    },
    "UMAP": {
        "settings": {
            "group": {
                "options": [None] + st.session_state.metadata_columns,
                "label": "Color according to",
            },
            "circle": {"label": "Circle"},
        },
        "function": st.session_state.dataset.plot_umap,
    },
    "t-SNE": {
        "settings": {
            "group": {
                "options": [None] + st.session_state.metadata_columns,
                "label": "Color according to",
            },
            "circle": {"label": "Circle"},
        },
        "function": st.session_state.dataset.plot_tsne,
    },
    "Volcano": {
        "between_two_groups": True,
        "function": st.session_state.dataset.plot_volcano,
    },
    "Clustermap": {"function": st.session_state.dataset.plot_clustermap},
    "Dendrogram": {"function": st.session_state.dataset.plot_dendrogram},
}

statistic_options = {
    "Differential Expression Analysis - T-test": {
        "between_two_groups": True,
        "function": st.session_state.dataset.perform_diff_expression_analysis,
    },
    "Differential Expression Analysis - Wald-test": {
        "between_two_groups": True,
        "function": st.session_state.dataset.perform_diff_expression_analysis,
    },
    "Tukey - Test": {
        "settings": {
            "protein_id": {
                "options": st.session_state.dataset.mat.columns.to_list(),
                "label": "ProteinID/ProteinGroup",
            },
            "group": {
                "options": st.session_state.metadata_columns,
                "label": "A metadata variable to calculate pairwise tukey",
            },
        },
        "function": st.session_state.dataset.calculate_tukey,
    },
    "ANOVA": {
        "settings": {
            "column": {
                "options": st.session_state.metadata_columns,
                "label": "A variable from the metadata to calculate ANOVA",
            },
            "protein_ids": {
                "options": ["all"] + st.session_state.dataset.mat.columns.to_list(),
                "label": "All ProteinIDs/or specific ProteinID to perform ANOVA",
            },
            "tukey": {"label": "Follow-up Tukey"},
        },
        "function": st.session_state.dataset.anova,
    },
    "ANCOVA": {
        "settings": {
            "protein_id": {
                "options": [None] + st.session_state.dataset.mat.columns.to_list(),
                "label": "Color according to",
            },
            "covar": {
                "options": st.session_state.metadata_columns,
                "label": "Name(s) of column(s) in metadata with the covariate.",
            },
            "between": {
                "options": st.session_state.metadata_columns,
                "label": "Name of the column in the metadata with the between factor.",
            },
        },
        "function": st.session_state.dataset.ancova,
    },
}
