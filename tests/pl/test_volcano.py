import pandas as pd

from alphastats.dataset.keys import Cols
from alphastats.pl.volcano import plot_volcano
from alphastats.tl.differential_expression_analysis import DeaColumns

show_figures = False

valid_statistics_results = pd.DataFrame(
    {
        DeaColumns.LOG2FC: [0.5, 1.0, 1.5, -2.0, 3.0, 0],
        DeaColumns.QVALUE: [0.1, 0.05, 0.06, 0.001, 0.002, 0.02],
        DeaColumns.PVALUE: [0.1, 0.05, 0.06, 0.001, 0.002, 0.02],
    },
    index=["f1", "f2", "f3", "f4", "f5", "f6"],
)
valid_statistics_results.index.name = Cols.INDEX
valid_feature_to_repr_map = {
    "f1": "gene1",
    "f2": "feature2;lorem ipsum",
    "f3": "gene3",
    "f4": "gene4",
    "f5": "gene5",
    "f6": "gene6",
}


def test_plot_volcano():
    fig, df = plot_volcano(
        statistics_results=valid_statistics_results.copy(),
        feature_to_repr_map=valid_feature_to_repr_map,
        group1="mutant",
        group2="ctrl",
        qvalue_cutoff=0.05,
        log2fc_cutoff=1,
        label_significant=True,
        flip_xaxis=False,
    )
    assert fig is not None
    if show_figures:
        fig.show()

    assert all(el in df.columns for el in ["label", "significant", "log2(mutant/ctrl)"])

    assert "mutant/ctrl" in fig.layout.xaxis.title.text
    annotation_texts = [annotation.text for annotation in fig.layout.annotations]
    assert set(annotation_texts) == {
        "<b>mutant</b>",
        "<b>ctrl</b>",
        "feature2;l...",
        "gene4",
        "gene5",
    }
    group1_annotation = [
        annotation
        for annotation in fig.layout.annotations
        if annotation.text == "<b>mutant</b>"
    ][0]
    assert group1_annotation.x > 0
