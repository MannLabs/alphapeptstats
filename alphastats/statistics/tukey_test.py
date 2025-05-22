import pandas as pd
import pingouin

from alphastats.dataset.keys import Cols


def tukey_test(
    df: pd.DataFrame,
    protein_id: str,
    group: str,
) -> pd.DataFrame:
    """Calculate Pairwise Tukey-HSD post-hoc test
    Wrapper around:
    https://pingouin-stats.org/generated/pingouin.pairwise_tukey.html#pingouin.pairwise_tukey

    Args:
        protein_id (str): ProteinID to calculate Pairwise Tukey-HSD post-hoc test - dependend variable
        group (str): A metadata column used calculate pairwise tukey
        df (pandas.DataFrame): DataFrame to perform the tukey test on

    Returns:
        pandas.DataFrame:
        * ``'A'``: Name of first measurement
        * ``'B'``: Name of second measurement
        * ``'mean(A)'``: Mean of first measurement
        * ``'mean(B)'``: Mean of second measurement
        * ``'diff'``: Mean difference (= mean(A) - mean(B))
        * ``'se'``: Standard error
        * ``'T'``: T-values
        * ``'p-tukey'``: Tukey-HSD corrected p-values
        * ``'hedges'``: Hedges effect size (or any effect size defined in
        ``effsize``)
        * ``'comparison'``: combination of measurment
        * ``'Protein ID'``: ProteinID/ProteinGroup
    """
    try:
        tukey_df = pingouin.pairwise_tukey(data=df, dv=protein_id, between=group)
        tukey_df["comparison"] = tukey_df["A"] + " vs. " + tukey_df["B"] + " Tukey Test"
        tukey_df[Cols.INDEX] = protein_id

    except ValueError:
        tukey_df = pd.DataFrame()

    return tukey_df
