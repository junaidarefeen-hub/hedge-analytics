import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-period Pearson correlation matrix across all tickers."""
    return returns.corr()


def rolling_correlation(
    returns: pd.DataFrame, ticker_a: str, ticker_b: str, window: int
) -> pd.Series:
    """Rolling pairwise Pearson correlation between two tickers."""
    return returns[ticker_a].rolling(window=window, min_periods=window).corr(
        returns[ticker_b]
    )


def correlation_clustering(
    corr_matrix: pd.DataFrame, method: str = "ward",
) -> dict:
    """Hierarchical clustering based on correlation distance.

    Args:
        corr_matrix: Square correlation matrix.
        method: Linkage method ('ward', 'complete', 'average', 'single').

    Returns:
        Dict with 'linkage_matrix' (Z), 'labels' (list), 'distance_matrix' (condensed).
    """
    if len(corr_matrix) < 3:
        raise ValueError("Need at least 3 tickers for clustering.")

    # Distance = 1 - |corr|
    dist_matrix = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(dist_matrix, 0)
    # Ensure symmetry and non-negativity
    dist_matrix = np.maximum(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=method)

    return {
        "linkage_matrix": Z,
        "labels": list(corr_matrix.columns),
        "distance_matrix": condensed,
    }
