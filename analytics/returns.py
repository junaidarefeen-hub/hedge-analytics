import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute daily returns from price DataFrame.

    Args:
        prices: DataFrame with date index and ticker columns.
        method: 'log' for log returns, 'simple' for arithmetic returns.
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    else:
        return prices.pct_change().dropna(how="all")
