import pandas as pd


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
