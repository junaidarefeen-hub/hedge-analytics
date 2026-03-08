import pandas as pd


def beta_matrix(returns: pd.DataFrame, benchmarks: list[str]) -> pd.DataFrame:
    """Full-period beta matrix: all tickers (rows) × benchmarks (columns).

    Beta = Cov(ticker, benchmark) / Var(benchmark), computed on overlapping data.
    """
    tickers = [c for c in returns.columns if c not in benchmarks]
    if not tickers:
        tickers = list(returns.columns)

    result = pd.DataFrame(index=tickers, columns=benchmarks, dtype=float)
    for bm in benchmarks:
        if bm not in returns.columns:
            continue
        bm_var = returns[bm].var()
        if bm_var == 0:
            continue
        for tk in tickers:
            if tk not in returns.columns:
                continue
            overlap = returns[[tk, bm]].dropna()
            if len(overlap) < 2:
                continue
            result.loc[tk, bm] = overlap[tk].cov(overlap[bm]) / overlap[bm].var()
    return result


def rolling_beta(
    returns: pd.DataFrame, ticker: str, benchmark: str, window: int
) -> pd.Series:
    """Rolling beta of ticker vs benchmark."""
    pair = returns[[ticker, benchmark]].dropna()
    cov = pair[ticker].rolling(window=window, min_periods=window).cov(pair[benchmark])
    var = pair[benchmark].rolling(window=window, min_periods=window).var()
    return (cov / var).rename(f"{ticker} β vs {benchmark}")
