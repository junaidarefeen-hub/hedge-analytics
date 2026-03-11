"""Load GS factor price indices from Factor Prices.xlsx."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from config import FA_FACTOR_XLSX

_XLSX_PATH = os.path.join(os.path.dirname(__file__), "..", FA_FACTOR_XLSX)

# Excel layout (0-indexed rows/cols with header=None):
#   Row 0, cols 5-20: ticker codes (16 GS factor indices)
#   Row 3, cols 5-20: display names
#   Row 5+, col 2: trading date
#   Row 5+, cols 5-20: factor prices


@dataclass
class FactorData:
    prices: pd.DataFrame  # DatetimeIndex, columns = display names
    returns: pd.DataFrame  # pct_change() of prices
    ticker_map: dict[str, str]  # display_name -> ticker_code
    name_map: dict[str, str]  # ticker_code -> display_name


@st.cache_data(ttl=86400, show_spinner="Loading factor data...")
def load_factor_data() -> FactorData:
    """Parse Factor Prices.xlsx and return factor prices + returns."""
    df = pd.read_excel(_XLSX_PATH, header=None, engine="openpyxl")

    # Extract ticker codes and display names
    tickers = [str(v).strip() for v in df.iloc[0, 5:21].tolist()]
    names = [str(v).strip() for v in df.iloc[3, 5:21].tolist()]

    ticker_map = dict(zip(names, tickers))
    name_map = dict(zip(tickers, names))

    # Extract price data (row 5 onward)
    data = df.iloc[5:].copy()
    dates = pd.to_datetime(data.iloc[:, 2], errors="coerce")

    prices = data.iloc[:, 5:21].copy()
    prices.columns = names
    prices.index = dates
    prices.index.name = None

    # Coerce #N/A strings and non-numeric values to NaN
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Clean up: drop rows with invalid dates or all-NaN prices
    prices = prices[prices.index.notna()]
    prices = prices.dropna(how="all")
    prices = prices.sort_index()

    returns = prices.pct_change()

    return FactorData(
        prices=prices,
        returns=returns,
        ticker_map=ticker_map,
        name_map=name_map,
    )


def clear_factor_cache():
    """Clear the factor data cache so the XLS is re-read on next access."""
    load_factor_data.clear()


def align_factor_returns(
    factor_returns: pd.DataFrame,
    basket_returns: pd.Series,
    market_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Align factor, basket, and market returns on common dates via inner join.

    Normalizes indices to date-only timestamps and drops rows with any NaN.
    """
    fr = factor_returns.copy()
    fr.index = fr.index.normalize()

    br = basket_returns.copy()
    br.index = br.index.normalize()

    mr = market_returns.copy()
    mr.index = mr.index.normalize()

    # Inner join on all three indices
    common = fr.index.intersection(br.index).intersection(mr.index)
    fr, br, mr = fr.loc[common], br.loc[common], mr.loc[common]

    # Drop rows with any remaining NaN
    mask = fr.notna().all(axis=1) & br.notna() & mr.notna()
    return fr[mask], br[mask], mr[mask]
