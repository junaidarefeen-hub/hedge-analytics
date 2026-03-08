from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf

from config import CACHE_TTL_SECONDS


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching price data...")
def validate_and_fetch(
    tickers: list[str], start: date, end: date
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch adjusted close prices from yfinance.

    Returns:
        (prices_df, failed_tickers) where prices_df has columns = valid tickers.
    """
    if not tickers:
        return pd.DataFrame(), tickers

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame(), list(tickers)

    # yf.download returns MultiIndex columns when len(tickers) > 1
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    # Identify failed tickers (all NaN)
    valid_mask = prices.notna().any()
    failed = [t for t in prices.columns if not valid_mask[t]]
    prices = prices[[c for c in prices.columns if valid_mask[c]]]

    prices = prices.sort_index()
    return prices, failed


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Looking up ticker names...")
def fetch_ticker_names(tickers: list[str]) -> dict[str, str]:
    """Fetch long names for a list of tickers. Returns {ticker: name} dict."""
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names[t] = info.get("longName") or info.get("shortName") or t
        except Exception:
            names[t] = t
    return names
