from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from time import sleep

import pandas as pd
import streamlit as st
import yfinance as yf

from config import CACHE_TTL_SECONDS

NAME_CACHE_TTL = 86400  # 24 hours — company names rarely change


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


def _fetch_single_name(ticker: str, retries: int = 2) -> tuple[str, str]:
    """Fetch the display name for a single ticker with retry."""
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName")
            if name:
                return ticker, name
        except Exception:
            pass
        if attempt < retries:
            sleep(0.5 * (attempt + 1))
    return ticker, ticker


@st.cache_data(ttl=NAME_CACHE_TTL, show_spinner="Looking up ticker names...")
def fetch_ticker_names(tickers: list[str]) -> dict[str, str]:
    """Fetch long names for a list of tickers in parallel."""
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as pool:
        futures = {pool.submit(_fetch_single_name, t): t for t in tickers}
        names = {}
        for future in as_completed(futures):
            ticker, name = future.result()
            names[ticker] = name
    return names
