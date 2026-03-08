from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from time import sleep

import pandas as pd
import streamlit as st
import yfinance as yf
from curl_cffi import requests as curl_requests

from config import CACHE_TTL_SECONDS, INTRADAY_MAX_DAYS

_session = curl_requests.Session(verify=False, impersonate="chrome")

NAME_CACHE_TTL = 86400  # 24 hours — company names rarely change


def validate_interval_date_range(interval: str, start: date, end: date) -> str | None:
    """Validate that the date range is within yfinance limits for the given interval.

    Returns an error message string if invalid, else None.
    """
    max_days = INTRADAY_MAX_DAYS.get(interval, 100000)
    actual_days = (end - start).days
    if actual_days > max_days:
        return f"Interval '{interval}' supports at most {max_days} days, but date range spans {actual_days} days."
    return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching price data...")
def validate_and_fetch(
    tickers: list[str], start: date, end: date, interval: str = "1d",
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
        interval=interval,
        auto_adjust=True,
        progress=False,
        session=_session,
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
            info = yf.Ticker(ticker, session=_session).info
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


def clear_cache():
    """Clear all cached data (prices + ticker names)."""
    validate_and_fetch.clear()
    fetch_ticker_names.clear()
