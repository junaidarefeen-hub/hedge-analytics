"""Load factor data from Prismatic (primary) or Factor Prices.xlsx (fallback).

Prismatic data is cached locally as a Parquet file to avoid slow MCP round-trips
on every app startup. The cache is refreshed via the sidebar "Reload factor data"
button or when the Parquet file is older than FACTOR_CACHE_MAX_AGE_HOURS.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    FA_FACTOR_XLSX,
    FACTOR_CACHE_MAX_AGE_HOURS,
    FACTOR_GADGET_IDS,
    FACTOR_MODEL_PREFIX,
)

logger = logging.getLogger(__name__)

_XLSX_PATH = os.path.join(os.path.dirname(__file__), "..", FA_FACTOR_XLSX)
_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_PARQUET = _CACHE_DIR / "factor_prices.parquet"
_CACHE_META = _CACHE_DIR / "factor_meta.json"


@dataclass
class FactorData:
    prices: pd.DataFrame  # DatetimeIndex, columns = display names (price index for display)
    returns: pd.DataFrame  # daily returns (diff of cumulative for Prismatic, pct_change for Excel)
    ticker_map: dict[str, str]  # display_name -> ticker_code
    name_map: dict[str, str]  # ticker_code -> display_name


# ---------------------------------------------------------------------------
# Local Parquet cache
# ---------------------------------------------------------------------------


def _cache_is_fresh() -> bool:
    """Check if the local Parquet cache exists and is within the max age."""
    if not _CACHE_PARQUET.exists():
        return False
    age_hours = (time.time() - _CACHE_PARQUET.stat().st_mtime) / 3600
    return age_hours < FACTOR_CACHE_MAX_AGE_HOURS


def _save_to_cache(cumulative: pd.DataFrame) -> None:
    """Save raw cumulative return data to local Parquet cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cumulative.to_parquet(_CACHE_PARQUET)
    logger.info(f"Factor data cached to {_CACHE_PARQUET}")


def _load_from_cache() -> pd.DataFrame:
    """Load raw cumulative return data from local Parquet cache."""
    df = pd.read_parquet(_CACHE_PARQUET)
    logger.info(f"Factor data loaded from cache ({_CACHE_PARQUET})")
    return df


# ---------------------------------------------------------------------------
# Prismatic fetcher
# ---------------------------------------------------------------------------


def _fetch_from_prismatic() -> pd.DataFrame:
    """Fetch raw cumulative return data from Prismatic gadgets via ECM MCP.

    Each gadget returns a CSV with columns: Date, GS {Factor} (L), MS {Factor} (L).
    Values are additive cumulative return indices starting at 0.

    Returns:
        DataFrame of raw cumulative returns (not price indices), indexed by date.
    """
    from data.ecm_client import ECMFetchError, render_gadget_csv, read_share_file

    all_series: dict[str, pd.Series] = {}

    for gadget_id, factor_name in FACTOR_GADGET_IDS.items():
        file_path = render_gadget_csv(gadget_id)
        csv_text = read_share_file(file_path)

        df = pd.read_csv(io.StringIO(csv_text))

        # Find the column matching our model prefix (GS or MS)
        target_col = None
        for col in df.columns:
            if col.startswith(f"{FACTOR_MODEL_PREFIX} ") and col.endswith(" (L)"):
                target_col = col
                break

        if target_col is None:
            logger.warning(
                f"No '{FACTOR_MODEL_PREFIX}' column found for factor '{factor_name}' "
                f"(gadget {gadget_id}). Available: {list(df.columns)}. Skipping."
            )
            continue

        dates = pd.to_datetime(df["Date"], errors="coerce")
        values = pd.to_numeric(df[target_col], errors="coerce")
        series = pd.Series(values.values, index=dates, name=factor_name)
        series = series[series.index.notna()]
        series = series.dropna()

        all_series[factor_name] = series

    if not all_series:
        raise ECMFetchError("No factor data retrieved from any Prismatic gadget.")

    cumulative = pd.DataFrame(all_series).sort_index()
    cumulative.index.name = None
    return cumulative


def _cumulative_to_factor_data(cumulative: pd.DataFrame) -> FactorData:
    """Convert raw cumulative return DataFrame into FactorData.

    Prismatic data is additive cumulative returns starting at 0.
    Daily returns = diff() of the cumulative series (NOT pct_change of price index).
    Price index = 1 + cumulative (for display only, not used in regressions).
    """
    prices = 1.0 + cumulative  # display-friendly price index starting at 1
    # Forward-fill before diff so returns across holiday NaN gaps are preserved
    returns = cumulative.ffill().diff()  # correct daily returns from additive cumulative

    prefix = FACTOR_MODEL_PREFIX.upper()
    ticker_map = {name: f"{prefix}_{name.upper().replace(' ', '_')}" for name in prices.columns}
    name_map = {code: name for name, code in ticker_map.items()}

    return FactorData(prices=prices, returns=returns, ticker_map=ticker_map, name_map=name_map)


def _load_prismatic_with_cache() -> FactorData:
    """Load factor data from Prismatic, using local Parquet cache when fresh."""
    if _cache_is_fresh():
        cumulative = _load_from_cache()
        return _cumulative_to_factor_data(cumulative)

    # Cache is stale or missing — fetch from Prismatic
    from data.ecm_client import is_available

    if not is_available():
        # No ECM credentials — try stale cache before giving up
        if _CACHE_PARQUET.exists():
            logger.info("ECM unavailable but stale cache exists — using it.")
            cumulative = _load_from_cache()
            return _cumulative_to_factor_data(cumulative)
        raise FileNotFoundError("No ECM credentials and no cached factor data.")

    logger.info("Fetching factor data from Prismatic...")
    cumulative = _fetch_from_prismatic()
    _save_to_cache(cumulative)
    logger.info(f"Loaded {len(cumulative.columns)} factors from Prismatic.")
    return _cumulative_to_factor_data(cumulative)


# ---------------------------------------------------------------------------
# Excel loader (fallback)
# ---------------------------------------------------------------------------


def _load_from_excel() -> FactorData:
    """Parse Factor Prices.xlsx and return factor prices + returns."""
    df = pd.read_excel(_XLSX_PATH, header=None, engine="openpyxl")

    tickers = [str(v).strip() for v in df.iloc[0, 5:21].tolist()]
    names = [str(v).strip() for v in df.iloc[3, 5:21].tolist()]

    ticker_map = dict(zip(names, tickers))
    name_map = dict(zip(tickers, names))

    data = df.iloc[5:].copy()
    dates = pd.to_datetime(data.iloc[:, 2], errors="coerce")

    prices = data.iloc[:, 5:21].copy()
    prices.columns = names
    prices.index = dates
    prices.index.name = None

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices[prices.index.notna()]
    prices = prices.dropna(how="all")
    prices = prices.sort_index()

    returns = prices.pct_change(fill_method=None)

    return FactorData(prices=prices, returns=returns, ticker_map=ticker_map, name_map=name_map)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@st.cache_data(ttl=86400, show_spinner="Loading factor data...")
def load_factor_data() -> FactorData:
    """Load factor data: Prismatic (with cache) → Excel fallback."""
    try:
        return _load_prismatic_with_cache()
    except Exception as e:
        logger.warning(f"Prismatic factor load failed ({e}) — falling back to Excel.")
        return _load_from_excel()


def clear_factor_cache():
    """Clear both the Streamlit cache and the local Parquet cache.

    Forces a fresh fetch from Prismatic on next access.
    """
    load_factor_data.clear()
    if _CACHE_PARQUET.exists():
        _CACHE_PARQUET.unlink()
        logger.info("Local factor Parquet cache deleted.")


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

    common = fr.index.intersection(br.index).intersection(mr.index)
    fr, br, mr = fr.loc[common], br.loc[common], mr.loc[common]

    mask = fr.notna().all(axis=1) & br.notna() & mr.notna()
    return fr[mask], br[mask], mr[mask]
