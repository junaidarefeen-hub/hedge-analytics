"""Load factor data from Prismatic with local Parquet cache fallback.

Prismatic data is cached locally as a Parquet file to avoid slow MCP round-trips
on every app startup. The cache is refreshed via the sidebar "Reload factor data"
button or when the Parquet file is older than FACTOR_CACHE_MAX_AGE_HOURS.

When ECM/Prismatic is unavailable (e.g. on Render), the loader uses the committed
Parquet cache. Staleness is reported based on the last date in the data.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    FACTOR_CACHE_MAX_AGE_HOURS,
    FACTOR_GADGET_IDS,
    FACTOR_MODEL_PREFIX,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_PARQUET = _CACHE_DIR / "factor_prices.parquet"


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


def _fetch_one_gadget(
    gadget_id: str,
    factor_name: str,
    start_date: str | None,
) -> tuple[str, pd.Series | None]:
    """Fetch a single gadget's CSV and parse it. Returns ``(name, series)``.

    Returned ``series`` is the cumulative return for the chosen model prefix
    (default ``GS``). When ``start_date`` is supplied, Prismatic resets the
    cumulative to 0 at that date — callers must rebase before merging into
    a longer history (see ``_rebase_incremental``).
    """
    from data.ecm_client import render_gadget_csv, read_share_file

    parameters = {"plotterStartDate": start_date} if start_date else None
    file_path = render_gadget_csv(gadget_id, parameters=parameters)
    csv_text = read_share_file(file_path)
    df = pd.read_csv(io.StringIO(csv_text))

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
        return factor_name, None

    dates = pd.to_datetime(df["Date"], errors="coerce")
    values = pd.to_numeric(df[target_col], errors="coerce")
    series = pd.Series(values.values, index=dates, name=factor_name)
    series = series[series.index.notna()].dropna()
    return factor_name, series


def _fetch_from_prismatic(
    start_date: str | None = None,
    max_workers: int = 9,
) -> pd.DataFrame:
    """Fetch raw cumulative return data from all factor gadgets in parallel.

    Each gadget returns a CSV with columns: Date, GS {Factor} (L), MS {Factor} (L).
    Values are additive cumulative return indices that start at 0 on the
    first returned date.

    Args:
        start_date: Optional ``M/D/YY`` string. If provided, gadgets only
            return rows from this date forward (cumulative resets to 0 at
            that date — caller must rebase before merging).
        max_workers: Thread pool size for parallel gadget fetches. The
            sequential per-gadget cost is dominated by network round-trip
            to Prismatic; parallelism cuts wall time roughly N-fold.

    Returns:
        DataFrame of raw cumulative returns (not price indices), indexed by
        date with one column per factor name.
    """
    from concurrent.futures import ThreadPoolExecutor

    from data.ecm_client import ECMFetchError

    items = list(FACTOR_GADGET_IDS.items())
    all_series: dict[str, pd.Series] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_fetch_one_gadget, gid, name, start_date)
            for gid, name in items
        ]
        for fut in futures:
            name, series = fut.result()
            if series is not None:
                all_series[name] = series

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
# Public API
# ---------------------------------------------------------------------------


@st.cache_data(ttl=86400, show_spinner="Loading factor data...")
def load_factor_data() -> FactorData:
    """Load factor data: Prismatic with local Parquet cache fallback."""
    return _load_prismatic_with_cache()


def clear_factor_cache():
    """Force a full re-fetch from Prismatic and clear Streamlit's in-memory cache.

    Behavior:
      * Local dev (ECM reachable): re-fetches all 9 GS factor gadgets from
        Prismatic in parallel (``ThreadPoolExecutor``) and overwrites the
        Parquet cache. The fetch is full-history because Prismatic's
        ``plotterStartDate`` filter changes the underlying factor scale —
        partial fetches are not arithmetically compatible with the long
        history, so a parallel full re-fetch is the correct trade-off.
      * Render (no ECM): clears Streamlit's in-memory cache only; the
        committed Parquet remains the source of truth.
    """
    load_factor_data.clear()
    from data.ecm_client import is_available

    if is_available():
        try:
            cumulative = _fetch_from_prismatic()
            _save_to_cache(cumulative)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(f"Factor cache refresh failed: {e}")


def get_factor_staleness_days(factor_data: FactorData) -> int | None:
    """Return calendar days between the last factor data date and today."""
    if factor_data is None or factor_data.prices.empty:
        return None
    last_date = factor_data.prices.index.max()
    return (pd.Timestamp.today().normalize() - last_date.normalize()).days


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
