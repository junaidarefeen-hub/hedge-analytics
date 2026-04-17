"""Parquet-based cache for market monitor price data.

Follows the incremental refresh pattern from the ne73-dashboard:
- Load from Parquet on app start (fast)
- Incremental refresh fetches only new dates
- Full refresh for new tickers
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import MM_CACHE_DIR, MM_CACHE_MAX_AGE_HOURS, MM_DEFAULT_LOOKBACK_DAYS

logger = logging.getLogger(__name__)

_PRICES_FILE = Path(MM_CACHE_DIR) / "prices.parquet"
_METADATA_FILE = Path(MM_CACHE_DIR) / "metadata.json"


@dataclass
class RefreshPlan:
    """Describes what data needs to be fetched."""

    new_tickers: list[str]  # need full history (not yet cached)
    existing_tickers: list[str]  # only need new dates
    incremental_start: str  # date string for existing tickers
    full_start: str  # date string for new tickers (1yr ago)
    end_date: str  # today
    is_full_refresh: bool


def load_cached_prices() -> pd.DataFrame | None:
    """Load prices from the Parquet cache.

    Returns None if cache doesn't exist.
    """
    if not _PRICES_FILE.exists():
        return None
    try:
        df = pd.read_parquet(_PRICES_FILE)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        logger.warning(f"Failed to read price cache: {e}")
        return None


def save_prices(prices: pd.DataFrame) -> None:
    """Write prices to Parquet cache and update metadata."""
    _PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(_PRICES_FILE)

    meta = {
        "last_refresh": datetime.now(timezone.utc).isoformat(),
        "ticker_count": len(prices.columns),
        "row_count": len(prices),
        "date_range": f"{prices.index.min().date()} to {prices.index.max().date()}",
    }
    _METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(prices.columns)} tickers, {len(prices)} days to cache")


def get_metadata() -> dict:
    """Load refresh metadata. Returns empty dict if missing."""
    if not _METADATA_FILE.exists():
        return {}
    try:
        return json.loads(_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_stale(max_age_hours: float = MM_CACHE_MAX_AGE_HOURS) -> bool:
    """Check if cached data is older than max_age_hours."""
    meta = get_metadata()
    last_refresh = meta.get("last_refresh")
    if not last_refresh:
        return True
    try:
        refresh_time = datetime.fromisoformat(last_refresh)
        if refresh_time.tzinfo is None:
            refresh_time = refresh_time.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - refresh_time).total_seconds() / 3600
        return age_hours > max_age_hours
    except (ValueError, TypeError):
        return True


def get_refresh_plan(constituents: list[str]) -> RefreshPlan:
    """Determine what data needs to be fetched.

    Compares current constituent list against cached data to minimize RVX calls.
    """
    today = date.today()
    one_year_ago = today - timedelta(days=MM_DEFAULT_LOOKBACK_DAYS)

    cached = load_cached_prices()
    if cached is None or cached.empty:
        return RefreshPlan(
            new_tickers=constituents,
            existing_tickers=[],
            incremental_start=one_year_ago.isoformat(),
            full_start=one_year_ago.isoformat(),
            end_date=today.isoformat(),
            is_full_refresh=True,
        )

    cached_tickers = set(cached.columns)
    last_date = cached.index.max().date()

    new_tickers = [t for t in constituents if t not in cached_tickers]
    existing_tickers = [t for t in constituents if t in cached_tickers]

    return RefreshPlan(
        new_tickers=new_tickers,
        existing_tickers=existing_tickers,
        incremental_start=last_date.isoformat(),  # re-fetch last date for intraday update
        full_start=one_year_ago.isoformat(),
        end_date=today.isoformat(),
        is_full_refresh=False,
    )


def merge_incremental(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Merge new data into existing, preferring new values for overlapping dates.

    Same combine_first pattern as ne73-dashboard.
    """
    if existing is None or existing.empty:
        return new_data
    if new_data is None or new_data.empty:
        return existing
    return new_data.combine_first(existing).sort_index()


def clear_cache() -> None:
    """Delete cached files."""
    for f in (_PRICES_FILE, _METADATA_FILE):
        if f.exists():
            f.unlink()
    logger.info("Market monitor cache cleared")
