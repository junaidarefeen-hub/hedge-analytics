"""Market data refresh script — runs outside Streamlit.

Two modes:
  python scripts/refresh_market_data.py --full     # Bulk historical fetch (10yr, one-time)
  python scripts/refresh_market_data.py --eod      # Daily close prices (incremental)

The --full mode fetches 10 years of daily close data for all S&P 500 constituents
plus SPX. This is a one-time operation that takes ~3-5 minutes.

The --eod mode fetches only new dates since the last cached date. This is designed
to run daily after market close (4:15 PM ET) via a scheduled trigger.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

# Add project root to path so imports work
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import MM_DEFAULT_LOOKBACK_DAYS
from data.market_monitor.cache_manager import (
    get_refresh_plan,
    incremental_eod_refresh,
    load_cached_prices,
    merge_incremental,
    save_prices,
)
from data.market_monitor.constituents import get_all_tickers
from data.market_monitor.rvx_fetcher import fetch_prices


def _progress(completed: int, total: int) -> None:
    pct = completed / total * 100
    bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
    print(f"\r  [{bar}] {completed}/{total} ({pct:.0f}%)", end="", flush=True)


def full_refresh() -> None:
    """Bulk historical fetch: 10 years for all S&P 500 + SPX."""
    tickers = get_all_tickers() + ["SPX"]
    today = date.today()
    start = today - timedelta(days=MM_DEFAULT_LOOKBACK_DAYS)

    print(f"Full historical fetch: {len(tickers)} tickers, {start} to {today}")
    print(f"  Lookback: {MM_DEFAULT_LOOKBACK_DAYS} days (~{MM_DEFAULT_LOOKBACK_DAYS // 365} years)")

    t0 = time.time()
    prices, failed = fetch_prices(
        tickers, start.isoformat(), today.isoformat(),
        progress_callback=_progress,
    )
    elapsed = time.time() - t0
    print()  # newline after progress bar

    if not prices.empty:
        # Merge with any existing data (preserves data for tickers not in current list)
        existing = load_cached_prices()
        merged = merge_incremental(existing, prices)
        save_prices(merged)
        print(f"Saved: {len(merged.columns)} tickers, {len(merged)} days")
    else:
        print("ERROR: No data fetched!")
        sys.exit(1)

    print(f"Elapsed: {elapsed:.0f}s")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(sorted(failed))}")
    else:
        print("All tickers fetched successfully.")


def eod_refresh() -> None:
    """Daily close refresh: incremental from last cached date.

    Thin CLI wrapper around ``incremental_eod_refresh``; the two paths (this
    script and the Streamlit sidebar button) share the same implementation.
    """
    cached = load_cached_prices()
    if cached is None or cached.empty:
        print("ERROR: No cached data. Run --full first.")
        sys.exit(1)

    last_date = cached.index.max().date()
    print(f"EOD refresh: {len(cached.columns)} tickers, {last_date} to {date.today()}")

    t0 = time.time()
    result = incremental_eod_refresh(progress_callback=_progress)
    elapsed = time.time() - t0
    print()

    if result.skipped_reason:
        print(result.skipped_reason)
    else:
        print(f"Added {result.added_days} new day(s). Last date: {result.last_date}.")

    print(f"Elapsed: {elapsed:.0f}s")
    if result.failed_tickers:
        print(f"Failed ({len(result.failed_tickers)}): "
              f"{', '.join(sorted(result.failed_tickers[:20]))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market data refresh")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="Bulk 10yr historical fetch")
    group.add_argument("--eod", action="store_true", help="Daily close incremental refresh")
    args = parser.parse_args()

    if args.full:
        full_refresh()
    else:
        eod_refresh()
