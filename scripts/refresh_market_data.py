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
    """Daily close refresh: incremental from last cached date."""
    cached = load_cached_prices()
    if cached is None or cached.empty:
        print("ERROR: No cached data. Run --full first.")
        sys.exit(1)

    last_date = cached.index.max().date()
    today = date.today()
    all_tickers = list(cached.columns)

    if last_date >= today:
        print(f"Cache is up to date ({last_date}). Nothing to fetch.")
        return

    print(f"EOD refresh: {len(all_tickers)} tickers, {last_date} to {today}")

    t0 = time.time()
    new_prices, failed = fetch_prices(
        all_tickers, last_date.isoformat(), today.isoformat(),
        progress_callback=_progress,
    )
    elapsed = time.time() - t0
    print()

    if not new_prices.empty:
        merged = merge_incremental(cached, new_prices)
        save_prices(merged)
        new_days = len(merged) - len(cached)
        print(f"Added {new_days} new day(s). Total: {len(merged)} days.")
    else:
        print("No new data returned.")

    print(f"Elapsed: {elapsed:.0f}s")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(sorted(failed[:20]))}")


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
