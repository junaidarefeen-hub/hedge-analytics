"""Unified daily cache refresh -- factor data + market monitor.

Refreshes both data caches from ECM (Prismatic for factors, RVX for market
monitor) and optionally commits + pushes the updated Parquet files to git
so that the Render deployment stays current.

Usage:
  python scripts/refresh_all.py                # Refresh both
  python scripts/refresh_all.py --commit       # Refresh + git commit & push
  python scripts/refresh_all.py --factors-only # Only refresh factor data
  python scripts/refresh_all.py --market-only  # Only refresh market monitor

Designed to run weekday evenings after market close (~4:45 PM ET).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecm_client import is_available

# Files to commit when --commit is used
GIT_FILES = [
    "data/cache/factor_prices.parquet",
    "data/cache/market_monitor/prices.parquet",
    "data/cache/market_monitor/metadata.json",
]


def refresh_factors() -> bool:
    """Fetch factor data from Prismatic and save to Parquet cache."""
    from data.factor_loader import _fetch_from_prismatic, _save_to_cache

    print("--- Factor Data Refresh ---")
    t0 = time.time()

    try:
        cumulative = _fetch_from_prismatic()
        _save_to_cache(cumulative)
        elapsed = time.time() - t0

        n_factors = len(cumulative.columns)
        first = cumulative.index.min().strftime("%Y-%m-%d")
        last = cumulative.index.max().strftime("%Y-%m-%d")
        print(f"  {n_factors} factors saved ({first} to {last})")
        print(f"  Elapsed: {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def refresh_market() -> bool:
    """Fetch EOD market monitor data from RVX and update Parquet cache."""
    from data.market_monitor.cache_manager import (
        load_cached_prices,
        merge_incremental,
        save_prices,
    )
    from data.market_monitor.rvx_fetcher import fetch_prices

    print("--- Market Monitor Refresh ---")

    cached = load_cached_prices()
    if cached is None or cached.empty:
        print("  ERROR: No cached data. Run 'python scripts/refresh_market_data.py --full' first.")
        return False

    last_date = cached.index.max().date()
    today = date.today()
    all_tickers = list(cached.columns)

    if last_date >= today:
        print(f"  Cache is up to date ({last_date}). Nothing to fetch.")
        return True

    print(f"  {len(all_tickers)} tickers, {last_date} to {today}")
    t0 = time.time()

    try:
        new_prices, failed = fetch_prices(
            all_tickers, last_date.isoformat(), today.isoformat(),
        )
        elapsed = time.time() - t0

        if not new_prices.empty:
            merged = merge_incremental(cached, new_prices)
            save_prices(merged)
            new_days = len(merged) - len(cached)
            print(f"  Added {new_days} new day(s). Total: {len(merged)} days.")
        else:
            print("  No new data returned.")

        print(f"  Elapsed: {elapsed:.1f}s")
        if failed:
            print(f"  Failed ({len(failed)}): {', '.join(sorted(failed[:20]))}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def git_commit_push() -> bool:
    """Stage updated cache files, commit, and push."""
    print("--- Git Commit & Push ---")

    try:
        # Stage files (only those that exist and have changes)
        files_to_add = [f for f in GIT_FILES if (PROJECT_ROOT / f).exists()]
        if not files_to_add:
            print("  No cache files to commit.")
            return True

        subprocess.run(
            ["git", "add"] + files_to_add,
            check=True, cwd=PROJECT_ROOT, capture_output=True,
        )

        # Check if anything is staged
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=PROJECT_ROOT, capture_output=True,
        )
        if result.returncode == 0:
            print("  No changes to commit.")
            return True

        # Commit
        today = date.today().isoformat()
        msg = f"chore: refresh market data caches ({today})"
        subprocess.run(
            ["git", "commit", "-m", msg],
            check=True, cwd=PROJECT_ROOT, capture_output=True,
        )
        print(f"  Committed: {msg}")

        # Push
        subprocess.run(
            ["git", "push"],
            check=True, cwd=PROJECT_ROOT, capture_output=True,
        )
        print("  Pushed to remote.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e}")
        if e.stderr:
            print(f"  {e.stderr.decode().strip()}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Daily data cache refresh")
    parser.add_argument("--commit", action="store_true", help="Git commit + push after refresh")
    parser.add_argument("--factors-only", action="store_true", help="Only refresh factor data")
    parser.add_argument("--market-only", action="store_true", help="Only refresh market monitor")
    args = parser.parse_args()

    if not is_available():
        print("ERROR: ECM credentials not available. Cannot refresh.")
        sys.exit(1)

    results = {}

    if not args.market_only:
        results["factors"] = refresh_factors()

    if not args.factors_only:
        results["market"] = refresh_market()

    # Summary
    print("\n--- Summary ---")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    if args.commit and any(results.values()):
        git_commit_push()

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
