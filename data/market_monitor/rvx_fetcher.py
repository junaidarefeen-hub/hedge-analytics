"""Batch RVX price fetcher for S&P 500 constituents.

Uses ThreadPoolExecutor to fetch prices in parallel via the ECM MCP client.
Supports two modes:
  - Historical: daily close prices (default)
  - Intraday: includes live price for current day via hist_live mode
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

from data.ecm_client import ECMAuthError, ECMFetchError, fetch_rvx_series

logger = logging.getLogger(__name__)

# Max parallel RVX calls — conservative to avoid overwhelming MCP server
MAX_WORKERS = 15


def _ticker_to_rvx(symbol: str) -> str:
    """Convert a plain ticker symbol to RVX EQSN format.

    Examples:
        AAPL  -> EQSN AAPL
        BRK-B -> EQSN BRK/B  (multi-class shares use slash)
        SPX   -> SPX  (indices passed through)
    """
    # Known index tickers that don't use EQSN prefix
    if symbol in ("SPX", "RTY", "NDX", "DJIA"):
        return symbol
    # Multi-class tickers: dash → slash for RVX (e.g. BRK-B → BRK/B)
    rvx_symbol = symbol.replace("-", "/")
    return f"EQSN {rvx_symbol}"


def _fetch_single(
    symbol: str, start_date: str, end_date: str, intraday: bool = False,
) -> tuple[str, pd.Series | None]:
    """Fetch price series for a single ticker. Returns (symbol, series_or_None)."""
    rvx_ticker = _ticker_to_rvx(symbol)
    try:
        raw = fetch_rvx_series(
            ticker=rvx_ticker,
            attribute="EQD(PRICE)",
            start_date=start_date,
            end_date=end_date,
            format="json",
            intraday_mode="hist_live" if intraday else None,
        )
        # Parse JSON response: [{"date":"YYYY-MM-DD","value":123.45}, ...]
        data = json.loads(raw)
        if not data:
            logger.warning(f"Empty response for {symbol}")
            return symbol, None

        dates = [d["date"] for d in data]
        values = [d["value"] for d in data]
        series = pd.Series(values, index=pd.DatetimeIndex(dates), name=symbol, dtype=float)
        return symbol, series

    except (ECMAuthError, ECMFetchError) as e:
        logger.warning(f"Failed to fetch {symbol}: {e}")
        return symbol, None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse response for {symbol}: {e}")
        return symbol, None
    except Exception as e:
        # Catch transient errors (502, connection resets, timeouts)
        logger.warning(f"Unexpected error fetching {symbol}: {e}")
        return symbol, None


def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    max_workers: int = MAX_WORKERS,
    intraday: bool = False,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch daily prices for multiple tickers from RVX in parallel.

    Args:
        tickers: List of ticker symbols (e.g. ["AAPL", "MSFT", "SPX"]).
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        max_workers: Max parallel RVX calls.
        intraday: If True, use hist_live mode to include current live price.
        progress_callback: Optional callable(completed, total) for progress updates.

    Returns:
        (prices_df, failed_tickers) — DataFrame with DatetimeIndex x ticker columns,
        and list of tickers that failed to fetch.
    """
    series_dict: dict[str, pd.Series] = {}
    failed: list[str] = []
    total = len(tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_single, ticker, start_date, end_date, intraday): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            symbol, series = future.result()
            completed += 1
            if series is not None:
                series_dict[symbol] = series
            else:
                failed.append(symbol)
            if progress_callback:
                progress_callback(completed, total)

    if not series_dict:
        return pd.DataFrame(), failed

    prices = pd.DataFrame(series_dict)
    prices.index.name = "date"
    prices = prices.sort_index()

    logger.info(
        f"Fetched {len(series_dict)}/{total} tickers "
        f"({len(failed)} failed, {len(prices)} days)"
    )
    return prices, failed
