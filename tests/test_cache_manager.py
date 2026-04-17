"""Tests for data.market_monitor.cache_manager.

The Parquet I/O is exercised by writing to ``tmp_path``; the RVX network
round-trip is mocked out via ``monkeypatch`` on ``fetch_prices``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data.market_monitor import cache_manager
from data.market_monitor.cache_manager import (
    EodRefreshResult,
    incremental_eod_refresh,
)


@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    """Redirect the cache module to a tmp directory so tests don't touch the
    real Parquet file."""
    monkeypatch.setattr(cache_manager, "_PRICES_FILE", tmp_path / "prices.parquet")
    monkeypatch.setattr(cache_manager, "_METADATA_FILE", tmp_path / "metadata.json")
    return tmp_path


def _fake_cache(start_date: str, n_days: int, tickers: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range(start_date, periods=n_days, freq="B")
    rng = np.random.default_rng(42)
    data = {t: 100 + rng.standard_normal(n_days).cumsum() for t in tickers}
    return pd.DataFrame(data, index=dates)


class TestIncrementalEodRefresh:
    def test_no_cache_returns_skipped_reason(self, tmp_cache):
        result = incremental_eod_refresh()
        assert isinstance(result, EodRefreshResult)
        assert result.added_days == 0
        assert "No cached data" in result.skipped_reason

    def test_cache_already_current_skipped(self, tmp_cache):
        df = _fake_cache(str(date.today()), 1, ["AAPL", "MSFT"])
        cache_manager.save_prices(df)
        result = incremental_eod_refresh()
        assert result.added_days == 0
        assert "already current" in result.skipped_reason

    def test_appends_new_rows_and_calls_fetch_with_correct_range(self, tmp_cache):
        # Cache ends 5 days ago.
        last_date = date.today() - timedelta(days=5)
        cached = _fake_cache(last_date.isoformat(), 1, ["AAPL", "MSFT"])
        cached.index = pd.DatetimeIndex([pd.Timestamp(last_date)])
        cache_manager.save_prices(cached)

        # Fake RVX response: 4 new business days.
        new_dates = pd.bdate_range(last_date + timedelta(days=1), periods=4, freq="B")
        new_df = pd.DataFrame(
            {"AAPL": [101, 102, 103, 104], "MSFT": [201, 202, 203, 204]},
            index=new_dates,
        )

        with patch(
            "data.market_monitor.rvx_fetcher.fetch_prices",
            return_value=(new_df, []),
        ) as mock_fetch:
            result = incremental_eod_refresh()

        assert result.skipped_reason is None
        assert result.added_days == 4
        assert result.failed_tickers == []
        # The fetch should have been called with the cache's last_date as the
        # start (so RVX is asked to include that date as overlap, allowing
        # ``merge_incremental`` to pick the freshest value).
        kwargs = mock_fetch.call_args.kwargs if mock_fetch.call_args.kwargs else {}
        args = mock_fetch.call_args.args
        # fetch_prices(tickers, start_date, end_date, progress_callback=...)
        passed_tickers, passed_start, passed_end = args[0], args[1], args[2]
        assert set(passed_tickers) == {"AAPL", "MSFT"}
        assert passed_start == last_date.isoformat()
        assert passed_end == date.today().isoformat()

        # Cache should now contain the new rows.
        merged = cache_manager.load_cached_prices()
        assert len(merged) == 1 + 4  # original + new
        assert merged.index.max().date() == new_dates.max().date()

    def test_failed_tickers_passed_through(self, tmp_cache):
        last_date = date.today() - timedelta(days=3)
        cached = _fake_cache(last_date.isoformat(), 1, ["AAPL", "MSFT", "BAD"])
        cached.index = pd.DatetimeIndex([pd.Timestamp(last_date)])
        cache_manager.save_prices(cached)

        new_dates = pd.bdate_range(last_date + timedelta(days=1), periods=2, freq="B")
        new_df = pd.DataFrame(
            {"AAPL": [101, 102], "MSFT": [201, 202]}, index=new_dates,
        )

        with patch(
            "data.market_monitor.rvx_fetcher.fetch_prices",
            return_value=(new_df, ["BAD"]),
        ):
            result = incremental_eod_refresh()
        assert result.failed_tickers == ["BAD"]
        assert result.added_days == 2

    def test_empty_fetch_returns_skipped_reason(self, tmp_cache):
        last_date = date.today() - timedelta(days=3)
        cached = _fake_cache(last_date.isoformat(), 1, ["AAPL"])
        cached.index = pd.DatetimeIndex([pd.Timestamp(last_date)])
        cache_manager.save_prices(cached)

        with patch(
            "data.market_monitor.rvx_fetcher.fetch_prices",
            return_value=(pd.DataFrame(), ["AAPL"]),
        ):
            result = incremental_eod_refresh()
        assert result.added_days == 0
        assert "no new data" in result.skipped_reason.lower()
        assert result.failed_tickers == ["AAPL"]

    def test_progress_callback_forwarded(self, tmp_cache):
        last_date = date.today() - timedelta(days=3)
        cached = _fake_cache(last_date.isoformat(), 1, ["AAPL"])
        cached.index = pd.DatetimeIndex([pd.Timestamp(last_date)])
        cache_manager.save_prices(cached)

        new_dates = pd.bdate_range(last_date + timedelta(days=1), periods=2, freq="B")
        new_df = pd.DataFrame({"AAPL": [101, 102]}, index=new_dates)

        with patch(
            "data.market_monitor.rvx_fetcher.fetch_prices",
            return_value=(new_df, []),
        ) as mock_fetch:
            sentinel = object()
            incremental_eod_refresh(progress_callback=sentinel)
            kwargs = mock_fetch.call_args.kwargs
            assert kwargs.get("progress_callback") is sentinel
