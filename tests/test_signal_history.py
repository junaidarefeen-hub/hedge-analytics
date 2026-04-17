"""Tests for analytics.signal_history — snapshots, deltas, watchlist."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.signal_history import (
    SignalDelta,
    SignalSnapshot,
    WatchlistEntry,
    add_ticker,
    append_snapshot,
    available_dates,
    build_snapshot,
    compute_deltas,
    load_history,
    load_watchlist,
    remove_ticker,
    save_watchlist,
)


@pytest.fixture()
def store_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture()
def sample_prices(rng):
    """Synthetic 10-ticker, 300-day price frame."""
    n_days = 300
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(10)]
    market = rng.normal(0.0003, 0.01, n_days)
    out = {}
    for i, tk in enumerate(tickers):
        beta = 0.6 + (i % 3) * 0.2
        idio = rng.normal(0, 0.013, n_days)
        out[tk] = np.exp(np.cumsum(0.0002 + beta * market + idio)) * 100
    return pd.DataFrame(out, index=dates)


def _make_snapshot(date: str, **overrides) -> SignalSnapshot:
    """Helper: construct a hand-crafted snapshot with a single ticker."""
    base = dict(
        composite=pd.Series({"AAPL": 30.0}),
        rsi=pd.Series({"AAPL": 45.0}),
        ma_distance_50d=pd.Series({"AAPL": 0.02}),
        rolling_max_252d=pd.Series({"AAPL": 200.0}),
        rolling_min_252d=pd.Series({"AAPL": 100.0}),
        last_price=pd.Series({"AAPL": 150.0}),
        factor_trends=None,
    )
    base.update(overrides)
    return SignalSnapshot(date=pd.Timestamp(date).normalize(), **base)


class TestBuildSnapshot:
    def test_returns_signalsnapshot(self, sample_prices):
        snap = build_snapshot(sample_prices)
        assert isinstance(snap, SignalSnapshot)
        assert snap.date == sample_prices.index[-1].normalize()

    def test_composite_keys_match_universe(self, sample_prices):
        snap = build_snapshot(sample_prices)
        assert set(snap.composite.dropna().index) <= set(sample_prices.columns)

    def test_no_factor_returns_yields_none(self, sample_prices):
        snap = build_snapshot(sample_prices, factor_returns=None)
        assert snap.factor_trends is None


class TestPersistence:
    def test_append_creates_file(self, store_dir):
        snap = _make_snapshot("2024-04-15")
        append_snapshot(snap, store_dir=store_dir)
        assert (store_dir / "signal_history.parquet").exists()

    def test_append_is_idempotent(self, store_dir):
        snap = _make_snapshot("2024-04-15")
        append_snapshot(snap, store_dir=store_dir)
        append_snapshot(snap, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        # Six numeric metrics * 1 ticker = 6 rows; duplicating the date should
        # NOT double-count (idempotent overwrite).
        assert len(history) == 6

    def test_two_dates_keep_both(self, store_dir):
        append_snapshot(_make_snapshot("2024-04-14"), store_dir=store_dir)
        append_snapshot(_make_snapshot("2024-04-15"), store_dir=store_dir)
        dates = available_dates(store_dir=store_dir)
        assert len(dates) == 2

    def test_lookback_days_limits_returned_dates(self, store_dir):
        for d in ["2024-04-10", "2024-04-11", "2024-04-12", "2024-04-15"]:
            append_snapshot(_make_snapshot(d), store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=2)
        assert history["date"].nunique() == 2

    def test_load_returns_empty_when_missing(self, store_dir):
        df = load_history(store_dir=store_dir)
        assert df.empty
        assert list(df.columns) == ["date", "key", "metric", "value", "label"]


class TestComputeDeltas:
    def test_entered_oversold(self, store_dir):
        prior = _make_snapshot("2024-04-14", composite=pd.Series({"AAPL": 40.0}))
        current = _make_snapshot("2024-04-15", composite=pd.Series({"AAPL": 18.0}))
        append_snapshot(prior, store_dir=store_dir)
        append_snapshot(current, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        deltas = compute_deltas(history, current.date, prior.date)
        assert any(d.category == "Entered Oversold" and d.ticker == "AAPL" for d in deltas)

    def test_exited_oversold(self, store_dir):
        prior = _make_snapshot("2024-04-14", composite=pd.Series({"AAPL": 18.0}))
        current = _make_snapshot("2024-04-15", composite=pd.Series({"AAPL": 30.0}))
        append_snapshot(prior, store_dir=store_dir)
        append_snapshot(current, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        deltas = compute_deltas(history, current.date, prior.date)
        assert any(d.category == "Exited Oversold" for d in deltas)

    def test_rsi_flip(self, store_dir):
        prior = _make_snapshot("2024-04-14", rsi=pd.Series({"AAPL": 65.0}))
        current = _make_snapshot("2024-04-15", rsi=pd.Series({"AAPL": 75.0}))
        append_snapshot(prior, store_dir=store_dir)
        append_snapshot(current, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        deltas = compute_deltas(history, current.date, prior.date)
        assert any(d.category == "RSI Flip >70" for d in deltas)

    def test_broke_50d_ma(self, store_dir):
        prior = _make_snapshot("2024-04-14", ma_distance_50d=pd.Series({"AAPL": 0.01}))
        current = _make_snapshot("2024-04-15", ma_distance_50d=pd.Series({"AAPL": -0.02}))
        append_snapshot(prior, store_dir=store_dir)
        append_snapshot(current, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        deltas = compute_deltas(history, current.date, prior.date)
        assert any(d.category == "Broke 50d MA" for d in deltas)

    def test_new_52w_high(self, store_dir):
        prior = _make_snapshot(
            "2024-04-14",
            last_price=pd.Series({"AAPL": 195.0}),
            rolling_max_252d=pd.Series({"AAPL": 200.0}),
            rolling_min_252d=pd.Series({"AAPL": 100.0}),
        )
        current = _make_snapshot(
            "2024-04-15",
            last_price=pd.Series({"AAPL": 205.0}),
            rolling_max_252d=pd.Series({"AAPL": 205.0}),
            rolling_min_252d=pd.Series({"AAPL": 100.0}),
        )
        append_snapshot(prior, store_dir=store_dir)
        append_snapshot(current, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        deltas = compute_deltas(history, current.date, prior.date)
        assert any(d.category == "New 52w High" for d in deltas)

    def test_factor_trend_flip(self, store_dir):
        prior = _make_snapshot(
            "2024-04-14", factor_trends=pd.Series({"Momentum": "Neutral"}),
        )
        current = _make_snapshot(
            "2024-04-15", factor_trends=pd.Series({"Momentum": "Trending Up"}),
        )
        append_snapshot(prior, store_dir=store_dir)
        append_snapshot(current, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        deltas = compute_deltas(history, current.date, prior.date)
        assert any(
            d.category == "Factor Trend Flip" and d.ticker == "Momentum"
            for d in deltas
        )

    def test_no_change_yields_empty(self, store_dir):
        snap = _make_snapshot("2024-04-15")
        append_snapshot(snap, store_dir=store_dir)
        # Same snapshot for both dates => no deltas
        snap2 = _make_snapshot("2024-04-16")  # exactly the same values
        append_snapshot(snap2, store_dir=store_dir)
        history = load_history(store_dir=store_dir, lookback_days=0)
        deltas = compute_deltas(history, snap2.date, snap.date)
        assert deltas == []


class TestWatchlist:
    def test_add_creates_entry(self, store_dir):
        add_ticker("AAPL", note="oversold play", tags=["mega-cap"], store_dir=store_dir)
        wl = load_watchlist(store_dir=store_dir)
        assert len(wl) == 1
        assert wl[0].ticker == "AAPL"
        assert wl[0].note == "oversold play"
        assert wl[0].tags == ["mega-cap"]

    def test_add_existing_updates_in_place(self, store_dir):
        add_ticker("AAPL", note="first", store_dir=store_dir)
        add_ticker("AAPL", note="second", tags=["v2"], store_dir=store_dir)
        wl = load_watchlist(store_dir=store_dir)
        assert len(wl) == 1
        assert wl[0].note == "second"
        assert wl[0].tags == ["v2"]

    def test_remove_ticker(self, store_dir):
        add_ticker("AAPL", store_dir=store_dir)
        add_ticker("MSFT", store_dir=store_dir)
        remove_ticker("AAPL", store_dir=store_dir)
        wl = load_watchlist(store_dir=store_dir)
        assert {e.ticker for e in wl} == {"MSFT"}

    def test_load_returns_empty_when_missing(self, store_dir):
        assert load_watchlist(store_dir=store_dir) == []

    def test_save_round_trip(self, store_dir):
        entries = [
            WatchlistEntry(ticker="AAPL", added_on="2024-04-15T10:00:00+00:00",
                           note="x", tags=["a"]),
        ]
        save_watchlist(entries, store_dir=store_dir)
        loaded = load_watchlist(store_dir=store_dir)
        assert loaded == entries
