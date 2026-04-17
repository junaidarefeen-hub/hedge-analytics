"""Tests for analytics.dispersion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.dispersion import (
    DispersionSnapshot,
    compute_dispersion_history,
    current_dispersion,
)


@pytest.fixture()
def universe_prices(rng):
    """Synthetic prices for 12 tickers spread across 3 sectors over 400 days."""
    n_days = 400
    dates = pd.bdate_range("2023-01-02", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(12)]
    market = rng.normal(0.0003, 0.01, n_days)
    data = {}
    for i, tk in enumerate(tickers):
        beta = 0.5 + (i % 3) * 0.3
        idio = rng.normal(0, 0.012 + (i % 4) * 0.003, n_days)
        log_ret = 0.0002 + beta * market + idio
        data[tk] = np.exp(np.cumsum(log_ret)) * 100
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def sector_map():
    return {f"T{i:02d}": ["Tech", "Energy", "Financials"][i % 3] for i in range(12)}


class TestComputeDispersionHistory:
    def test_columns(self, universe_prices, sector_map):
        h = compute_dispersion_history(universe_prices, sector_map, lookback_days=60)
        assert list(h.columns) == ["cs_std", "sector_std"]

    def test_length_matches_lookback(self, universe_prices, sector_map):
        h = compute_dispersion_history(universe_prices, sector_map, lookback_days=60)
        assert len(h) == 60

    def test_values_non_negative(self, universe_prices, sector_map):
        h = compute_dispersion_history(universe_prices, sector_map, lookback_days=60)
        assert (h["cs_std"] >= 0).all()
        assert (h["sector_std"] >= 0).all()

    def test_returns_empty_when_no_constituents(self, universe_prices):
        h = compute_dispersion_history(universe_prices, {}, lookback_days=60)
        assert h.empty

    def test_unknown_tickers_excluded(self, universe_prices, sector_map):
        # Add a non-mapped ticker; it should not influence cs_std.
        prices = universe_prices.copy()
        prices["UNKNOWN"] = 100.0
        h = compute_dispersion_history(prices, sector_map, lookback_days=60)
        h_baseline = compute_dispersion_history(universe_prices, sector_map, lookback_days=60)
        np.testing.assert_allclose(h["cs_std"].values, h_baseline["cs_std"].values)


class TestCurrentDispersion:
    def test_returns_dataclass(self, universe_prices, sector_map):
        snap = current_dispersion(universe_prices, sector_map, lookback_days=60)
        assert isinstance(snap, DispersionSnapshot)

    def test_pctile_within_bounds(self, universe_prices, sector_map):
        snap = current_dispersion(universe_prices, sector_map, lookback_days=60)
        assert 0 <= snap.historical_pctile_cs <= 100
        assert 0 <= snap.historical_pctile_sector <= 100

    def test_top_decile_flag_when_extreme(self, sector_map, rng):
        """A constructed dataset where the last day is the most-volatile day
        should produce historical_pctile == 100 and top_decile_flag True."""
        n_days = 100
        dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
        # Tame returns for first 99 days, then a huge dispersion shock on the last day.
        normal = rng.normal(0, 0.005, (n_days - 1, 12))
        shock = np.array([rng.normal(0, 0.05, 12)])
        rets = np.vstack([normal, shock])
        # Convert to a price path (start at 100 for all).
        prices_arr = 100 * np.exp(np.cumsum(rets, axis=0))
        # The first row of pct_change is NaN, so prepend a synthetic baseline row
        # to anchor the dispersion calculation properly.
        prices_arr = np.vstack([np.full((1, 12), 100.0), prices_arr])
        dates_full = pd.bdate_range("2024-01-01", periods=n_days + 1, freq="B")
        prices = pd.DataFrame(
            prices_arr,
            index=dates_full,
            columns=[f"T{i:02d}" for i in range(12)],
        )
        snap = current_dispersion(prices, sector_map, lookback_days=80)
        assert snap.historical_pctile_cs == pytest.approx(100.0)
        assert snap.top_decile_flag is True

    def test_history_attached_for_sparkline(self, universe_prices, sector_map):
        snap = current_dispersion(universe_prices, sector_map, lookback_days=60)
        assert len(snap.history_cs) == 60
        assert len(snap.history_sector) == 60

    def test_empty_universe_returns_nan_snapshot(self, universe_prices):
        snap = current_dispersion(universe_prices, {}, lookback_days=60)
        assert np.isnan(snap.cross_sectional_std)
        assert snap.top_decile_flag is False
