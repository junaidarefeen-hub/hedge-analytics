"""Tests for analytics.universe_drawdown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.universe_drawdown import (
    UniverseDrawdownRow,
    rank_underwater_basing,
)


@pytest.fixture()
def crashing_prices(rng):
    """Synthetic prices: a few tickers with deep, ongoing drawdowns and others
    that recovered or never drew down."""
    n_days = 400
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")

    # T_DEEP: peak at day 100, then -40% by day 200, slowly basing through 400.
    deep = np.concatenate([
        np.linspace(100, 150, 100),  # uptrend to peak
        np.linspace(150, 90, 100),   # crash to -40% from peak
        np.linspace(90, 110, 200),   # slow recovery, still ~-27% from peak
    ])

    # T_RECOVERED: peak at day 50, fell, fully recovered by day 250.
    recovered = np.concatenate([
        np.linspace(100, 130, 50),
        np.linspace(130, 100, 100),
        np.linspace(100, 140, 250),
    ])

    # T_FLAT: gentle uptrend, no drawdown.
    flat = np.linspace(100, 130, n_days)

    df = pd.DataFrame(
        {"T_DEEP": deep, "T_RECOVERED": recovered, "T_FLAT": flat},
        index=dates,
    )
    # Add small noise to avoid degenerate stats.
    df += rng.normal(0, 0.4, df.shape)
    return df.clip(lower=1.0)


class TestRankUnderwaterBasing:
    def test_picks_deep_drawdown_ticker(self, crashing_prices):
        df = rank_underwater_basing(
            crashing_prices, min_dd=-0.15, min_days_underwater=40,
        )
        assert "T_DEEP" in df["ticker"].values

    def test_excludes_flat_ticker(self, crashing_prices):
        df = rank_underwater_basing(
            crashing_prices, min_dd=-0.15, min_days_underwater=40,
        )
        assert "T_FLAT" not in df["ticker"].values

    def test_excludes_recovered_ticker(self, crashing_prices):
        df = rank_underwater_basing(
            crashing_prices, min_dd=-0.15, min_days_underwater=40,
        )
        # T_RECOVERED is at new highs by end of period, so it shouldn't be flagged.
        assert "T_RECOVERED" not in df["ticker"].values

    def test_columns(self, crashing_prices):
        df = rank_underwater_basing(crashing_prices)
        for col in [
            "ticker", "current_dd", "peak_date", "trough_date",
            "days_underwater", "recovery_from_trough_pct",
            "composite_today", "composite_60d_improvement",
        ]:
            assert col in df.columns

    def test_dd_below_threshold(self, crashing_prices):
        df = rank_underwater_basing(crashing_prices, min_dd=-0.15)
        assert (df["current_dd"] <= -0.15).all()

    def test_min_days_underwater_filter(self, crashing_prices):
        df = rank_underwater_basing(
            crashing_prices, min_days_underwater=10,
        )
        # All surfaced rows must respect the filter.
        assert (df["days_underwater"] >= 10).all()

    def test_returns_empty_for_short_series(self):
        prices = pd.DataFrame(
            {"X": np.linspace(100, 90, 20)},
            index=pd.bdate_range("2024-01-02", periods=20, freq="B"),
        )
        df = rank_underwater_basing(prices)
        assert df.empty
