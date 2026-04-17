"""Tests for analytics.universe_regime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.universe_regime import (
    latest_regime_label,
    regime_conditional_stats,
)


@pytest.fixture()
def regime_universe(rng):
    """Synthetic 5-ticker universe with a clear vol-regime structure on SPX."""
    n_days = 400
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    # SPX: low vol first half, high vol second half — gives clean regimes.
    half = n_days // 2
    spx_rets = np.concatenate([
        rng.normal(0.0005, 0.005, half),
        rng.normal(-0.0002, 0.025, n_days - half),
    ])
    spx = 100 * np.exp(np.cumsum(spx_rets))
    tickers = {f"T{i:02d}": rng.normal(0.0003, 0.012 + 0.005 * i, n_days) for i in range(5)}
    prices = pd.DataFrame(
        {tk: 100 * np.exp(np.cumsum(rets)) for tk, rets in tickers.items()},
        index=dates,
    )
    spx_returns = pd.Series(spx_rets, index=dates, name="SPX")
    return prices, spx_returns


class TestRegimeConditionalStats:
    def test_columns(self, regime_universe):
        prices, spx = regime_universe
        df = regime_conditional_stats(prices, spx)
        for col in ["ticker", "regime_label", "avg_return_ann",
                    "vol_ann", "sharpe", "vs_spx_alpha", "days"]:
            assert col in df.columns

    def test_three_regimes_present(self, regime_universe):
        prices, spx = regime_universe
        df = regime_conditional_stats(prices, spx)
        assert df["regime_label"].nunique() == 3

    def test_per_ticker_per_regime(self, regime_universe):
        """For each ticker, expect one row per regime (subject to min_days)."""
        prices, spx = regime_universe
        df = regime_conditional_stats(prices, spx, min_days_per_regime=2)
        for ticker in prices.columns:
            ticker_rows = df[df["ticker"] == ticker]
            assert ticker_rows["regime_label"].nunique() <= 3

    def test_high_vol_regime_has_higher_vol(self, regime_universe):
        prices, spx = regime_universe
        df = regime_conditional_stats(prices, spx)
        # Average vol across tickers in High Vol regime > average vol in Low Vol regime.
        high_vol = df[df["regime_label"] == "High Vol"]["vol_ann"].mean()
        low_vol = df[df["regime_label"] == "Low Vol"]["vol_ann"].mean()
        assert high_vol > low_vol

    def test_min_days_filters_short_buckets(self, regime_universe):
        prices, spx = regime_universe
        df = regime_conditional_stats(prices, spx, min_days_per_regime=10)
        assert (df["days"] >= 10).all()


class TestLatestRegimeLabel:
    def test_returns_string(self, regime_universe):
        _, spx = regime_universe
        label = latest_regime_label(spx)
        assert label in ("Low Vol", "Normal", "High Vol")
