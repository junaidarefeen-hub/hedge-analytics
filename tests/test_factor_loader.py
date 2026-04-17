"""Tests for analytics-side helpers in data/factor_loader.

The Prismatic round-trip itself is exercised by integration scripts, not
unit tests. These tests assert the deterministic helpers around it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.factor_loader import (
    FactorData,
    _cumulative_to_factor_data,
    _fetch_one_gadget,
)


def _build_cumulative(dates, daily_returns_by_col: dict[str, list[float]]) -> pd.DataFrame:
    """Build a cumulative DataFrame from per-column daily returns starting at 0."""
    out = {}
    for col, rets in daily_returns_by_col.items():
        assert len(rets) == len(dates), f"length mismatch for {col}"
        out[col] = np.concatenate([[0.0], np.cumsum(rets[1:])])
    return pd.DataFrame(out, index=dates)


class TestCumulativeToFactorData:
    """The cumulative-to-FactorData conversion is the only piece of pure math
    in factor_loader; the network round-trips are exercised in integration."""

    def test_returns_factordata(self):
        dates = pd.bdate_range("2026-04-01", periods=10, freq="B")
        cumulative = _build_cumulative(dates, {"Momentum": [0.0] + [0.001] * 9})
        result = _cumulative_to_factor_data(cumulative)
        assert isinstance(result, FactorData)

    def test_prices_are_one_plus_cumulative(self):
        dates = pd.bdate_range("2026-04-01", periods=5, freq="B")
        cumulative = _build_cumulative(dates, {"Momentum": [0.0, 0.01, 0.02, 0.03, 0.04]})
        result = _cumulative_to_factor_data(cumulative)
        # prices = 1 + cumulative (display index starting at 1)
        np.testing.assert_allclose(
            result.prices["Momentum"].values,
            1.0 + cumulative["Momentum"].values,
        )

    def test_returns_are_diff_of_cumulative(self):
        dates = pd.bdate_range("2026-04-01", periods=5, freq="B")
        cumulative = _build_cumulative(dates, {"Momentum": [0.0, 0.01, 0.02, 0.03, 0.04]})
        result = _cumulative_to_factor_data(cumulative)
        # First daily return is NaN (diff loses first row); the rest are deltas.
        rest = result.returns["Momentum"].iloc[1:]
        expected = cumulative["Momentum"].diff().iloc[1:]
        np.testing.assert_allclose(rest.values, expected.values)

    def test_ticker_map_uses_prefix(self):
        dates = pd.bdate_range("2026-04-01", periods=5, freq="B")
        cumulative = _build_cumulative(dates, {"Momentum": [0.0] * 5})
        result = _cumulative_to_factor_data(cumulative)
        # FACTOR_MODEL_PREFIX is 'GS' in config; ticker map should reflect that.
        assert result.ticker_map["Momentum"] == "GS_MOMENTUM"
        assert result.name_map["GS_MOMENTUM"] == "Momentum"

    def test_holiday_gaps_filled_before_diff(self):
        """Forward-fill before diff so holiday-NaN gaps yield real returns,
        not NaN. Critical for the regression engine downstream."""
        dates = pd.bdate_range("2026-04-01", periods=6, freq="B")
        # Inject a NaN holiday on day 3 (cumulative is missing)
        values = [0.0, 0.01, 0.02, np.nan, 0.04, 0.05]
        cumulative = pd.DataFrame({"Momentum": values}, index=dates)
        result = _cumulative_to_factor_data(cumulative)
        # The day after the NaN should have a finite return (ffill carries 0.02
        # forward to the NaN row, then diff between 0.02 and 0.04 = 0.02).
        assert pd.notna(result.returns["Momentum"].iloc[4])
