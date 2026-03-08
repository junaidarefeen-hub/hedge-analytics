"""Tests for analytics.stress — stress testing scenarios."""

import numpy as np
import pandas as pd
import pytest

from analytics.stress import (
    ScenarioResult,
    StressTestResult,
    _run_custom_scenario,
    _run_historical_scenario,
    run_stress_test,
)


# ---------------------------------------------------------------------------
# Custom scenarios (no historical data needed)
# ---------------------------------------------------------------------------

class TestCustomScenario:
    def test_basic_pnl(self):
        """Target drops 10%, hedge instrument rises 5%: check P&L math."""
        result = _run_custom_scenario(
            target="AAPL",
            hedge_instruments=["SPY"],
            weights=np.array([-1.0]),
            notional=1_000_000,
            shocks={"AAPL": -10, "SPY": -15},
            name="Test Crash",
        )
        assert isinstance(result, ScenarioResult)
        assert result.is_custom is True
        # Unhedged P&L: 1M × -10% = -100k
        assert result.unhedged_pnl == pytest.approx(-100_000)
        # Hedge P&L: 1M × (-1.0) × (-15%) = +150k
        # Hedged P&L = unhedged + hedge = -100k + 150k = 50k
        assert result.hedged_pnl == pytest.approx(50_000)
        assert result.hedge_benefit == pytest.approx(150_000)

    def test_no_shock_means_zero(self):
        result = _run_custom_scenario(
            "AAPL", ["SPY"], np.array([-1.0]), 1_000_000,
            shocks={}, name="No shock",
        )
        assert result.unhedged_pnl == pytest.approx(0.0)
        assert result.hedged_pnl == pytest.approx(0.0)

    def test_multiple_hedges(self):
        result = _run_custom_scenario(
            "AAPL", ["SPY", "QQQ"], np.array([-0.5, -0.5]), 1_000_000,
            shocks={"AAPL": -20, "SPY": -10, "QQQ": -15},
            name="Multi-hedge",
        )
        # Unhedged: 1M × -20% = -200k
        assert result.unhedged_pnl == pytest.approx(-200_000)
        # SPY hedge: 1M × (-0.5) × (-10%) = 50k
        # QQQ hedge: 1M × (-0.5) × (-15%) = 75k
        # Hedged: -200k + 50k + 75k = -75k
        assert result.hedged_pnl == pytest.approx(-75_000)


# ---------------------------------------------------------------------------
# Historical scenarios
# ---------------------------------------------------------------------------

class TestHistoricalScenario:
    @pytest.fixture()
    def crisis_returns(self):
        """Synthetic returns covering a 'crisis' window."""
        dates = pd.bdate_range("2020-02-01", "2020-04-30")
        rng = np.random.default_rng(7)
        n = len(dates)
        return pd.DataFrame({
            "AAPL": rng.normal(-0.005, 0.03, n),
            "SPY": rng.normal(-0.004, 0.025, n),
        }, index=dates)

    def test_runs_within_date_range(self, crisis_returns):
        scenario = {
            "name": "COVID Test",
            "start": "2020-02-19",
            "end": "2020-03-23",
            "description": "Test",
        }
        result = _run_historical_scenario(
            crisis_returns, "AAPL", ["SPY"], np.array([-1.0]), 1_000_000, scenario,
        )
        assert isinstance(result, ScenarioResult)
        assert result.is_custom is False
        assert result.n_days > 0

    def test_insufficient_data_raises(self, crisis_returns):
        scenario = {
            "name": "Future",
            "start": "2025-01-01",
            "end": "2025-06-01",
            "description": "No data",
        }
        with pytest.raises(ValueError, match="Insufficient data"):
            _run_historical_scenario(
                crisis_returns, "AAPL", ["SPY"], np.array([-1.0]), 1_000_000, scenario,
            )


# ---------------------------------------------------------------------------
# Full stress test orchestrator
# ---------------------------------------------------------------------------

class TestRunStressTest:
    def test_custom_only(self, returns):
        result = run_stress_test(
            returns, "AAPL", ["SPY", "MSFT"], np.array([-0.5, -0.5]),
            1_000_000, "Min Var",
            custom_scenarios=[
                {"name": "Mild", "shocks": {"AAPL": -5, "SPY": -3, "MSFT": -4}},
                {"name": "Severe", "shocks": {"AAPL": -30, "SPY": -25, "MSFT": -20}},
            ],
        )
        assert isinstance(result, StressTestResult)
        assert len(result.scenarios) == 2
        assert len(result.summary_df) == 2

    def test_summary_df_columns(self, returns):
        result = run_stress_test(
            returns, "AAPL", ["SPY"], np.array([-1.0]), 1_000_000, "Min Var",
            custom_scenarios=[{"name": "X", "shocks": {"AAPL": -10, "SPY": -8}}],
        )
        expected_cols = {
            "Scenario", "Type", "Days", "Unhedged Return", "Hedged Return",
            "Unhedged P&L ($)", "Hedged P&L ($)", "Hedge Benefit ($)", "Hedge Benefit (pp)",
        }
        assert expected_cols == set(result.summary_df.columns)

    def test_no_scenarios_raises(self, returns):
        with pytest.raises(ValueError, match="No scenarios"):
            run_stress_test(
                returns, "AAPL", ["SPY"], np.array([-1.0]), 1_000_000, "X",
            )

    def test_skipped_scenarios_tracked(self):
        """Historical scenarios outside data range should be skipped, not crash."""
        dates = pd.bdate_range("2023-01-01", periods=100)
        rng = np.random.default_rng(3)
        r = pd.DataFrame({"A": rng.normal(0, 0.01, 100), "B": rng.normal(0, 0.01, 100)}, index=dates)
        result = run_stress_test(
            r, "A", ["B"], np.array([-1.0]), 1_000_000, "X",
            selected_scenarios=[{
                "name": "Old Crisis", "start": "2008-09-01", "end": "2009-03-09",
                "description": "No data for this",
            }],
            custom_scenarios=[{"name": "Fallback", "shocks": {"A": -5, "B": -3}}],
        )
        assert "Old Crisis" in result.skipped
        assert len(result.scenarios) == 1  # only the custom survived
