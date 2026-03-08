"""Tests for analytics.drawdown — compute_drawdowns()."""

import numpy as np
import pandas as pd
import pytest

from analytics.drawdown import DrawdownAnalysis, DrawdownPeriod, compute_drawdowns


class TestComputeDrawdowns:
    def test_monotonic_series_no_drawdowns(self):
        """Monotonically increasing series should have no drawdowns."""
        cum = pd.Series(np.linspace(1.0, 2.0, 100), index=pd.bdate_range("2022-01-03", periods=100))
        analysis = compute_drawdowns(cum)
        assert isinstance(analysis, DrawdownAnalysis)
        assert len(analysis.drawdown_periods) == 0
        assert analysis.max_drawdown == 0.0
        assert analysis.max_duration == 0

    def test_known_drawdown(self):
        """Known synthetic drawdown should be detected."""
        # Go up, then down, then recover
        values = [1.0, 1.1, 1.2, 1.0, 0.9, 1.0, 1.1, 1.3]
        cum = pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)))
        analysis = compute_drawdowns(cum, top_n=5)

        assert len(analysis.drawdown_periods) >= 1
        worst = analysis.drawdown_periods[0]
        # Worst drawdown: from peak 1.2 to trough 0.9 = -25%
        assert worst.max_drawdown == pytest.approx(-0.25, abs=0.01)

    def test_underwater_always_nonpositive(self):
        """Underwater series should always be <= 0."""
        rng = np.random.default_rng(42)
        daily = rng.normal(0.0005, 0.02, 200)
        cum = pd.Series(np.cumprod(1 + daily), index=pd.bdate_range("2022-01-03", periods=200))
        analysis = compute_drawdowns(cum)
        assert (analysis.underwater_series <= 1e-10).all()

    def test_periods_sorted_by_severity(self):
        """Drawdown periods should be sorted by severity (worst first)."""
        rng = np.random.default_rng(7)
        daily = rng.normal(0.0002, 0.02, 300)
        cum = pd.Series(np.cumprod(1 + daily), index=pd.bdate_range("2022-01-03", periods=300))
        analysis = compute_drawdowns(cum, top_n=10)
        if len(analysis.drawdown_periods) >= 2:
            for i in range(len(analysis.drawdown_periods) - 1):
                assert analysis.drawdown_periods[i].max_drawdown <= analysis.drawdown_periods[i + 1].max_drawdown

    def test_unrecovered_drawdown(self):
        """A series ending in a drawdown should have end=None."""
        values = [1.0, 1.2, 1.0, 0.8]
        cum = pd.Series(values, index=pd.bdate_range("2022-01-03", periods=4))
        analysis = compute_drawdowns(cum)
        assert len(analysis.drawdown_periods) >= 1
        # The drawdown never recovers
        last_period = analysis.drawdown_periods[0]
        assert last_period.end is None
        assert last_period.recovery_days is None

    def test_top_n_limit(self):
        """Should return at most top_n periods."""
        rng = np.random.default_rng(99)
        daily = rng.normal(0, 0.03, 500)
        cum = pd.Series(np.cumprod(1 + daily), index=pd.bdate_range("2022-01-03", periods=500))
        analysis = compute_drawdowns(cum, top_n=3)
        assert len(analysis.drawdown_periods) <= 3
