"""Tests for analytics.intraday_rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.intraday_rotation import (
    derive_inputs_from_snapshot,
    rank_rotation_candidates,
)


@pytest.fixture()
def sector_map():
    # 3 sectors, 4 tickers each
    sectors = ["Tech", "Energy", "Financials"]
    return {f"T{i:02d}": sectors[i // 4] for i in range(12)}


@pytest.fixture()
def name_map():
    return {f"T{i:02d}": f"Company {i}" for i in range(12)}


def _build_inputs(daily_returns_dict, sector_map):
    daily = pd.Series(daily_returns_dict, name="ret")
    sector_ret = (
        daily.groupby(daily.index.map(sector_map)).mean()
    )
    composite = pd.Series({t: 50.0 for t in daily.index})
    return daily, sector_ret, composite


class TestRankRotationCandidates:
    def test_lagger_in_leader_picked(self, sector_map, name_map):
        # Tech: T00-T03. Make Tech the leader by giving 3/4 strong positive returns.
        # T00 lags inside Tech (down day) — should be flagged.
        # Energy and Financials are flat/negative.
        daily = pd.Series({
            "T00": -0.02,  # Tech laggard
            "T01": +0.03, "T02": +0.03, "T03": +0.03,
            "T04": -0.005, "T05": -0.005, "T06": -0.005, "T07": -0.005,  # Energy
            "T08": -0.005, "T09": -0.005, "T10": -0.005, "T11": -0.005,  # Financials
        })
        sector_ret = daily.groupby(daily.index.map(sector_map)).mean()
        composite = pd.Series({t: 30.0 if t == "T00" else 60.0 for t in daily.index})

        df = rank_rotation_candidates(
            daily, sector_ret, composite, sector_map, name_map,
            leader_pctile=66.0, laggard_pctile=33.0,
        )
        assert "T00" in df["ticker"].values
        row = df[df["ticker"] == "T00"].iloc[0]
        assert row["signal"] == "Lagger in Leader"
        assert row["sector"] == "Tech"
        assert row["composite_score"] == pytest.approx(30.0)

    def test_leader_in_laggard_picked(self, sector_map, name_map):
        # Energy is the unambiguous laggard (mean ~ -0.013). T04 bucks the trend.
        # Tech is mid (mean ~ 0.005), Financials is leader (mean ~ +0.015).
        daily = pd.Series({
            "T00": +0.005, "T01": +0.005, "T02": +0.005, "T03": +0.005,
            "T04": +0.04,  # Energy outlier UP
            "T05": -0.025, "T06": -0.025, "T07": -0.025,
            "T08": +0.015, "T09": +0.015, "T10": +0.015, "T11": +0.015,
        })
        sector_ret = daily.groupby(daily.index.map(sector_map)).mean()
        composite = pd.Series({t: 50.0 for t in daily.index})

        # With only 3 sectors, the lowest pctile_rank lands at 33.33; widen
        # the threshold slightly so the test exercises the laggard branch.
        df = rank_rotation_candidates(
            daily, sector_ret, composite, sector_map, name_map,
            leader_pctile=66.0, laggard_pctile=40.0,
        )
        assert "T04" in df["ticker"].values
        row = df[df["ticker"] == "T04"].iloc[0]
        assert row["signal"] == "Leader in Laggard"
        assert row["sector"] == "Energy"

    def test_no_signal_for_neutral_sector(self, sector_map, name_map):
        # All sectors close to flat — none should be classified leader or laggard.
        daily = pd.Series({t: 0.001 for t in sector_map})
        daily["T00"] = -0.05  # individual outlier, but its sector isn't a leader
        sector_ret = daily.groupby(daily.index.map(sector_map)).mean()
        composite = pd.Series({t: 50.0 for t in daily.index})

        df = rank_rotation_candidates(
            daily, sector_ret, composite, sector_map, name_map,
            leader_pctile=95.0, laggard_pctile=5.0,
        )
        # With strict thresholds and uniform sector returns, nothing should qualify.
        assert df.empty

    def test_top_n_caps_each_signal(self, sector_map, name_map, rng):
        # Many laggers in a leading Tech sector — output should be capped at top_n.
        daily = pd.Series({f"T{i:02d}": +0.03 for i in range(12)})
        for t in ["T00", "T01", "T02", "T03"]:
            daily[t] = -0.01  # laggers in Tech
        sector_ret = daily.groupby(daily.index.map(sector_map)).mean()
        composite = pd.Series({t: float(rng.uniform(0, 100)) for t in daily.index})

        df = rank_rotation_candidates(
            daily, sector_ret, composite, sector_map, name_map,
            leader_pctile=66.0, laggard_pctile=33.0, top_n=2,
        )
        laggers = df[df["signal"] == "Lagger in Leader"]
        assert len(laggers) <= 2

    def test_unknown_ticker_skipped(self, sector_map, name_map):
        # Daily returns include a ticker not in sector_map — it must be ignored.
        daily = pd.Series({
            "T00": -0.02, "T01": +0.03, "T02": +0.03, "T03": +0.03,
            "T04": -0.01, "T05": -0.01, "T06": -0.01, "T07": -0.01,
            "T08": -0.01, "T09": -0.01, "T10": -0.01, "T11": -0.01,
            "MYSTERY": 0.10,
        })
        sector_ret = daily.iloc[:-1].groupby(
            daily.iloc[:-1].index.map(sector_map)
        ).mean()
        composite = pd.Series({t: 50.0 for t in daily.index})

        df = rank_rotation_candidates(
            daily, sector_ret, composite, sector_map, name_map,
            leader_pctile=66.0, laggard_pctile=33.0,
        )
        assert "MYSTERY" not in df["ticker"].values

    def test_empty_inputs_return_empty_df(self, sector_map, name_map):
        df = rank_rotation_candidates(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), sector_map, name_map,
        )
        assert df.empty
        assert "signal" in df.columns


class TestDeriveInputsFromSnapshot:
    def test_shapes(self, sector_map):
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        prices = pd.DataFrame(
            np.random.RandomState(0).randn(10, 12).cumsum(axis=0) + 100,
            index=dates,
            columns=[f"T{i:02d}" for i in range(12)],
        )
        daily, sector_ret = derive_inputs_from_snapshot(prices, sector_map)
        assert len(daily) == 12
        assert set(sector_ret.index) == {"Tech", "Energy", "Financials"}

    def test_handles_missing_sector_map(self):
        prices = pd.DataFrame(
            {"AAPL": [100, 101, 102]},
            index=pd.bdate_range("2024-01-01", periods=3, freq="B"),
        )
        daily, sector_ret = derive_inputs_from_snapshot(prices, {})
        assert daily.empty
        assert sector_ret.empty
