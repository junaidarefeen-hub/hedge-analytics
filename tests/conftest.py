"""Shared fixtures for hedge-analytics tests.

Generates small, deterministic synthetic price & return DataFrames so tests
are fast, reproducible, and never hit the network.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so `from config import ...` works.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture()
def rng():
    """Seeded random generator for reproducible synthetic data."""
    return np.random.default_rng(42)


@pytest.fixture()
def prices(rng):
    """Synthetic daily prices for 5 tickers over 300 trading days.

    Tickers: AAPL, MSFT (stocks) + SPY, QQQ (factors/benchmarks) + XLE (hedge).
    All generated as geometric Brownian motion around different drifts / vols so
    correlations are non-trivial but deterministic.
    """
    n_days = 300
    dates = pd.bdate_range("2022-01-03", periods=n_days, freq="B")
    tickers = ["AAPL", "MSFT", "SPY", "QQQ", "XLE"]
    # Common market factor drives positive correlation among most tickers.
    market = rng.normal(0.0003, 0.01, n_days)
    data = {}
    params = {
        "AAPL": (0.0005, 0.015, 0.7),
        "MSFT": (0.0004, 0.013, 0.6),
        "SPY":  (0.0003, 0.009, 0.9),
        "QQQ":  (0.0004, 0.011, 0.85),
        "XLE":  (0.0001, 0.018, 0.4),
    }
    for tk, (drift, vol, beta) in params.items():
        idio = rng.normal(0, vol, n_days)
        log_ret = drift + beta * market + idio
        cum = np.exp(np.cumsum(log_ret))
        data[tk] = cum * 100  # start near $100
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def returns(prices):
    """Simple (arithmetic) returns from the synthetic prices fixture."""
    return prices.pct_change().dropna(how="all")


@pytest.fixture()
def simple_returns(prices):
    """Simple (arithmetic) returns from the synthetic prices fixture."""
    return prices.pct_change().dropna(how="all")
