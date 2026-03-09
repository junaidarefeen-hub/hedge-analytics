# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit app for portfolio hedge analysis: correlations, betas, hedge optimization, backtesting, drawdowns, and regime detection.

## Run

```
cd ~/projects/hedge-analytics
python -m streamlit run app.py
pytest tests/ -v
```

Dependencies: `pip install -r requirements.txt` (streamlit, yfinance, plotly, scipy, pandas, numpy).

## Architecture

**4-layer structure**: `data/` (fetching) → `analytics/` (computation) → `ui/` (display) → `app.py` (wiring)

- `app.py` — Entry point, 10-tab layout (Data, Correlation, Beta, Hedge Optimizer, Strategy Compare, Backtest, Monte Carlo, Stress Test, Drawdown, Regime)
- `config.py` — All defaults and constants (tickers, dates, strategies, bounds, intervals, regime/rolling params)
- `ui/sidebar.py` — Returns `params` dict consumed by all tabs. Includes interval selector and cache clear button.
- `ui/style.py` — `PLOTLY_LAYOUT` base config applied to every chart + CSS injection

## Data Flow

```
Sidebar → validate_and_fetch(interval=) [cached 1hr] → compute_returns()
  ├─ Correlation tab → heatmap + rolling chart + dendrogram (3+ tickers via scipy linkage)
  ├─ Beta tab → heatmap + rolling chart
  ├─ Optimizer tab → optimize_hedge() → HedgeResult → session_state; toggle for rolling optimization
  ├─ Compare tab → compare_strategies() → 4 strategies + backtests → CompareResult
  ├─ Backtest tab → run_backtest() → BacktestResult; toggle for dynamic rebalancing
  ├─ Monte Carlo tab → run_monte_carlo() → MonteCarloResult
  ├─ Stress Test tab → run_stress_test() → StressTestResult
  ├─ Drawdown tab → compute_drawdowns() → DrawdownAnalysis (standalone or hedged vs unhedged)
  └─ Regime tab → detect_regimes() → RegimeResult + regime_hedge_effectiveness()
```

**Session state bridge**: Optimizer stores `HedgeResult` + params hash in `st.session_state`. Backtest, Monte Carlo, Stress, Drawdown (hedged mode), and Regime (hedge effectiveness) read it. Staleness detection via `_params_hash()`.

## Key Modules

### `analytics/optimization.py` — 4 strategies with max gross notional constraint
- Minimum Variance (SLSQP), Beta-Neutral (SLSQP + soft penalty fallback), CVaR (Rockafellar-Uryasev), Risk Parity (inverse-vol)
- `_multivariate_beta_matrix()`: single aligned sample OLS for beta additivity
- `_apply_min_names()`: caps weight to 1/N to force diversification
- **Max gross notional**: `optimize_hedge(max_gross_notional=)` controls total hedge budget
  - `None` (default): equality constraint `sum(w) = -1`, backward compatible
  - When set: inequality constraint `|sum(w)| <= max_gross/notional`, optimizer freely chooses optimal hedge size
  - Per-instrument bounds auto-scale up when max_gross > notional (so full budget is accessible)
  - Post-optimization enforcement clips weights if beta-neutral equality constraints violate the gross cap
  - Risk Parity (formula-based): uses `min(1.0, max_hedge_ratio)` since there's no optimizer objective
  - `HedgeResult` stores `hedge_ratio` (actual) and `max_gross_notional` (cap) for downstream use
- **Weight display**: UI normalizes by `|sum(weights)|` so allocation percentages always sum to 100%; notional $ columns show actual deployed amounts
- All downstream analytics (backtest, MC, stress, drawdown, regime) are weight-sum agnostic — they use `target + hedges @ weights` directly

### `analytics/backtest.py` — Static + dynamic backtesting
- 11 metrics: Total/Ann. Return, Ann. Vol, Sharpe, Sortino, Max DD, Calmar, Omega, Max DD Duration, Tracking Error, Information Ratio
- Tracking Error/Information Ratio: hedged-only (0 for unhedged)
- `run_dynamic_backtest()`: periodic re-optimization via `_rebalance_dates()` (weekly/monthly/quarterly), produces 3-way comparison (unhedged/static/dynamic)

### `analytics/rolling_optimization.py` — Walk-forward optimization
- Slides window at `step` intervals, calls `optimize_hedge()` at each point
- Outputs weight evolution, vol history, turnover, weight stability

### `analytics/drawdown.py` — Drawdown period detection
- `compute_drawdowns()`: identifies contiguous underwater spans from cumulative series
- `DrawdownPeriod`: start (peak), trough, end (recovery or None), depth, duration, recovery days

### `analytics/regime.py` — Volatility-based regime detection
- Two methods: `quantile` (percentile thresholds) and `kmeans` (scipy kmeans2, sorted so regime 0 = lowest vol)
- `regime_hedge_effectiveness()`: per-regime hedged/unhedged vol, vol reduction, correlation

### `analytics/correlation.py` — Includes hierarchical clustering
- `correlation_clustering()`: distance = `1 - |corr|`, scipy linkage, rendered as plotly dendrogram in `ui/matrices.py`

### `data/fetcher.py` — yfinance with interval support
- `validate_interval_date_range()`: enforces yfinance limits (1m→7d, 5m/15m→60d, 1h→730d)
- `clear_cache()`: clears both price and name caches

## Key Design Decisions

- Weights displayed as allocation percentages (normalized to 100% of hedge basket) with Side column (Short/Long); zero-weight filtered out
- Correlation = target vs weighted hedge basket (using `abs(weights)`)
- Backtest metrics: returns/vol/drawdown/TE as percentages; ratios (Sharpe/Sortino/Calmar/Omega/IR) as decimals; DD duration as integer days
- All charts use `PLOTLY_LAYOUT` from `ui/style.py` as base
- Theme: blue primary (#2563eb), configured in `.streamlit/config.toml`
- Rolling optimization and dynamic rebalancing use radio toggles (static vs dynamic) within their respective tabs
- Late import of `optimize_hedge` in `run_dynamic_backtest()` to avoid circular dependency (backtest ↔ optimization)
- `max_gross_notional` propagates through: `optimize_hedge` → `compare_strategies` / `rolling_optimize` / `run_dynamic_backtest`; UI always passes the actual max_gross value (inequality mode) so the optimizer can freely choose the optimal hedge size up to the cap
