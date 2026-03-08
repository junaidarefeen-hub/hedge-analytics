# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit app for portfolio hedge analysis: correlations, betas, hedge optimization, and backtesting.

## Run

```
cd ~/projects/hedge-analytics
python -m streamlit run app.py
```

No tests or linting configured. Dependencies: `pip install -r requirements.txt` (streamlit, yfinance, plotly, scipy, pandas, numpy).

## Architecture

**4-layer structure**: `data/` (fetching) → `analytics/` (computation) → `ui/` (display) → `app.py` (wiring)

- `app.py` — Entry point, 6-tab layout (Data, Correlation, Beta, Hedge Optimizer, Strategy Compare, Backtest)
- `config.py` — All defaults and constants (tickers, dates, strategy options, bounds, annualization)
- `ui/sidebar.py` — Returns a `params` dict consumed by all tabs. Factors double as beta benchmarks.
- `ui/style.py` — `PLOTLY_LAYOUT` base config applied to every chart + CSS injection

## Data Flow

```
Sidebar → validate_and_fetch() [cached 1hr] → compute_returns()
  ├─ Correlation/Beta tabs → matrices + rolling charts
  ├─ Optimizer tab → optimize_hedge() → HedgeResult → session_state
  ├─ Compare tab → compare_strategies() → runs all 4 strategies + backtests → CompareResult
  └─ Backtest tab → reads HedgeResult from session_state → run_backtest() → BacktestResult
```

**Session state bridge**: The Optimizer (and Compare tab's "Use strategy" buttons) stores `HedgeResult` and a params hash in `st.session_state`. The Backtest tab reads it. Staleness detection compares the stored hash against current sidebar params.

## Strategy Compare (`analytics/compare.py`)

Runs all 4 strategies for a given target, backtests each, and ranks by composite score across: Vol Reduction, Total Return, Ann. Return, Ann. Volatility, Sharpe, Sortino, Max Drawdown, Calmar. Per-metric rank (1=best) averaged into a composite rank. "Use strategy" buttons load the selected HedgeResult/BacktestResult into session_state for the Optimizer/Backtest tabs.

## Hedge Optimizer (`analytics/optimization.py`)

4 strategies, all constrained so weights sum to -1 (short-only) or +1 (long-only):

| Strategy | Method | Key detail |
|----------|--------|------------|
| Minimum Variance | scipy SLSQP | `Var(r_target + w @ r_hedges)` objective |
| Beta-Neutral | scipy SLSQP | Min variance + per-factor `beta=0` equality constraints. Falls back to soft penalty (1e6) if infeasible, keeping sum as hard constraint |
| Tail Risk (CVaR) | scipy SLSQP | Rockafellar-Uryasev linearization with VaR auxiliary variable |
| Risk Parity | Closed-form | Inverse-volatility weighting, no optimizer |

**Multivariate betas**: `_multivariate_beta_matrix()` regresses all tickers on all factors simultaneously using a single `dropna`-aligned sample. This ensures beta additivity and handles multicollinearity. Used both in Beta-Neutral constraints and in the results display.

**Min hedge names**: `_apply_min_names()` caps max weight per instrument to `1/N`, forcing at least N instruments to be active.

**Over-determination guard**: Beta constraints limited to `n_hedges - 1` to avoid infeasibility from too many equality constraints.

## Key Design Decisions

- Weights displayed as absolute values with a Side column (Short/Long); zero-weight instruments filtered out
- Correlation metric = target vs weighted hedge basket (using `abs(weights)`), not hedged vs unhedged portfolio
- Betas shown only against factor/index tickers, not other stocks
- Backtest metrics: returns/vol/drawdown formatted as percentages, ratios (Sharpe/Sortino/Calmar) as decimals
- All charts use `PLOTLY_LAYOUT` from `ui/style.py` as base, then override per-chart settings
- Theme: blue primary (#2563eb), configured in `.streamlit/config.toml`
