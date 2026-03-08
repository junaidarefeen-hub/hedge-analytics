# Hedge Analytics

A Streamlit app for portfolio hedge analysis — correlation matrices, beta estimation, multi-strategy hedge optimization, and backtesting.

## Features

**5 interactive tabs:**

- **Data** — View stock and factor/index prices fetched from Yahoo Finance
- **Correlation** — Full-period correlation heatmap + rolling pairwise correlation chart
- **Beta** — Beta matrix (tickers × benchmarks) heatmap + rolling beta chart
- **Hedge Optimizer** — Find optimal hedge portfolios using one of 4 strategies
- **Backtest** — Compare hedged vs unhedged portfolio performance over a historical period

### Hedge Optimization Strategies

| Strategy | Description |
|----------|-------------|
| **Minimum Variance** | Minimizes portfolio variance of the combined long + hedge position |
| **Beta-Neutral** | Minimum variance with constraints to zero out beta exposure to selected factors |
| **Tail Risk (CVaR)** | Minimizes Conditional Value at Risk using Rockafellar-Uryasev linearization |
| **Risk Parity** | Allocates hedge weights inversely proportional to each instrument's volatility |

### Optimizer Controls

- **Hedge universe filtering** — restrict hedges to stocks only, factors/indices only, or all tickers
- **Min hedge names** — force diversification by requiring a minimum number of instruments in the basket
- **Configurable weight bounds** — default short-only (`[-1, 0]`), supports long-only or mixed
- **Notional sizing** — translate percentage weights into dollar notionals

### Backtest Metrics

- Total Return, Annualized Return, Annualized Volatility
- Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio
- Cumulative return chart and rolling volatility chart (hedged vs unhedged)

## Getting Started

### Prerequisites

- Python 3.10+

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## How It Works

1. **Configure** tickers and date range in the sidebar (stocks and factors/indices separately)
2. **Explore** correlations and betas across your universe
3. **Optimize** — pick a target long position, choose a strategy, and click Optimize to get hedge weights
4. **Backtest** — the hedge weights carry over to the Backtest tab where you can evaluate historical performance

## Tech Stack

- **Streamlit** — UI framework
- **yfinance** — Market data
- **pandas / numpy** — Data manipulation
- **scipy** — Constrained optimization (SLSQP)
- **plotly** — Interactive charts
