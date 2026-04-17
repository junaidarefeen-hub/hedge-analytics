# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit app for portfolio hedge analysis and S&P 500 market monitoring: correlations, betas, per-ticker factor exposure, hedge optimization, backtesting, drawdowns, regime detection, market-wide sector/factor/reversion screening, persistent signal-change tracking, and universe-wide drawdown/regime/pair screens.

## Run

```
cd ~/projects/hedge-analytics
python -m streamlit run app.py
pytest tests/ -v
```

Dependencies: `pip install -r requirements.txt` (streamlit, yfinance, plotly, scipy, pandas, numpy).

## Architecture

**4-layer structure**: `data/` (fetching) → `analytics/` (computation) → `ui/` (display) → `app.py` (wiring)

- `app.py` — Entry point, 21-tab layout. Tab groups: Analysis (1-6: Data · Price Performance · Correlation · Beta · Pairs/Spread · Factor Exposure), Hedging (7-12), Deep Dives (13-15), Market Monitor (16-21: Market Snapshot · Sector & Reversion · Factor Monitor · Trade Ideas · Watchlist & Changes · Universe Screener). Passes `returns`, `params`, and `factor_data` to Analysis/Hedging/Deep tabs; Market Monitor tabs load independently from Parquet cache. CSS dividers in `ui/style.py` mark group boundaries at tabs 7/13/16.
- `config.py` — All defaults and constants (tickers, dates, strategies, bounds, intervals, regime/rolling params)
- `ui/sidebar.py` — Returns `params` dict consumed by all tabs. Includes interval selector, cache clear button, and peer group load/save/delete.
- `ui/style.py` — `PLOTLY_LAYOUT` base config applied to every chart + CSS injection

## Data Flow

```
Sidebar → validate_and_fetch(interval=) [cached 1hr] → compute_returns()
  ├─ Correlation tab → heatmap + rolling chart + dendrogram (3+ tickers via scipy linkage)
  ├─ Beta tab → heatmap + rolling chart
  ├─ Optimizer tab → optimize_hedge() → HedgeResult → session_state; toggle for rolling optimization
  ├─ Compare tab → compare_strategies() → 4 strategies + backtests → CompareResult
  ├─ Custom Hedge tab → run_custom_hedge_analysis() → CustomHedgeResult; also embeds MC + Factor Analytics sections
  ├─ Backtest tab → run_backtest() → BacktestResult; toggle for dynamic rebalancing
  ├─ Monte Carlo tab → run_monte_carlo() → MonteCarloResult
  ├─ Stress Test tab → run_stress_test() → StressTestResult
  ├─ Drawdown tab → compute_drawdowns() → DrawdownAnalysis (standalone or hedged vs unhedged)
  ├─ Regime tab → detect_regimes() → RegimeResult + regime_hedge_effectiveness()
  └─ Factor Analytics tab → load_factor_data() + run_factor_analytics() → FactorAnalyticsResult

Market Monitor (independent data pipeline):
  Sidebar "Refresh Market Data" → yfinance live prices → merge into Parquet cache
  Refresh also calls build_snapshot() + append_snapshot() → signal_history.parquet
  Scheduled refresh (scripts/refresh_all.py) → RVX EQD(PRICE) → Parquet cache → git push
  Parquet cache → load_cached_prices() [cached 5min in Streamlit]
    ├─ Market Snapshot tab → compute_daily_snapshot() → breadth, sector heatmap, top movers
    │                       + current_dispersion() + rank_rotation_candidates() (Phase 1 augment)
    ├─ Sector & Reversion tab → compute_reversion_signals() → RSI, z-scores, Bollinger, composite
    ├─ Factor Monitor tab → compute_factor_monitor() → factor trends + stock betas
    ├─ Trade Ideas tab → screen_*_candidates() → ranked trade ideas + "+ Watchlist" capture
    ├─ Watchlist & Changes tab → load_history() + compute_deltas() + load_watchlist()
    │                            (signal transitions + persistent ideas + CSV/Bloomberg export)
    └─ Universe Screener tab → 3 modes: rank_underwater_basing, regime_conditional_stats,
                               scan_industry_pairs (gated behind "Run scan" + 1hr cache)

Analysis Factor Exposure tab (sidebar tickers, not Market Monitor):
  returns + factor_data → compute_per_ticker_exposures() → PerTickerExposures
    └─ heatmap (tickers x [market | selected GS factors]) + per-ticker drill-down
       (regression table + cumulative factor/idio decomposition)
```

**Session state bridge**: Optimizer stores `HedgeResult` + params hash in `st.session_state`. Backtest, Monte Carlo, Stress, Drawdown (hedged mode), and Regime (hedge effectiveness) read it. Staleness detection via `_params_hash()`.

**Multi-ticker long basket**: Optimizer and Compare tabs support selecting multiple long tickers via `st.multiselect` with per-ticker weight inputs. A synthetic basket column (`__LONG_BASKET__`) is injected into the returns DataFrame before any analytics call. All downstream tabs reconstruct the basket from `HedgeResult.target_tickers/target_weights`. Single-ticker case is exact passthrough (zero regression).

## Key Modules

### `analytics/optimization.py` — 4 strategies with max gross notional constraint
- Minimum Variance (SLSQP), Beta-Neutral (SLSQP + soft penalty fallback), CVaR (Rockafellar-Uryasev), Risk Parity (inverse-vol)
- `_multivariate_beta_matrix()`: single aligned sample OLS for beta additivity
- `_apply_min_names()`: caps weight to 1/N to force diversification
- **Max gross notional**: `optimize_hedge(max_gross_notional=)` controls total hedge budget
  - UI always passes the actual value (inequality mode); `None` only used in unit tests for legacy equality mode
  - Inequality constraint `|sum(w)| <= max_gross/notional`, optimizer freely chooses optimal hedge size
  - Per-instrument bounds auto-scale up when max_gross > notional (so full budget is accessible)
  - Post-optimization enforcement clips weights if beta-neutral equality constraints violate the gross cap
  - Risk Parity (formula-based): uses `min(1.0, max_hedge_ratio)` since there's no optimizer objective
  - `HedgeResult` stores `hedge_ratio` (actual) and `max_gross_notional` (cap) for downstream use
- **Weight display**: UI normalizes by `|sum(weights)|` so allocation percentages always sum to 100%; notional $ columns show actual deployed amounts
- All downstream analytics (backtest, MC, stress, drawdown, regime) are weight-sum agnostic — they use `target + hedges @ weights` directly
- **Basket metadata**: `HedgeResult.target_tickers` and `target_weights` store basket constituents; downstream tabs call `inject_basket_column()` to reconstruct the synthetic column

### `analytics/backtest.py` — Static + dynamic backtesting
- 11 metrics: Total/Ann. Return, Ann. Vol, Sharpe, Sortino, Max DD, Calmar, Omega, Max DD Duration, Tracking Error, Information Ratio
- Tracking Error/Information Ratio: hedged-only (0 for unhedged)
- `run_dynamic_backtest()`: periodic re-optimization via `_rebalance_dates()` (weekly/monthly/quarterly), produces 3-way comparison (unhedged/static/dynamic)

### `analytics/rolling_optimization.py` — Walk-forward optimization
- Slides window at `step` intervals, calls `optimize_hedge()` at each point
- Outputs weight evolution, vol history, turnover, weight stability

### `analytics/custom_hedge.py` — Custom multi-asset hedge analysis
- User-defined long/hedge portfolios with custom weights, notionals, and date range
- `run_custom_hedge_analysis()`: daily standalone/hedged returns, rolling vol/correlation, per-instrument net beta decomposition, hedge efficiency, constituent P&L contributions
- Reuses `_compute_metrics()`, `_tracking_error()`, `_information_ratio()` from `analytics/backtest.py`
- Independent of optimizer session state — fully self-contained tab
- **Net beta**: single-benchmark univariate `cov/var`, decomposed per hedge instrument with effective hedge ratios (`hedge_ratio × weight`). Benchmark sourced from `params["benchmarks"]` (sidebar indices), defaults to SPY. Falls back to hedge tickers if no benchmarks available.
- **Live net beta**: `compute_net_beta()` runs on every Streamlit rerun (no Analyze click needed) — metric cards for long beta, per-instrument contribution, and net beta update live as user changes notionals, tickers, or weights
- **Rolling net beta**: computed inside `run_custom_hedge_analysis()` using rolling `cov/var`; chart shows beta-neutral zero line and full-period reference line
- **P&L attribution**: uses two Plotly stackgroups (`"long"` / `"short"` based on `(L)` / `(S)` suffix) so negative short contributions display below zero
- **Embedded Monte Carlo**: creates synthetic long basket via `inject_basket_column`, constructs MC weights as `-hedge_ratio * hedge_weights` (negative = short), calls `run_monte_carlo()`. Reuses chart functions from `ui/montecarlo.py`. Session keys prefixed `cha_mc_`.
- **Embedded Factor Analytics**: passes `daily_standalone` / `daily_hedge_basket` / `daily_hedged` through `align_factor_returns()` into `run_factor_analytics()` for 3-leg OLS (Long/Short/Combined). Reuses chart functions from `ui/factor_analytics.py`. Requires `factor_data` param. Session keys prefixed `cha_fa_`. Gracefully hidden when factor data unavailable.
- **Weights**: auto-reset to equal weight when tickers change; normalize button uses deferred flag + `st.rerun()` to avoid Streamlit's "cannot modify session_state after widget instantiation" error

### `analytics/drawdown.py` — Drawdown period detection
- `compute_drawdowns()`: identifies contiguous underwater spans from cumulative series
- `DrawdownPeriod`: start (peak), trough, end (recovery or None), depth, duration, recovery days

### `analytics/regime.py` — Volatility-based regime detection
- Two methods: `quantile` (percentile thresholds) and `kmeans` (scipy kmeans2, sorted so regime 0 = lowest vol)
- `regime_hedge_effectiveness()`: per-regime hedged/unhedged vol, vol reduction, correlation

### `analytics/correlation.py` — Includes hierarchical clustering
- `correlation_clustering()`: distance = `1 - |corr|`, scipy linkage, rendered as plotly dendrogram in `ui/matrices.py`

### `utils/basket.py` — Multi-ticker basket support
- `BASKET_COLUMN_NAME = "__LONG_BASKET__"` — synthetic column name, cannot collide with real tickers
- `inject_basket_column(returns, tickers, weights)` → `(augmented_df, col_name)` — single ticker: passthrough; multi: weighted sum
- `exclude_basket_constituents(candidates, basket_tickers)` — remove basket members from hedge universe
- `basket_display_name(tickers, weights)` — human-readable label like "AAPL (50%) + MSFT (50%)"
- Used by Optimizer, Compare, and all downstream tabs that read `HedgeResult`

### `ui/weight_helpers.py` — Shared weight management UI helpers
- Extracted from `ui/custom_hedge.py` for reuse across Optimizer, Compare, and Custom Hedge tabs
- `equal_weight()`, `sync_weights()`, `handle_normalize()`, `render_weight_inputs()`, `weights_array()`
- Deferred normalize pattern: button sets flag → `st.rerun()` → pre-render applies normalization

### `analytics/factor_analytics.py` — Factor regression engine
- OLS via `numpy.linalg.lstsq` + `scipy.stats` for inference (no statsmodels dependency)
- Model: `y = α + β_market × Market + Σ βᵢ × Fᵢ + ε` for each active leg
- **Short basket optional**: long-only analysis when no short tickers selected; short + combined legs only populated when short basket provided
- `LegDecomposition` dataclass bundles per-leg OLS result, regression table, cumulative decomposition (both compounded and additive cumsum), vol stats, and daily factor/idio/total series
- Return decomposition: toggle between compounded `(1+r).cumprod()-1` and additive `r.cumsum()` (exact factor + idio = total)
- Volatility decomposition: rolling window vol (user-selectable: 20–252 days) for factor, idio, and total
- Long basket weights allow negative values (-100% to +100%) to represent short positions within the basket
- Beta heatmap rows match active legs (1 row long-only, 3 rows with short); significance stars from OLS p-values
- Multicollinearity warning via condition number check on design matrix
- **`compute_per_ticker_exposures()` + `PerTickerExposures`**: per-ticker OLS variant powering the standalone Factor Exposure tab. Reuses `_ols_regression` and `_build_leg`. Per-ticker re-aligns design matrix to each ticker's own non-null window (so a recent IPO doesn't truncate the universe). Skips tickers with fewer than k+5 obs. Returns shared-shape `betas`, `pvalues`, `significance` frames plus `r_squared`, `alpha_ann`, `n_obs` series and a `per_ticker_legs` dict of full `LegDecomposition` objects for drill-down. Default factor selection lives in `config.FA_DEFAULT_PER_TICKER_FACTORS` (excludes Equal Weight + Beta).

### `ui/data_tab_exposures.py` — Factor Exposure tab UI
- Standalone tab in the Analysis group (6th tab). Multiselect for GS factor regressors + selectbox for market regressor (sourced from sidebar benchmarks).
- Heatmap (rows = sidebar stocks, cols = market + selected factors) with significance stars baked into cells, mirrors the diverging colorscale + annotation pattern from `ui/factor_analytics.py::_beta_heatmap_chart`.
- Per-ticker expander → full OLS regression table + cumulative additive decomposition mini-chart (factor + idio = total).
- Right-column "Fit summary" table: R², annualized α, n_obs.

### `analytics/dispersion.py` — Cross-sectional + sector dispersion
- `compute_dispersion_history()` returns daily `cs_std` (per-name dispersion) + `sector_std` (equal-weight sector-mean dispersion) over a lookback window. Uses `groupby(daily_returns.columns.map(sector_map))` on the transposed returns frame for sector aggregation.
- `current_dispersion()` returns a `DispersionSnapshot` with current values, percentile rank in the lookback (`(series <= latest).mean() * 100`), and a `top_decile_flag` for UI emphasis.
- Used by the Market Snapshot tab "Dispersion Regime" section.

### `analytics/intraday_rotation.py` — Lagger-in-Leader / Leader-in-Laggard
- `rank_rotation_candidates()` consumes today's per-ticker daily returns + per-sector daily returns + reversion composite. Sectors are ranked by pctile; "leading" sectors are above `leader_pctile`, "lagging" below `laggard_pctile`. Single names down inside leading sectors → "Lagger in Leader"; up inside lagging sectors → "Leader in Laggard".
- `derive_inputs_from_snapshot()` is a UI-side helper that pulls `daily_returns` + `sector_returns_today` from a wide prices frame so the tab doesn't have to duplicate the `pct_change(fill_method=None).iloc[-1]` + `groupby(sector_map).mean()` boilerplate.
- Lagger sort: composite ascending (most-oversold first). Leader sort: `relative_to_sector` descending (biggest outperformer first).

### `analytics/signal_history.py` — Persistent snapshots + delta classifier + watchlist
- **On-disk format** (`data/cache/market_monitor/signal_history.parquet`, long format): columns `date, key, metric, value, label`. Numeric metrics (composite, rsi, ma_distance_50d, rolling_max/min_252d, last_price) populate `value` with `label=""`; categorical `factor_trend` rows populate `label` with `value=NaN`. One Parquet, one round-trip per write.
- `build_snapshot(prices, factor_returns=None) -> SignalSnapshot` — pure compute. Reuses `compute_reversion_signals` + `classify_factor_trends`.
- `append_snapshot()` — idempotent: drops all rows for the snapshot date before appending, so a refresh that fires twice in one day produces the same on-disk state. Called from `app.py::_run_mm_refresh` after every market data refresh.
- `compute_deltas(history, current_date, prior_date)` — categorizer producing `SignalDelta` rows. Categories: Entered/Exited Oversold (composite crosses 25), RSI Flip <30 / >70, Broke 50d MA / Reclaimed 50d MA (sign flip on `ma_distance_50d`), New 52w High/Low (today's `last_price` vs. *prior-day* `rolling_max_252d` × 0.999 — comparing to prior day avoids a tautology), Factor Trend Flip (label change on `factor_trend`).
- Watchlist CRUD (`load_watchlist`, `save_watchlist`, `add_ticker`, `remove_ticker`) backed by `watchlist.json`. `add_ticker` is idempotent: re-adding a ticker updates note/tags in place rather than duplicating.
- All persistence functions take an optional `store_dir` so tests can write to `tmp_path` without polluting the real cache.

### `ui/watchlist_tab.py` — "Watchlist & Changes" tab
- Three sections: (1) "What Changed Since…" with date picker + category multiselect → grouped pills; each pill has a "+ watchlist" button. (2) Watchlist table with live composite/RSI overlay merged from today's reversion signals. (3) Export — CSV download + Bloomberg-format ticker list (`AAPL US Equity ...`).
- Trade Ideas tab (`ui/trade_ideas_tab.py`) gained a `_watchlist_capture()` widget rendered after each screener mode's results table.

### `analytics/universe_drawdown.py` — Universe-wide deep-drawdown screen
- `rank_underwater_basing(prices, signals_df, min_dd, min_days_underwater, composite_lookback_days)` calls `compute_drawdowns` per ticker, picks the latest *unrecovered* drawdown period (filters `p.end is None`, takes the one closest to today), and computes a **`composite_60d_improvement`** delta — the novel signal: how much *less* oversold the stock has gotten over 60 days even while still in the drawdown. Sorted by improvement descending so most-improving basers surface first.
- ~16s for 504 tickers; gated behind `@st.cache_data(ttl=3600)` keyed on `prices.index.max()`.

### `analytics/universe_regime.py` — Per-ticker regime-conditional stats
- `regime_conditional_stats(prices, spx_returns, regime_series=None)` detects regimes once on SPX (defaults to `analytics.regime.detect_regimes(method='quantile', n_regimes=3)`), then per-ticker computes `avg_return_ann`, `vol_ann`, `sharpe`, `vs_spx_alpha`, `days` for each regime label. Returns long-format DataFrame.
- `latest_regime_label()` is a standalone helper for headers — runs regime detection on SPX and returns today's label string.

### `analytics/universe_pairs.py` — Within-industry pair scanner
- `scan_industry_pairs(prices, returns, industry_map, top_k_per_industry=3)` — two-stage cost bounding: (1) cheap `corr().abs()` per industry to pick top-K most-correlated unique pairs (caps total candidates at ~127 industries × K), (2) expensive `compute_pairs_analysis` (ADF + half-life + z-score) only on those survivors. Returns `PairCandidateRow` rows sorted by `|current_zscore|` descending.
- Industry mapping comes from `data.market_monitor.constituents.get_industry_map()` (added alongside the existing `get_sector_map`/`get_name_map`).

### `ui/universe_screen_tab.py` — "Universe Screener" tab
- Three modes via `st.radio`:
  - **Deep Drawdown Basing** — scatter (days underwater × current dd, colored by composite_60d_improvement) + ranked table + "+ Watchlist" capture.
  - **Regime-Conditional** — picker for vol regime (defaults to today's), top-50 by Sharpe in chosen regime, "+ Watchlist" capture.
  - **Industry Pair Spreads** — gated behind a "Run scan" button (~30-60s); cached result stored in `st.session_state["us_pairs_result"]`. Filterable by minimum |z-score|.
- Caching: each mode wraps its analytics call in `@st.cache_data(ttl=3600)` keyed on a `_price_hash(prices)` string (price index max + column count) so the cache invalidates on every market data refresh.

### `data/ecm_client.py` — ECM MCP client for RVX / Prismatic
- OAuth2 via Keycloak (60s JWT TTL, auto-refresh). Credentials stored in `~/.claude/.credentials.json` → `mcpOAuth["ecm|72c3fb7b2b25e5da"]`.
- `is_available()`: checks if credentials exist (no network call). Returns False on Render.
- `fetch_rvx_series(ticker, attribute, start_date, end_date)`: fetch timeseries data from RVX
- `render_gadget_csv(gadget_id)` / `read_share_file(path)`: fetch Prismatic gadget data as CSV
- SSE transport over HTTPS to `mcp.elementcapital.corp/mcp`, JSON-RPC 2.0 protocol
- Thread-safe token management via `_TOKEN_LOCK`
- Used by: `data/factor_loader.py` (Prismatic gadgets), `data/market_monitor/rvx_fetcher.py` (RVX EQD), `scripts/refresh_all.py`

### `data/factor_loader.py` — GS factor data (Prismatic → Parquet cache)
- **Source priority**: Prismatic gadgets (layout 10052, gadgets 10685–10693) → local Parquet cache (`data/cache/factor_prices.parquet`). No Excel fallback.
- Prismatic returns additive cumulative series starting at 0; daily returns = `diff()`.
- `load_factor_data()`: cached 24hr (`@st.cache_data(ttl=86400)`), returns `FactorData` (prices, returns, ticker/name maps)
- `get_factor_staleness_days()`: returns calendar days since last data point; app shows warning when >2 days stale
- `align_factor_returns()`: inner join on normalized DatetimeIndex, dropna for clean regression input
- `clear_factor_cache()` (sidebar "Reload factor data" button): runs a parallel full re-fetch + overwrites the Parquet when ECM is reachable; on Render (no ECM) it only clears Streamlit's in-memory cache. Always full-history — see "Prismatic incremental gotcha" below.
- `_fetch_from_prismatic()`: fetches all 9 gadgets in parallel via `ThreadPoolExecutor(max_workers=9)`. ~10-11s steady-state vs ~26s sequential (~2.5×). Accepts an optional `start_date` for date-filtered renders, but **see the gotcha** before using it.
- **Prismatic incremental gotcha**: passing `plotterStartDate` does NOT just trim rows — Prismatic recomputes the factor weights/leverage for the new start date, producing a recomputed series at a **different scale**. Empirically, cumulative deltas across the same date range come out ~0.44× the long-history version. This means date-filtered fetches are **not arithmetically compatible** with the long-history cache; you cannot rebase or merge them in any sane way. True incremental refresh is therefore not viable; the parallel full re-fetch is the right pattern.
- **Render deployment**: Prismatic MCP is unavailable on Render. The Parquet cache file is committed to git (`.gitignore` exception) so Render always has factor data. On deploy, `is_available()` returns False → loader uses the committed Parquet cache. Daily automated refresh keeps the cache current (see `scripts/refresh_all.py`).

### `data/ecm_client.py::render_gadget_csv()` — accepts optional `parameters`
- Signature: `render_gadget_csv(gadget_id, parameters=None) -> str` (returns the server file path).
- `parameters` is forwarded to the underlying `ecm_prismatic_render_gadgets` MCP tool. Discover valid keys via `ecm_prismatic_get_form_fields` — common ones for plotter-style gadgets are `plotterStartDate` / `plotterEndDate` (`M/D/YY` format).

### `data/fetcher.py` — yfinance with interval support
- `validate_interval_date_range()`: enforces yfinance limits (1m→7d, 5m/15m→60d, 1h→730d)
- `clear_cache()`: clears both price and name caches

### `data/market_monitor/` — S&P 500 market data pipeline (RVX)
- **`constituents.py`**: Loads `sp500_constituents.json` (503 tickers with GICS sector/industry). Provides `get_sector_map()`, `get_name_map()`, `get_all_tickers()`, `GICS_SECTORS`.
- **`rvx_fetcher.py`**: Batch price fetcher using `ThreadPoolExecutor(max_workers=15)` calling `ecm_client.fetch_rvx_series()`. Ticker format: `EQSN {SYMBOL}` with `EQD(PRICE)`. Multi-class tickers use slash (BRK-B → BRK/B). Supports `intraday=True` for `hist_live` mode (current live price). Returns DataFrame + failed list.
- **`cache_manager.py`**: Parquet cache at `data/cache/market_monitor/prices.parquet` + `metadata.json`. Supports incremental refresh via `get_refresh_plan()` + `merge_incremental()` (combine_first pattern from ne73-dashboard). `is_stale()` checks metadata timestamp.
- **Three-tier refresh**:
  1. **Bulk historical** (`python scripts/refresh_market_data.py --full`): One-time 10yr backfill from RVX. ~60s for 504 tickers.
  2. **EOD close** (`python scripts/refresh_market_data.py --eod`): Daily incremental from RVX, designed for scheduled trigger at 4:30 PM ET. Only fetches new settled closes. ~30s.
  3. **Intraday live** (sidebar "Refresh Market Data" button): Uses yfinance for live prices during market hours. ~25s for 504 tickers. RVX `EQD(PRICE)` only provides settled daily closes (no intraday), so yfinance is used for live quotes.
- **Data flow**: App loads from Parquet on startup (`@st.cache_data(ttl=300)`). Sidebar button triggers yfinance live refresh. EOD script runs standalone via RVX.
- **Hybrid data sources**: Historical data (10yr) from RVX `EQD(PRICE)` for accuracy/consistency. Live intraday from yfinance for real-time monitoring. SPX is mapped to `^GSPC` for yfinance.
- **Independent from hedge tabs**: Market monitor tabs do not use sidebar ticker/date inputs. They load exclusively from the Parquet cache.
- **Cache size**: ~7 MB for 504 tickers × 2515 days (10yr). Parquet is efficient for columnar float data.
- **Render deployment**: Both `prices.parquet` and `metadata.json` are committed to git (`.gitignore` exceptions). RVX is unavailable on Render; sidebar yfinance refresh provides intraday updates. Historical data stays current via daily scheduled refresh + commit/push.
- **`.gitignore` cache exceptions**: `data/cache/*` is ignored except: `!data/cache/factor_prices.parquet`, `!data/cache/market_monitor/`, `!data/cache/market_monitor/prices.parquet`, `!data/cache/market_monitor/metadata.json`, `!data/cache/market_monitor/signal_history.parquet`, `!data/cache/market_monitor/watchlist.json`.

### `scripts/refresh_all.py` — Unified daily cache refresh
- Refreshes both factor data (Prismatic) and market monitor data (RVX EOD) in one invocation
- `python scripts/refresh_all.py` — refresh both caches
- `python scripts/refresh_all.py --commit` — refresh + git commit & push updated Parquet files
- `python scripts/refresh_all.py --factors-only` / `--market-only` — selective refresh
- Requires ECM credentials (local only, not available on Render)
- **Scheduled**: Windows Task Scheduler (`HedgeAnalytics-DailyRefresh`), weekdays at 4:45 PM ET with `--commit`
- **Render deployment**: Automated commit + push keeps Render's committed Parquet caches current
- Log output appended to `scripts/refresh_all.log`

### `analytics/reversion.py` — Oversold/overbought screening
- Signals: RSI(14), Z-score(20d/60d), MA distance(50d/200d), Bollinger %B
- Composite score: percentile rank each signal across universe, then weighted average (RSI 30%, z-score_20d 25%, z-score_60d 15%, MA_50d 15%, Bollinger 15%). Lower score = more oversold.
- `compute_reversion_signals(prices)` → `ReversionSignals` with per-ticker signals and composite.

### `analytics/factor_monitor.py` — Factor performance + stock-level betas
- Reuses GS factor data from `data/factor_loader.py`
- `compute_factor_summary()`: multi-period cumulative returns per factor
- `classify_factor_trends()`: 20d MA vs 60d MA → Trending Up / Neutral / Mean Reverting
- `estimate_stock_factor_betas()`: OLS (numpy.linalg.lstsq) per stock against market + factors

### `analytics/trade_screener.py` — Trade candidate scoring
- Three modes: `screen_reversion_candidates()`, `screen_factor_candidates()`, `screen_factor_reversion()` (combo)
- Combo mode: intersection of oversold stocks + high factor exposure to trending factors

## Key Design Decisions

- Weights displayed as allocation percentages (normalized to 100% of hedge basket) with Side column (Short/Long); zero-weight filtered out
- Correlation = target vs weighted hedge basket (using `abs(weights)`)
- Backtest metrics: returns/vol/drawdown/TE as percentages; ratios (Sharpe/Sortino/Calmar/Omega/IR) as decimals; DD duration as integer days
- All charts use `PLOTLY_LAYOUT` from `ui/style.py` as base
- Theme: blue primary (#2563eb), configured in `.streamlit/config.toml`
- Rolling optimization and dynamic rebalancing use radio toggles (static vs dynamic) within their respective tabs
- Late import of `optimize_hedge` in `run_dynamic_backtest()` to avoid circular dependency (backtest ↔ optimization)
- `max_gross_notional` propagates through: `optimize_hedge` → `compare_strategies` / `rolling_optimize` / `run_dynamic_backtest`; UI always passes the actual max_gross value (inequality mode) so the optimizer can freely choose the optimal hedge size up to the cap
- Correlation/beta matrices use full-period calculations; rolling window only affects rolling line charts
- Peer group loading uses a two-phase rerun: phase 1 stages pending tickers + selectbox reset, phase 2 (before widget render) applies them. This avoids Streamlit's "cannot modify session_state after widget instantiation" error and prevents infinite rerun loops.
- Drawdown (hedged mode) and Regime tabs display strategy, target, and analysis period for context
- Multi-ticker long basket: Optimizer uses `opt_target_tickers` (multiselect) instead of `opt_target` (selectbox) to avoid Streamlit session key type mismatch. Compare uses `cmp_target_tickers`. Stress test custom shocks show per-constituent inputs, with basket shock computed as weighted sum before passing to analytics.
