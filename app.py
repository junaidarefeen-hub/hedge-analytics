import streamlit as st

st.set_page_config(page_title="Hedge Analytics", layout="wide")

import pandas as pd

from analytics.beta import beta_matrix, rolling_beta
from analytics.correlation import correlation_clustering, correlation_matrix, rolling_correlation
from analytics.returns import compute_returns
from data.factor_loader import load_factor_data
from data.fetcher import fetch_ticker_names, validate_and_fetch
from ui.drawdown import render_drawdown_tab
from ui.factor_monitor_tab import render_factor_monitor_tab
from ui.market_snapshot_tab import render_market_snapshot_tab
from ui.matrices import beta_heatmap, correlation_dendrogram, correlation_heatmap
from ui.regime import render_regime_tab
from ui.sector_reversion_tab import render_sector_reversion_tab
from ui.sidebar import render_sidebar
from ui.style import inject_css
from ui.trade_ideas_tab import render_trade_ideas_tab
from ui.universe_screen_tab import render_universe_screen_tab
from ui.watchlist_tab import render_watchlist_tab
from ui.backtest import render_backtest_tab
from ui.compare import render_compare_tab
from ui.custom_hedge import render_custom_hedge_tab
from ui.data_tab_exposures import render_per_ticker_exposures
from ui.factor_analytics import render_factor_analytics_tab
from ui.montecarlo import render_montecarlo_tab
from ui.optimizer import render_optimizer_tab
from ui.pairs import render_pairs_tab
from ui.performance import render_performance_tab
from ui.stress import render_stress_tab
from ui.timeseries import rolling_beta_chart, rolling_correlation_chart
from utils.validation import check_data_sufficiency

inject_css()

st.title("Hedge Analytics")

params = render_sidebar()
if params is None:
    st.stop()

# Fetch data for both groups
interval = params.get("interval", "1d")
stock_prices, stock_failed = validate_and_fetch(
    params["stock_tickers"], params["start_date"], params["end_date"], interval=interval,
)
factor_prices, factor_failed = validate_and_fetch(
    params["factor_tickers"], params["start_date"], params["end_date"], interval=interval,
)

all_failed = stock_failed + factor_failed
if all_failed:
    st.warning(f"Could not fetch data for: {', '.join(all_failed)}")

valid_stocks = [t for t in params["stock_tickers"] if t not in stock_failed]
valid_factors = [t for t in params["factor_tickers"] if t not in factor_failed]
valid_all = valid_stocks + valid_factors

if len(valid_all) < 2:
    st.error("Need at least 2 valid tickers with data. Please adjust your inputs.")
    st.stop()

# Merge price DataFrames
prices = pd.concat(
    [df for df in [stock_prices[valid_stocks] if valid_stocks else None,
                   factor_prices[valid_factors] if valid_factors else None]
     if df is not None],
    axis=1,
)

# Fetch names
stock_names = fetch_ticker_names(valid_stocks) if valid_stocks else {}
factor_names = fetch_ticker_names(valid_factors) if valid_factors else {}

# Compute returns
returns = compute_returns(prices, method=params["return_method"])

# Data sufficiency warning
sufficiency_warn = check_data_sufficiency(len(returns), params["window"])
if sufficiency_warn:
    st.warning(sufficiency_warn)

# Load GS factor data (Prismatic → Parquet cache)
try:
    factor_data = load_factor_data()
except Exception as e:
    st.warning(f"Could not load factor data — Factor Analytics tab will be unavailable. ({e})")
    factor_data = None

if factor_data is not None:
    from data.factor_loader import get_factor_staleness_days

    _staleness_days = get_factor_staleness_days(factor_data)
    if _staleness_days is not None and _staleness_days > 2:
        st.warning(
            f"Factor data is {_staleness_days} days stale "
            f"(last date: {factor_data.prices.index.max().strftime('%Y-%m-%d')}). "
            f"Reload via sidebar button."
        )

# Tabs
# Tab groups:
#   Analysis (1-6):       Data · Price Performance · Correlation · Beta · Pairs/Spread · Factor Exposure
#   Hedging  (7-12):      Hedge Optimizer · Strategy Compare · Backtest · Monte Carlo · Stress Test · Drawdown
#   Deep     (13-15):     Custom Hedge · Factor Analytics · Regime
#   Market Monitor (16-21): Market Snapshot · Sector & Reversion · Factor Monitor · Trade Ideas · Watchlist · Universe Screener
(tab_data, tab_perf, tab_corr, tab_beta, tab_pairs, tab_exposure,
 tab_optim, tab_compare, tab_backtest, tab_mc, tab_stress, tab_dd,
 tab_custom, tab_factor, tab_regime,
 tab_mm_snapshot, tab_mm_sector, tab_mm_factors, tab_mm_trades,
 tab_mm_watchlist, tab_mm_screener) = st.tabs([
    "Data", "Price Performance", "Correlation", "Beta", "Pairs/Spread", "Factor Exposure",
    "Hedge Optimizer", "Strategy Compare", "Backtest", "Monte Carlo", "Stress Test", "Drawdown",
    "Custom Hedge", "Factor Analytics", "Regime",
    "Market Snapshot", "Sector & Reversion", "Factor Monitor", "Trade Ideas",
    "Watchlist & Changes", "Universe Screener",
])

# --- Data Tab ---
with tab_data:
    st.caption("View the tickers loaded from the sidebar and their historical closing prices.")
    if valid_stocks:
        st.subheader("Stocks / ETFs")
        names_df = pd.DataFrame(
            [{"Ticker": t, "Name": stock_names.get(t, t)} for t in valid_stocks]
        )
        st.dataframe(names_df, use_container_width=True, hide_index=True)

        with st.expander("View closing prices", expanded=False):
            st.dataframe(
                stock_prices[valid_stocks].sort_index(ascending=False).style.format("{:.2f}"),
                use_container_width=True,
                height=350,
            )

    if valid_factors:
        st.subheader("Factors / Indices")
        fnames_df = pd.DataFrame(
            [{"Ticker": t, "Name": factor_names.get(t, t)} for t in valid_factors]
        )
        st.dataframe(fnames_df, use_container_width=True, hide_index=True)

        with st.expander("View closing prices", expanded=False):
            st.dataframe(
                factor_prices[valid_factors].sort_index(ascending=False).style.format("{:.2f}"),
                use_container_width=True,
                height=350,
            )

    if factor_data is not None:
        st.subheader("GS Factor Indices")
        factor_summary = pd.DataFrame([
            {
                "Factor": name,
                "Ticker": factor_data.ticker_map[name],
                "Start": factor_data.prices[name].first_valid_index().strftime("%Y-%m-%d"),
                "End": factor_data.prices[name].last_valid_index().strftime("%Y-%m-%d"),
                "Latest": f"{factor_data.prices[name].dropna().iloc[-1]:.2f}",
            }
            for name in factor_data.prices.columns
        ])
        st.dataframe(factor_summary, use_container_width=True, hide_index=True)
        with st.expander("View factor prices", expanded=False):
            st.dataframe(
                factor_data.prices.sort_index(ascending=False).style.format("{:.2f}"),
                use_container_width=True,
                height=350,
            )

# --- Price Performance Tab ---
with tab_perf:
    render_performance_tab(returns, params)

# --- Correlation Tab ---
with tab_corr:
    st.caption(
        "Explore how your tickers move together. The heatmap shows pairwise correlations across all tickers. "
        "The rolling chart tracks how the relationship between any two tickers changes over time. "
        "The dendrogram (3+ tickers) groups tickers by similarity."
    )
    col_left, col_right = st.columns(2)

    with col_left:
        corr = correlation_matrix(returns)
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    with col_right:
        all_t = params["all_tickers"]
        c1, c2 = st.columns(2)
        with c1:
            ta = st.selectbox("Ticker A", options=all_t, index=0, key="corr_ticker_a",
                              help="First ticker for the rolling correlation chart.")
        with c2:
            tb = st.selectbox("Ticker B", options=all_t, index=min(1, len(all_t) - 1), key="corr_ticker_b",
                              help="Second ticker for the rolling correlation chart.")
        if ta == tb:
            st.caption("Same ticker selected — correlation = 1.0")
        if ta in returns.columns and tb in returns.columns:
            rc = rolling_correlation(returns, ta, tb, params["window"])
            st.plotly_chart(
                rolling_correlation_chart(rc, ta, tb, params["window"]),
                use_container_width=True,
            )

    # Correlation clustering dendrogram (3+ tickers)
    if len(corr) >= 3:
        try:
            clustering = correlation_clustering(corr)
            st.plotly_chart(correlation_dendrogram(clustering), use_container_width=True)
        except Exception:
            pass

# --- Beta Tab ---
with tab_beta:
    st.caption(
        "Measure how sensitive each ticker is to market benchmarks. Beta > 1 means the ticker amplifies benchmark moves; "
        "beta < 1 means it's less volatile. The rolling chart shows how this sensitivity changes over time."
    )
    benchmarks = [b for b in params["benchmarks"] if b in returns.columns]
    if not benchmarks:
        st.info("Add at least one factor/index ticker to compute betas.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            bm = beta_matrix(returns, benchmarks)
            st.plotly_chart(beta_heatmap(bm), use_container_width=True)

        with col_right:
            all_t = params["all_tickers"]
            b1, b2 = st.columns(2)
            with b1:
                bb = st.selectbox("Benchmark", options=benchmarks, index=0, key="beta_benchmark",
                                  help="The market index or factor to measure beta against.")
            with b2:
                non_bench = [t for t in all_t if t != bb]
                bt = st.selectbox("Ticker", options=non_bench or all_t, index=0, key="beta_ticker",
                                  help="The ticker whose beta to the benchmark you want to track over time.")
            if bt in returns.columns and bb in returns.columns:
                rb = rolling_beta(returns, bt, bb, params["window"])
                st.plotly_chart(
                    rolling_beta_chart(rb, bt, bb, params["window"]),
                    use_container_width=True,
                )

# --- Pairs/Spread ---
with tab_pairs:
    render_pairs_tab(prices, returns, params)

# --- Factor Exposure (per-ticker drill-down) ---
with tab_exposure:
    render_per_ticker_exposures(
        valid_stocks=valid_stocks,
        returns=returns,
        factor_data=factor_data,
        benchmarks=params.get("benchmarks", []),
    )

# --- Hedge Optimizer ---
with tab_optim:
    render_optimizer_tab(returns, params)

# --- Strategy Compare ---
with tab_compare:
    render_compare_tab(returns, params)

# --- Custom Hedge ---
with tab_custom:
    render_custom_hedge_tab(returns, params, factor_data)

# --- Backtest ---
with tab_backtest:
    render_backtest_tab(returns, params)

# --- Monte Carlo ---
with tab_mc:
    render_montecarlo_tab(returns, params)

# --- Stress Test ---
with tab_stress:
    render_stress_tab(returns, params)

# --- Drawdown ---
with tab_dd:
    render_drawdown_tab(returns, params)

# --- Regime ---
with tab_regime:
    render_regime_tab(returns, params)

# --- Factor Analytics ---
with tab_factor:
    render_factor_analytics_tab(returns, params, factor_data)


# ---------------------------------------------------------------------------
# Market Monitor tabs — self-contained data loading from Parquet cache
# ---------------------------------------------------------------------------
from data.market_monitor.cache_manager import (
    get_refresh_plan,
    load_cached_prices,
    merge_incremental,
    save_prices,
)
from data.market_monitor.constituents import get_all_tickers


@st.cache_data(ttl=300, show_spinner="Loading market data...")
def _load_mm_prices():
    """Load market monitor prices from Parquet cache."""
    return load_cached_prices()


def _run_mm_refresh():
    """Intraday refresh: fetch today's live prices via yfinance.

    RVX EQD only provides settled daily closes (no intraday).
    yfinance provides live prices during market hours (~25s for 500 tickers).
    Historical data (10yr) is pre-populated from RVX via the offline script.
    """
    import yfinance as yf

    cached = load_cached_prices()

    if cached is None or cached.empty:
        st.error(
            "No historical data cached. Run the bulk historical fetch first "
            "(see CLAUDE.md for instructions)."
        )
        return

    # Map cached tickers to yfinance format
    # SPX -> ^GSPC for yfinance; also fetch SPY as live proxy
    cached_tickers = list(cached.columns)
    yf_tickers = [t for t in cached_tickers if t != "SPX"]
    # Add SPY to get live S&P 500 proxy; we'll also try ^GSPC
    yf_tickers.append("^GSPC")

    with st.spinner("Fetching live prices via yfinance (~25s)..."):
        data = yf.download(
            yf_tickers, period="1d", interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )

    if data.empty:
        st.error("yfinance returned no data.")
        return

    # Extract closes
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    # Rename ^GSPC back to SPX for our cache
    if "^GSPC" in closes.columns:
        closes = closes.rename(columns={"^GSPC": "SPX"})

    # Only keep tickers that are in our cache
    valid_cols = [c for c in closes.columns if c in cached.columns]
    closes = closes[valid_cols]

    if closes.empty:
        st.error("No matching tickers returned from yfinance.")
        return

    # Merge with existing cache
    merged = merge_incremental(cached, closes)
    save_prices(merged)
    _load_mm_prices.clear()

    # Append a signal snapshot for cross-day delta tracking (Phase 2).
    # Failure here must not block the refresh — log and continue.
    try:
        from analytics.signal_history import append_snapshot, build_snapshot

        sector_tickers = [c for c in merged.columns if c != "SPX"]
        factor_returns = factor_data.returns if factor_data is not None else None
        snap = build_snapshot(merged[sector_tickers], factor_returns=factor_returns)
        append_snapshot(snap)
    except Exception as snap_err:  # pragma: no cover — defensive
        st.warning(f"Snapshot append failed (deltas may be stale): {snap_err}")

    n_valid = closes.iloc[-1].dropna().count()
    latest_date = closes.index[-1].date()
    st.success(f"Live prices updated: {n_valid} tickers as of {latest_date}")


# Handle refresh trigger from sidebar
if st.session_state.pop("mm_trigger_refresh", False):
    try:
        _run_mm_refresh()
    except Exception as e:
        st.error(f"Market data refresh failed: {e}")

mm_prices = _load_mm_prices()
mm_data_available = mm_prices is not None and not mm_prices.empty

# --- Market Snapshot ---
with tab_mm_snapshot:
    render_market_snapshot_tab(mm_prices, mm_data_available)

# --- Sector & Reversion ---
with tab_mm_sector:
    render_sector_reversion_tab(mm_prices, mm_data_available)

# --- Factor Monitor ---
with tab_mm_factors:
    render_factor_monitor_tab(mm_prices, factor_data, mm_data_available)

# --- Trade Ideas ---
with tab_mm_trades:
    render_trade_ideas_tab(mm_prices, factor_data, mm_data_available)

# --- Watchlist & Changes ---
with tab_mm_watchlist:
    render_watchlist_tab(mm_prices, mm_data_available)

# --- Universe Screener ---
with tab_mm_screener:
    render_universe_screen_tab(mm_prices, mm_data_available)
