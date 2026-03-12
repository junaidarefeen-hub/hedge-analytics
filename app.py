import streamlit as st

st.set_page_config(page_title="Hedge Analytics", layout="wide")

import pandas as pd

from analytics.beta import beta_matrix, rolling_beta
from analytics.correlation import correlation_clustering, correlation_matrix, rolling_correlation
from analytics.returns import compute_returns
from data.factor_loader import load_factor_data
from data.fetcher import fetch_ticker_names, validate_and_fetch
from ui.drawdown import render_drawdown_tab
from ui.matrices import beta_heatmap, correlation_dendrogram, correlation_heatmap
from ui.regime import render_regime_tab
from ui.sidebar import render_sidebar
from ui.style import inject_css
from ui.backtest import render_backtest_tab
from ui.compare import render_compare_tab
from ui.custom_hedge import render_custom_hedge_tab
from ui.factor_analytics import render_factor_analytics_tab
from ui.montecarlo import render_montecarlo_tab
from ui.optimizer import render_optimizer_tab
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

# Load GS factor data from Excel
try:
    factor_data = load_factor_data()
except Exception as e:
    st.warning(f"Could not load Factor Prices.xlsx — Factor Analytics tab will be unavailable. ({e})")
    factor_data = None

# Tabs
tab_data, tab_corr, tab_beta, tab_optim, tab_compare, tab_custom, tab_backtest, tab_mc, tab_stress, tab_dd, tab_regime, tab_factor = st.tabs(
    ["Data", "Correlation", "Beta", "Hedge Optimizer", "Strategy Compare", "Custom Hedge", "Backtest", "Monte Carlo", "Stress Test", "Drawdown", "Regime", "Factor Analytics"]
)

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

# --- Hedge Optimizer ---
with tab_optim:
    render_optimizer_tab(returns, params)

# --- Strategy Compare ---
with tab_compare:
    render_compare_tab(returns, params)

# --- Custom Hedge ---
with tab_custom:
    render_custom_hedge_tab(returns, params)

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
