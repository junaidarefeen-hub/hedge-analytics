import streamlit as st

st.set_page_config(page_title="Hedge Analytics", layout="wide")

import pandas as pd

from analytics.beta import beta_matrix, rolling_beta
from analytics.correlation import correlation_matrix, rolling_correlation
from analytics.returns import compute_returns
from data.fetcher import fetch_ticker_names, validate_and_fetch
from ui.matrices import beta_heatmap, correlation_heatmap
from ui.sidebar import render_sidebar
from ui.style import inject_css
from ui.backtest import render_backtest_tab
from ui.compare import render_compare_tab
from ui.optimizer import render_optimizer_tab
from ui.timeseries import rolling_beta_chart, rolling_correlation_chart
from utils.validation import check_data_sufficiency

inject_css()

st.title("Hedge Analytics")

params = render_sidebar()
if params is None:
    st.stop()

# Fetch data for both groups
stock_prices, stock_failed = validate_and_fetch(
    params["stock_tickers"], params["start_date"], params["end_date"]
)
factor_prices, factor_failed = validate_and_fetch(
    params["factor_tickers"], params["start_date"], params["end_date"]
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

# Tabs
tab_data, tab_corr, tab_beta, tab_optim, tab_compare, tab_backtest = st.tabs(
    ["Data", "Correlation", "Beta", "Hedge Optimizer", "Strategy Compare", "Backtest"]
)

# --- Data Tab ---
with tab_data:
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

# --- Correlation Tab ---
with tab_corr:
    col_left, col_right = st.columns(2)

    with col_left:
        corr = correlation_matrix(returns)
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    with col_right:
        ta, tb = params["ticker_a"], params["ticker_b"]
        if ta in returns.columns and tb in returns.columns:
            rc = rolling_correlation(returns, ta, tb, params["window"])
            st.plotly_chart(
                rolling_correlation_chart(rc, ta, tb, params["window"]),
                use_container_width=True,
            )
        else:
            st.info("Select valid tickers in the sidebar for rolling correlation.")

# --- Beta Tab ---
with tab_beta:
    benchmarks = [b for b in params["benchmarks"] if b in returns.columns]
    if not benchmarks:
        st.info("Add at least one factor/index ticker to compute betas.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            bm = beta_matrix(returns, benchmarks)
            st.plotly_chart(beta_heatmap(bm), use_container_width=True)

        with col_right:
            bt = params["beta_ticker"]
            bb = params["beta_benchmark"]
            if bt in returns.columns and bb in returns.columns:
                rb = rolling_beta(returns, bt, bb, params["window"])
                st.plotly_chart(
                    rolling_beta_chart(rb, bt, bb, params["window"]),
                    use_container_width=True,
                )
            else:
                st.info("Select valid ticker/benchmark for rolling beta.")

# --- Hedge Optimizer ---
with tab_optim:
    render_optimizer_tab(returns, params)

# --- Strategy Compare ---
with tab_compare:
    render_compare_tab(returns, params)

# --- Backtest ---
with tab_backtest:
    render_backtest_tab(returns, params)
