import streamlit as st

from config import (
    DEFAULT_END_DATE,
    DEFAULT_FACTOR_TICKERS,
    DEFAULT_RETURN_METHOD,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_START_DATE,
    DEFAULT_STOCK_TICKERS,
    RETURN_METHODS,
    ROLLING_WINDOW_OPTIONS,
)
from utils.validation import parse_tickers, validate_date_range


def render_sidebar() -> dict | None:
    """Render all sidebar inputs and return a params dict, or None if inputs are invalid."""
    with st.sidebar:
        st.header("Settings")

        # --- Stock tickers ---
        st.subheader("Stocks / ETFs")
        raw_stocks = st.text_area(
            "Stock tickers (comma-separated)",
            value=DEFAULT_STOCK_TICKERS,
            help="Enter US stock or ETF tickers separated by commas.",
            key="stock_tickers",
        )
        stock_tickers = parse_tickers(raw_stocks)

        # --- Factor / Index tickers ---
        st.subheader("Factors / Indices")
        raw_factors = st.text_area(
            "Factor & index tickers (comma-separated)",
            value=DEFAULT_FACTOR_TICKERS,
            help="Enter factor ETFs, sector ETFs, or index tickers (e.g. SPY, QQQ, IWM, XLF).",
            key="factor_tickers",
        )
        factor_tickers = parse_tickers(raw_factors)

        # Deduplicate across both lists (factor list wins if overlap)
        stock_tickers = [t for t in stock_tickers if t not in factor_tickers]

        all_tickers = stock_tickers + factor_tickers
        if len(all_tickers) < 2:
            st.error("Enter at least 2 valid tickers total across stocks and factors.")
            return None

        if not factor_tickers:
            st.warning("Add at least one factor/index ticker for beta analysis.")

        st.divider()

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=DEFAULT_START_DATE)
        with col2:
            end_date = st.date_input("End", value=DEFAULT_END_DATE)

        date_err = validate_date_range(start_date, end_date)
        if date_err:
            st.error(date_err)
            return None

        st.divider()

        # Rolling window
        window = st.select_slider(
            "Rolling window (days)",
            options=ROLLING_WINDOW_OPTIONS,
            value=DEFAULT_ROLLING_WINDOW,
        )

        # Return method
        return_method = st.radio(
            "Return method",
            options=RETURN_METHODS,
            index=RETURN_METHODS.index(DEFAULT_RETURN_METHOD),
            horizontal=True,
        )

        st.divider()

        # Pair selection for rolling charts
        st.subheader("Pair selection")
        ticker_a = st.selectbox("Ticker A", options=all_tickers, index=0)
        ticker_b = st.selectbox(
            "Ticker B",
            options=all_tickers,
            index=min(1, len(all_tickers) - 1),
        )

        if ticker_a == ticker_b:
            st.info("Same ticker selected — correlation = 1.0, beta = 1.0.")

        # Benchmark for rolling beta
        beta_benchmark = None
        beta_ticker = stock_tickers[0] if stock_tickers else all_tickers[0]
        if factor_tickers:
            beta_benchmark = st.selectbox(
                "Rolling beta benchmark",
                options=factor_tickers,
                index=0,
            )
            non_benchmark = [t for t in all_tickers if t != beta_benchmark]
            beta_ticker = st.selectbox(
                "Rolling beta ticker",
                options=non_benchmark or all_tickers,
                index=0,
            )

    return {
        "stock_tickers": stock_tickers,
        "factor_tickers": factor_tickers,
        "all_tickers": all_tickers,
        "benchmarks": factor_tickers,  # factors/indices serve as benchmarks
        "start_date": start_date,
        "end_date": end_date,
        "window": window,
        "return_method": return_method,
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "beta_benchmark": beta_benchmark,
        "beta_ticker": beta_ticker,
    }
