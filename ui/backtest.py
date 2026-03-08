"""Backtest tab — charts + metrics table."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.backtest import BacktestResult, run_backtest
from config import DEFAULT_RISK_FREE_RATE, DEFAULT_ROLLING_VOL_WINDOW
from ui.optimizer import _params_hash
from ui.style import PLOTLY_LAYOUT


def _cumulative_chart(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.cumulative_unhedged.index,
        y=result.cumulative_unhedged.values,
        mode="lines",
        name="Unhedged",
        line=dict(width=2, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=result.cumulative_hedged.index,
        y=result.cumulative_hedged.values,
        mode="lines",
        name="Hedged",
        line=dict(width=2, color="#16a34a"),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Cumulative Returns (Growth of $1)",
        xaxis_title="",
        yaxis_title="Cumulative Return",
        height=400,
    )
    return fig


def _rolling_vol_chart(result: BacktestResult, window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.rolling_vol_unhedged.index,
        y=result.rolling_vol_unhedged.values,
        mode="lines",
        name="Unhedged",
        line=dict(width=2, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>Vol: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=result.rolling_vol_hedged.index,
        y=result.rolling_vol_hedged.values,
        mode="lines",
        name="Hedged",
        line=dict(width=2, color="#16a34a"),
        hovertemplate="%{x|%b %d, %Y}<br>Vol: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{window}-Day Rolling Volatility (Annualized)",
        xaxis_title="",
        yaxis_title="Volatility",
        height=400,
    )
    return fig


def render_backtest_tab(returns: pd.DataFrame, params: dict):
    """Render the Backtest tab."""
    hedge_result = st.session_state.get("hedge_result")

    if hedge_result is None:
        st.info("Run the **Hedge Optimizer** first, then come back here to backtest.")
        return

    # Staleness check
    stored_hash = st.session_state.get("hedge_params_hash")
    current_hash = _params_hash(params)
    if stored_hash and stored_hash != current_hash:
        st.warning("Sidebar parameters have changed since last optimization. Backtest may use stale hedge weights.")

    st.caption(f"Strategy: **{hedge_result.strategy}** | Target: **{hedge_result.target_ticker}**")

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    min_date = returns.index.min().date()
    max_date = returns.index.max().date()

    with c1:
        bt_start = st.date_input("Backtest start", value=min_date, min_value=min_date, max_value=max_date, key="bt_start")
    with c2:
        bt_end = st.date_input("Backtest end", value=max_date, min_value=min_date, max_value=max_date, key="bt_end")
    with c3:
        roll_window = st.number_input("Rolling vol window", min_value=10, max_value=252, value=DEFAULT_ROLLING_VOL_WINDOW, step=10, key="bt_roll")
    with c4:
        risk_free = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=DEFAULT_RISK_FREE_RATE, step=0.01, format="%.2f", key="bt_rf")

    run_bt = st.button("Run Backtest", type="primary", key="bt_run")

    if run_bt:
        try:
            bt_result = run_backtest(
                returns=returns,
                target=hedge_result.target_ticker,
                hedge_instruments=hedge_result.hedge_instruments,
                weights=hedge_result.weights,
                start_date=pd.Timestamp(bt_start),
                end_date=pd.Timestamp(bt_end),
                rolling_window=roll_window,
                risk_free=risk_free,
            )
            st.session_state["backtest_result"] = bt_result
            st.session_state["bt_roll_window"] = roll_window
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            return

    bt_result: BacktestResult | None = st.session_state.get("backtest_result")

    if bt_result is None:
        st.info("Set backtest parameters and click **Run Backtest**.")
        return

    roll_w = st.session_state.get("bt_roll_window", DEFAULT_ROLLING_VOL_WINDOW)

    # Charts
    st.plotly_chart(_cumulative_chart(bt_result), use_container_width=True)
    st.plotly_chart(_rolling_vol_chart(bt_result, roll_w), use_container_width=True)

    # Metrics table
    st.subheader("Performance Metrics")
    pct_rows = {"Total Return", "Ann. Return", "Ann. Volatility", "Max Drawdown"}
    fmt_metrics = bt_result.metrics.copy()
    for col in fmt_metrics.columns:
        fmt_metrics[col] = fmt_metrics.index.map(
            lambda idx, c=col: f"{bt_result.metrics.loc[idx, c]:.2%}"
            if idx in pct_rows
            else f"{bt_result.metrics.loc[idx, c]:.2f}"
        )
    st.dataframe(fmt_metrics, use_container_width=True)
