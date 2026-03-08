"""Backtest tab — charts + metrics table."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.backtest import BacktestResult, DynamicBacktestResult, run_backtest, run_dynamic_backtest
from config import (
    DEFAULT_LOOKBACK_WINDOW,
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_ROLLING_VOL_WINDOW,
    REBALANCE_FREQUENCIES,
)
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
    pct_rows = {"Total Return", "Ann. Return", "Ann. Volatility", "Max Drawdown", "Tracking Error"}
    int_rows = {"Max DD Duration (days)"}
    fmt_metrics = bt_result.metrics.copy()
    for col in fmt_metrics.columns:
        fmt_metrics[col] = fmt_metrics.index.map(
            lambda idx, c=col: f"{bt_result.metrics.loc[idx, c]:.2%}"
            if idx in pct_rows
            else (f"{int(bt_result.metrics.loc[idx, c])}" if idx in int_rows
                  else f"{bt_result.metrics.loc[idx, c]:.2f}")
        )
    st.dataframe(fmt_metrics, use_container_width=True)

    # --- Dynamic Rebalancing Backtest ---
    st.divider()
    mode = st.radio(
        "Backtest mode",
        options=["Static Backtest", "Dynamic Rebalancing Backtest"],
        index=0,
        horizontal=True,
        key="bt_mode",
    )

    if mode == "Dynamic Rebalancing Backtest":
        _render_dynamic_backtest(returns, params, hedge_result, bt_result)


def _render_dynamic_backtest(returns, params, hedge_result, static_bt_result):
    """Render dynamic rebalancing backtest controls and results."""
    dc1, dc2 = st.columns(2)
    with dc1:
        rebal_freq = st.selectbox(
            "Rebalance frequency",
            options=REBALANCE_FREQUENCIES,
            index=REBALANCE_FREQUENCIES.index(DEFAULT_REBALANCE_FREQUENCY),
            key="dyn_freq",
        )
    with dc2:
        lookback = st.number_input(
            "Lookback window (days)",
            min_value=30,
            max_value=504,
            value=DEFAULT_LOOKBACK_WINDOW,
            step=10,
            key="dyn_lookback",
        )

    run_dyn = st.button("Run Dynamic Backtest", type="primary", key="dyn_run")

    if run_dyn:
        progress = st.progress(0, text="Running dynamic rebalancing...")
        try:
            dyn_result = run_dynamic_backtest(
                returns=returns,
                target=hedge_result.target_ticker,
                hedges=hedge_result.hedge_instruments,
                static_weights=hedge_result.weights,
                strategy=hedge_result.strategy,
                bounds=(-1.0, 0.0) if hedge_result.weights[0] <= 0 else (0.0, 1.0),
                rebalance_freq=rebal_freq,
                lookback_window=lookback,
                rolling_window=st.session_state.get("bt_roll_window", DEFAULT_ROLLING_VOL_WINDOW),
                risk_free=st.session_state.get("bt_rf", DEFAULT_RISK_FREE_RATE),
                factors=list(hedge_result.portfolio_betas.keys()) if hedge_result.portfolio_betas else None,
                confidence=hedge_result.confidence_level or 0.95,
                min_names=0,
                notional=hedge_result.target_notional,
                progress_callback=lambda p: progress.progress(p, text=f"Rebalancing... {p:.0%}"),
            )
            st.session_state["dynamic_bt_result"] = dyn_result
            progress.empty()
        except Exception as e:
            progress.empty()
            st.error(f"Dynamic backtest failed: {e}")
            return

    dyn_result: DynamicBacktestResult | None = st.session_state.get("dynamic_bt_result")
    if dyn_result is None:
        st.info("Configure parameters and click **Run Dynamic Backtest**.")
        return

    # 3-line cumulative return chart
    fig_cum = go.Figure()
    for name, series, color in [
        ("Unhedged", dyn_result.cumulative_unhedged, "#2563eb"),
        ("Static Hedge", dyn_result.cumulative_static, "#16a34a"),
        ("Dynamic Hedge", dyn_result.cumulative_dynamic, "#dc2626"),
    ]:
        fig_cum.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name=name,
            line=dict(width=2, color=color),
        ))
    fig_cum.update_layout(**PLOTLY_LAYOUT)
    fig_cum.update_layout(title="Cumulative Returns: Unhedged vs Static vs Dynamic", height=400)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Weight evolution
    fig_wt = go.Figure()
    for col in dyn_result.weight_history.columns:
        fig_wt.add_trace(go.Scatter(
            x=dyn_result.weight_history.index,
            y=dyn_result.weight_history[col].values,
            mode="lines",
            name=col,
            stackgroup="one",
        ))
    fig_wt.update_layout(**PLOTLY_LAYOUT)
    fig_wt.update_layout(title="Weight Evolution Over Time", height=350)
    st.plotly_chart(fig_wt, use_container_width=True)

    # Turnover bars
    if len(dyn_result.turnover) > 0:
        fig_to = go.Figure(data=go.Bar(
            x=dyn_result.turnover.index,
            y=dyn_result.turnover.values,
            marker_color="#7c3aed",
        ))
        fig_to.update_layout(**PLOTLY_LAYOUT)
        fig_to.update_layout(title="Turnover at Rebalance Dates", height=300, yaxis_title="Turnover")
        st.plotly_chart(fig_to, use_container_width=True)

    # 3-column metrics table
    st.subheader("Dynamic Backtest Metrics")
    pct_rows = {"Total Return", "Ann. Return", "Ann. Volatility", "Max Drawdown", "Tracking Error"}
    int_rows = {"Max DD Duration (days)"}
    fmt = dyn_result.metrics.copy()
    for col in fmt.columns:
        fmt[col] = fmt.index.map(
            lambda idx, c=col: f"{dyn_result.metrics.loc[idx, c]:.2%}"
            if idx in pct_rows
            else (f"{int(dyn_result.metrics.loc[idx, c])}" if idx in int_rows
                  else f"{dyn_result.metrics.loc[idx, c]:.2f}")
        )
    st.dataframe(fmt, use_container_width=True)
