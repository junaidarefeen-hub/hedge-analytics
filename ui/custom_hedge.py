"""Custom Hedge Analyzer tab — user-defined long/hedge portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.custom_hedge import CustomHedgeResult, run_custom_hedge_analysis
from analytics.drawdown import compute_drawdowns
from config import (
    CHA_DEFAULT_HEDGE_NOTIONAL,
    CHA_DEFAULT_LONG_NOTIONAL,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_ROLLING_VOL_WINDOW,
)
from ui.style import PLOTLY_LAYOUT, render_metrics_table


def _equal_weight(n: int) -> float:
    """Return equal weight percentage for *n* constituents."""
    return round(100.0 / n, 2) if n > 0 else 0.0


def _sync_weights(prefix: str, tickers: list[str]) -> None:
    """Reset weight keys to equal weight when the ticker list changes.

    Runs BEFORE widgets are rendered so session_state is clean.
    """
    state_key = f"{prefix}_prev_tickers"
    prev = st.session_state.get(state_key, None)
    if prev != tickers:
        eq = _equal_weight(len(tickers))
        for tk in tickers:
            st.session_state[f"{prefix}_{tk}"] = eq
        # Clean up keys for removed tickers
        if prev is not None:
            for old_tk in prev:
                if old_tk not in tickers:
                    st.session_state.pop(f"{prefix}_{old_tk}", None)
        st.session_state[state_key] = list(tickers)


def _handle_normalize(prefix: str, tickers: list[str]) -> None:
    """If a normalize was requested last run, apply it before widgets render."""
    flag = f"{prefix}_do_normalize"
    if st.session_state.pop(flag, False):
        vals = {tk: st.session_state.get(f"{prefix}_{tk}", 0.0) for tk in tickers}
        total = sum(vals.values())
        if total > 0:
            for tk in tickers:
                st.session_state[f"{prefix}_{tk}"] = round(vals[tk] / total * 100, 2)


def render_custom_hedge_tab(returns: pd.DataFrame, params: dict):
    """Render the Custom Hedge Analyzer tab."""
    all_tickers = [c for c in returns.columns]

    st.caption(
        "Define a custom multi-asset long portfolio and a hedge (short) portfolio with your own "
        "notional splits and weights. Analyze standalone vs hedged performance, drawdowns, "
        "correlations, net beta, and P&L attribution."
    )

    # ── Portfolio Definition ──────────────────────────────────────────────
    col_long, col_hedge = st.columns(2)

    # --- Long side ---
    with col_long:
        st.subheader("Long Portfolio")
        long_notional = st.number_input(
            "Long notional ($)",
            min_value=1000.0,
            value=CHA_DEFAULT_LONG_NOTIONAL,
            step=1_000_000.0,
            format="%.0f",
            key="cha_long_notional",
            help="Dollar value of your long portfolio.",
        )
        long_tickers = st.multiselect(
            "Long tickers",
            options=all_tickers,
            default=[all_tickers[0]] if all_tickers else [],
            key="cha_long_tickers",
            help="Select tickers for the long portfolio.",
        )

    # --- Hedge side ---
    with col_hedge:
        st.subheader("Hedge (Short) Portfolio")
        hedge_notional = st.number_input(
            "Hedge notional ($)",
            min_value=1000.0,
            value=CHA_DEFAULT_HEDGE_NOTIONAL,
            step=1_000_000.0,
            format="%.0f",
            key="cha_hedge_notional",
            help="Dollar value of your hedge portfolio.",
        )
        hedge_tickers = st.multiselect(
            "Hedge tickers",
            options=all_tickers,
            default=[all_tickers[-1]] if len(all_tickers) > 1 else [],
            key="cha_hedge_tickers",
            help="Select tickers for the hedge (short) portfolio.",
        )

    # Pre-render: sync weights to equal-weight on ticker change & apply pending normalizes
    _sync_weights("cha_lw", long_tickers)
    _handle_normalize("cha_lw", long_tickers)
    _sync_weights("cha_hw", hedge_tickers)
    _handle_normalize("cha_hw", hedge_tickers)

    # --- Render weight inputs ---
    with col_long:
        long_weights_raw = {}
        if long_tickers:
            eq = _equal_weight(len(long_tickers))
            for tk in long_tickers:
                long_weights_raw[tk] = st.number_input(
                    f"{tk} weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=eq,
                    step=1.0,
                    key=f"cha_lw_{tk}",
                )
            long_sum = sum(long_weights_raw.values())
            st.caption(f"Weight sum: **{long_sum:.1f}%**")
            if st.button("Normalize to 100%", key="cha_norm_long"):
                st.session_state["cha_lw_do_normalize"] = True
                st.rerun()

    with col_hedge:
        hedge_weights_raw = {}
        if hedge_tickers:
            eq = _equal_weight(len(hedge_tickers))
            for tk in hedge_tickers:
                hedge_weights_raw[tk] = st.number_input(
                    f"{tk} weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=eq,
                    step=1.0,
                    key=f"cha_hw_{tk}",
                )
            hedge_sum = sum(hedge_weights_raw.values())
            st.caption(f"Weight sum: **{hedge_sum:.1f}%**")
            if st.button("Normalize to 100%", key="cha_norm_hedge"):
                st.session_state["cha_hw_do_normalize"] = True
                st.rerun()

    # ── Analysis Settings ─────────────────────────────────────────────────
    st.divider()
    min_date = returns.index.min().date()
    max_date = returns.index.max().date()

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        cha_start = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="cha_start",
            help="Start of the analysis period.",
        )
    with s2:
        cha_end = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="cha_end",
            help="End of the analysis period.",
        )
    with s3:
        rolling_window = st.number_input(
            "Rolling window (days)",
            min_value=10,
            max_value=252,
            value=DEFAULT_ROLLING_VOL_WINDOW,
            step=10,
            key="cha_roll_window",
            help="Window for rolling volatility and correlation charts.",
        )
    with s4:
        risk_free = st.number_input(
            "Risk-free rate",
            min_value=0.0,
            max_value=0.20,
            value=DEFAULT_RISK_FREE_RATE,
            step=0.01,
            format="%.2f",
            key="cha_rf",
            help="Annual risk-free rate for Sharpe and Sortino ratios.",
        )
    with s5:
        selected_benchmark = None
        if hedge_tickers:
            selected_benchmark = st.selectbox(
                "Benchmark (for net beta)",
                options=hedge_tickers,
                index=0,
                key="cha_benchmark",
                help="Select a hedge constituent to see your net portfolio beta against it. "
                     "Shows how much exposure to this instrument remains after hedging.",
            )

    # ── Run Button ────────────────────────────────────────────────────────
    if not long_tickers:
        st.info("Select at least one long ticker to begin.")
        return
    if not hedge_tickers:
        st.info("Select at least one hedge ticker to begin.")
        return

    long_w_arr = np.array([long_weights_raw[tk] for tk in long_tickers]) / 100.0
    hedge_w_arr = np.array([hedge_weights_raw[tk] for tk in hedge_tickers]) / 100.0

    run_analysis = st.button("Analyze", type="primary", use_container_width=True, key="cha_run")

    if run_analysis:
        try:
            result = run_custom_hedge_analysis(
                returns=returns,
                long_tickers=long_tickers,
                long_weights=long_w_arr,
                long_notional=long_notional,
                hedge_tickers=hedge_tickers,
                hedge_weights=hedge_w_arr,
                hedge_notional=hedge_notional,
                benchmarks=[selected_benchmark] if selected_benchmark else None,
                rolling_window=rolling_window,
                risk_free=risk_free,
                start_date=pd.Timestamp(cha_start),
                end_date=pd.Timestamp(cha_end),
            )
            st.session_state["cha_result"] = result
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    result: CustomHedgeResult | None = st.session_state.get("cha_result")
    if result is None:
        st.info("Configure portfolios and click **Analyze** to see results.")
        return

    # ── Results ───────────────────────────────────────────────────────────
    _render_results(result, rolling_window)


def _render_results(result: CustomHedgeResult, rolling_window: int):
    """Render all result sections."""
    # 1. Summary metric cards
    standalone_vol = result.metrics.loc["Ann. Volatility", "Standalone"]
    hedged_vol = result.metrics.loc["Ann. Volatility", "Hedged"]
    vol_reduction = result.metrics.loc["Vol Reduction", "Hedged"]
    net_beta_str = "N/A"
    if len(result.beta_table) > 0:
        net_rows = result.beta_table[result.beta_table["Component"] == "Net Portfolio"]
        if len(net_rows) > 0:
            net_beta_str = f"{net_rows.iloc[0]['Beta Contribution']:.3f}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Standalone Vol (ann.)", f"{standalone_vol:.1%}")
    m2.metric("Hedged Vol (ann.)", f"{hedged_vol:.1%}")
    m3.metric("Vol Reduction", f"{vol_reduction:.1%}")
    m4.metric("Net Beta", net_beta_str)

    # 2. Performance metrics table
    st.subheader("Performance Metrics")
    pct_rows = {"Total Return", "Ann. Return", "Ann. Volatility", "Max Drawdown", "Tracking Error", "Vol Reduction"}
    int_rows = {"Max DD Duration (days)"}
    fmt_metrics = result.metrics.copy()
    for col in fmt_metrics.columns:
        fmt_metrics[col] = fmt_metrics.index.map(
            lambda idx, c=col: f"{result.metrics.loc[idx, c]:.2%}"
            if idx in pct_rows
            else (f"{int(result.metrics.loc[idx, c])}" if idx in int_rows
                  else f"{result.metrics.loc[idx, c]:.2f}")
        )
    render_metrics_table(fmt_metrics)

    # 3. Cumulative return chart
    st.plotly_chart(_cumulative_chart(result), use_container_width=True)

    # 4. Rolling volatility chart
    st.plotly_chart(_rolling_vol_chart(result, rolling_window), use_container_width=True)

    # 5. Rolling correlation chart
    st.plotly_chart(_rolling_corr_chart(result, rolling_window), use_container_width=True)

    # 6. Drawdown comparison
    st.subheader("Drawdown Comparison")
    dd_standalone = compute_drawdowns(result.cumulative_standalone)
    dd_hedged = compute_drawdowns(result.cumulative_hedged)
    st.plotly_chart(
        _drawdown_chart(dd_standalone.underwater_series, dd_hedged.underwater_series),
        use_container_width=True,
    )

    # 7. P&L attribution chart
    st.subheader("P&L Attribution")
    st.plotly_chart(_attribution_chart(result), use_container_width=True)

    # 8. Net beta table
    if len(result.beta_table) > 0:
        st.subheader("Net Portfolio Beta")
        st.dataframe(result.beta_table, use_container_width=True, hide_index=True)

    # Hedge efficiency
    st.caption(
        f"Hedge efficiency: **{result.hedge_efficiency:.2f}** "
        f"(vol reduction % per unit of annualized return sacrifice %) | "
        f"Hedge ratio: **{result.hedge_ratio:.2%}** | "
        f"Full-period correlation: **{result.full_period_correlation:.3f}**"
    )


# ── Chart Helpers ─────────────────────────────────────────────────────────


def _cumulative_chart(result: CustomHedgeResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.cumulative_standalone.index,
        y=result.cumulative_standalone.values,
        mode="lines",
        name="Standalone",
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


def _rolling_vol_chart(result: CustomHedgeResult, window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.rolling_vol_standalone.index,
        y=result.rolling_vol_standalone.values,
        mode="lines",
        name="Standalone",
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


def _rolling_corr_chart(result: CustomHedgeResult, window: int) -> go.Figure:
    series = result.rolling_correlation.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name="Long vs Hedge Basket",
        line=dict(width=2, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>Corr: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=result.full_period_correlation,
        line_dash="dash",
        line_color="#dc2626",
        line_width=1,
        annotation_text=f"Full-period: {result.full_period_correlation:.3f}",
        annotation_position="top left",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{window}-Day Rolling Correlation: Long vs Hedge Basket",
        xaxis_title="",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
        height=380,
        showlegend=False,
    )
    return fig


def _drawdown_chart(
    underwater_standalone: "pd.Series",
    underwater_hedged: "pd.Series",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=underwater_standalone.index,
        y=underwater_standalone.values,
        mode="lines",
        name="Standalone",
        line=dict(width=2, color="#2563eb"),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.1)",
        hovertemplate="%{x|%b %d, %Y}<br>DD: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=underwater_hedged.index,
        y=underwater_hedged.values,
        mode="lines",
        name="Hedged",
        line=dict(width=2, color="#16a34a"),
        fill="tozeroy",
        fillcolor="rgba(22, 163, 74, 0.1)",
        hovertemplate="%{x|%b %d, %Y}<br>DD: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Underwater (Drawdown) Chart",
        xaxis_title="",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        height=350,
    )
    return fig


def _attribution_chart(result: CustomHedgeResult) -> go.Figure:
    contrib = result.constituent_contributions
    cum_contrib = contrib.cumsum()
    fig = go.Figure()
    for col in cum_contrib.columns:
        fig.add_trace(go.Scatter(
            x=cum_contrib.index,
            y=cum_contrib[col].values,
            mode="lines",
            name=col,
            stackgroup="one",
            hovertemplate=f"{col}<br>" + "%{x|%b %d, %Y}<br>Contrib: %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Cumulative P&L Attribution by Constituent",
        xaxis_title="",
        yaxis_title="Cumulative Contribution",
        height=400,
    )
    return fig
