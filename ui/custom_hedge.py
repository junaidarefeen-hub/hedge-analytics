"""Custom Hedge Analyzer tab — user-defined long/hedge portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.custom_hedge import (
    CustomHedgeResult,
    compute_net_beta,
    run_custom_hedge_analysis,
)
from analytics.drawdown import compute_drawdowns
from analytics.factor_analytics import FactorAnalyticsResult, run_factor_analytics
from analytics.montecarlo import MonteCarloResult, run_monte_carlo
from config import (
    CHA_DEFAULT_HEDGE_NOTIONAL,
    CHA_DEFAULT_LONG_NOTIONAL,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_ROLLING_VOL_WINDOW,
    FA_DEFAULT_P_THRESHOLD,
    MC_DEFAULT_HORIZON,
    MC_DEFAULT_NUM_SIMS,
    MC_DEFAULT_SEED,
    MC_HORIZON_OPTIONS,
    MC_NUM_SIMS_OPTIONS,
    MC_SPAGHETTI_PATHS,
)
from data.factor_loader import FactorData, align_factor_returns
from ui.factor_analytics import (
    _active_legs,
    _beta_heatmap_chart,
    _render_regression_table,
    _render_return_decomposition,
    _render_vol_decomposition,
)
from ui.montecarlo import _distribution_chart, _fan_chart, _spaghetti_chart
from ui.style import PLOTLY_LAYOUT, render_metrics_table
from ui.weight_helpers import equal_weight, handle_normalize, sync_weights
from utils.basket import inject_basket_column


def render_custom_hedge_tab(
    returns: pd.DataFrame, params: dict, factor_data: FactorData | None = None,
):
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
    sync_weights("cha_lw", long_tickers)
    handle_normalize("cha_lw", long_tickers)
    sync_weights("cha_hw", hedge_tickers)
    handle_normalize("cha_hw", hedge_tickers)

    # --- Render weight inputs ---
    with col_long:
        long_weights_raw = {}
        if long_tickers:
            eq = equal_weight(len(long_tickers))
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
            eq = equal_weight(len(hedge_tickers))
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
        benchmark_options = [b for b in params.get("benchmarks", []) if b in returns.columns]
        if not benchmark_options and hedge_tickers:
            benchmark_options = list(hedge_tickers)
        if benchmark_options:
            default_idx = benchmark_options.index("SPY") if "SPY" in benchmark_options else 0
            selected_benchmark = st.selectbox(
                "Benchmark (for net beta)",
                options=benchmark_options,
                index=default_idx,
                key="cha_benchmark",
                help="Index to compute net portfolio beta against. Uses sidebar factor/index tickers.",
            )

    # ── Early exit if portfolios incomplete ───────────────────────────────
    if not long_tickers:
        st.info("Select at least one long ticker to begin.")
        return
    if not hedge_tickers:
        st.info("Select at least one hedge ticker to begin.")
        return

    long_w_arr = np.array([long_weights_raw[tk] for tk in long_tickers]) / 100.0
    hedge_w_arr = np.array([hedge_weights_raw[tk] for tk in hedge_tickers]) / 100.0
    hedge_ratio = hedge_notional / long_notional if long_notional > 0 else 0.0

    # ── Live Net Beta (updates without clicking Analyze) ──────────────────
    if selected_benchmark:
        live = compute_net_beta(
            returns=returns,
            long_tickers=long_tickers,
            long_weights=long_w_arr,
            hedge_tickers=hedge_tickers,
            hedge_weights=hedge_w_arr,
            hedge_ratio=hedge_ratio,
            benchmark=selected_benchmark,
            start_date=pd.Timestamp(cha_start),
            end_date=pd.Timestamp(cha_end),
        )
        if live is not None:
            beta_cols = st.columns(2 + len(live["instruments"]))
            beta_cols[0].metric(
                f"Long Beta ({selected_benchmark})",
                f"{live['long_beta']:.3f}",
            )
            for i, inst in enumerate(live["instruments"]):
                beta_cols[1 + i].metric(
                    f"{inst['ticker']} contribution",
                    f"{inst['contribution']:+.3f}",
                    help=f"Beta: {inst['beta']:.3f} | Eff. ratio: {inst['eff_ratio']:.3f}",
                )
            net_val = live["net_beta"]
            beta_cols[-1].metric(
                f"Net Beta ({selected_benchmark})",
                f"{net_val:+.3f}",
                delta=f"{'neutral' if abs(net_val) < 0.05 else ''}",
                delta_color="off" if abs(net_val) < 0.05 else ("inverse" if net_val > 0 else "normal"),
            )

    # ── Run Button ────────────────────────────────────────────────────────
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
    _render_results(result, rolling_window, selected_benchmark)

    # ── Monte Carlo Section ──────────────────────────────────────────────
    _render_montecarlo_section(returns, result)

    # ── Factor Exposure Section ──────────────────────────────────────────
    if factor_data is not None:
        _render_factor_section(returns, result, factor_data, params)


def _render_results(
    result: CustomHedgeResult,
    rolling_window: int,
    selected_benchmark: str | None,
):
    """Render all result sections."""
    # 1. Summary metric cards
    standalone_vol = result.metrics.loc["Ann. Volatility", "Standalone"]
    hedged_vol = result.metrics.loc["Ann. Volatility", "Hedged"]
    vol_reduction = result.metrics.loc["Vol Reduction", "Hedged"]
    net_beta_str = "N/A"
    net_beta_label = "Net Beta"
    if len(result.beta_table) > 0:
        net_rows = result.beta_table[result.beta_table["Component"] == "Net Portfolio"]
        if len(net_rows) > 0:
            net_beta_str = f"{net_rows.iloc[0]['Beta Contribution']:.3f}"
        if result.beta_benchmark:
            net_beta_label = f"Net Beta ({result.beta_benchmark})"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Standalone Vol (ann.)", f"{standalone_vol:.1%}")
    m2.metric("Hedged Vol (ann.)", f"{hedged_vol:.1%}")
    m3.metric("Vol Reduction", f"{vol_reduction:.1%}")
    m4.metric(net_beta_label, net_beta_str)

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

    # 6. Rolling net beta chart — only show if benchmark hasn't changed since Analyze
    beta_bm = result.beta_benchmark
    benchmark_stale = selected_benchmark != beta_bm
    if result.rolling_net_beta is not None and beta_bm:
        if benchmark_stale:
            st.caption(
                f"Rolling net beta was computed against **{beta_bm}**. "
                f"Click **Analyze** to update for **{selected_benchmark}**."
            )
        st.plotly_chart(
            _rolling_net_beta_chart(result, rolling_window, beta_bm),
            use_container_width=True,
        )

    # 7. Drawdown comparison
    st.subheader("Drawdown Comparison")
    dd_standalone = compute_drawdowns(result.cumulative_standalone)
    dd_hedged = compute_drawdowns(result.cumulative_hedged)
    st.plotly_chart(
        _drawdown_chart(dd_standalone.underwater_series, dd_hedged.underwater_series),
        use_container_width=True,
    )

    # 8. P&L attribution chart
    st.subheader("P&L Attribution")
    st.plotly_chart(_attribution_chart(result), use_container_width=True)

    # 9. Net beta table — label with the benchmark it was computed against
    if len(result.beta_table) > 0:
        bm_label = f" (vs {beta_bm})" if beta_bm else ""
        st.subheader(f"Net Portfolio Beta{bm_label}")
        if benchmark_stale:
            st.caption(
                f"Table was computed against **{beta_bm}**. "
                f"Click **Analyze** to update for **{selected_benchmark}**."
            )
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


def _rolling_net_beta_chart(
    result: CustomHedgeResult, window: int, benchmark: str,
) -> go.Figure:
    series = result.rolling_net_beta.dropna()
    # Full-period net beta from the table
    full_net = None
    if len(result.beta_table) > 0:
        net_rows = result.beta_table[result.beta_table["Component"] == "Net Portfolio"]
        if len(net_rows) > 0:
            full_net = net_rows.iloc[0]["Beta Contribution"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name="Net Beta",
        line=dict(width=2, color="#7c3aed"),
        hovertemplate="%{x|%b %d, %Y}<br>Net Beta: %{y:.3f}<extra></extra>",
    ))
    # Zero line (beta-neutral reference)
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626", line_width=1,
                  annotation_text="Beta-neutral",
                  annotation_position="top left")
    if full_net is not None:
        fig.add_hline(y=full_net, line_dash="dot", line_color="#64748b", line_width=1,
                      annotation_text=f"Full-period: {full_net:.3f}",
                      annotation_position="bottom left")
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{window}-Day Rolling Net Beta to {benchmark}",
        xaxis_title="",
        yaxis_title="Net Beta",
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
        is_short = col.endswith("(S)")
        fig.add_trace(go.Scatter(
            x=cum_contrib.index,
            y=cum_contrib[col].values,
            mode="lines",
            name=col,
            stackgroup="short" if is_short else "long",
            hovertemplate=f"{col}<br>" + "%{x|%b %d, %Y}<br>Contrib: %{y:.4f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Cumulative P&L Attribution by Constituent",
        xaxis_title="",
        yaxis_title="Cumulative Contribution",
        height=400,
    )
    return fig


# ── Monte Carlo Section ──────────────────────────────────────────────────


def _render_montecarlo_section(
    returns: pd.DataFrame,
    result: CustomHedgeResult,
):
    """Monte Carlo simulation for the custom hedge position."""
    st.divider()
    st.subheader("Monte Carlo Simulation")
    st.caption(
        "Simulate forward paths for the standalone (unhedged) and hedged portfolios "
        "using Monte Carlo methods based on historical return distributions."
    )

    # Build synthetic long basket column
    augmented_returns, target_col = inject_basket_column(
        returns, result.long_tickers, result.long_weights,
    )

    # MC hedge weights: negative because short positions
    mc_hedge_weights = -result.hedge_ratio * result.hedge_weights

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon = st.selectbox(
            "Horizon (days)",
            options=MC_HORIZON_OPTIONS,
            index=MC_HORIZON_OPTIONS.index(MC_DEFAULT_HORIZON),
            key="cha_mc_horizon",
            help="Number of trading days to simulate forward.",
        )
    with c2:
        num_sims = st.selectbox(
            "Simulations",
            options=MC_NUM_SIMS_OPTIONS,
            index=MC_NUM_SIMS_OPTIONS.index(MC_DEFAULT_NUM_SIMS),
            key="cha_mc_num_sims",
        )
    with c3:
        initial_value = st.number_input(
            "Initial value ($)",
            min_value=1000.0,
            value=result.long_notional,
            step=10_000.0,
            format="%.0f",
            key="cha_mc_initial",
        )
    with c4:
        use_seed = st.checkbox("Reproducible", value=True, key="cha_mc_seed")

    run_mc = st.button(
        "Run Monte Carlo", type="secondary", use_container_width=True, key="cha_mc_run",
    )

    if run_mc:
        with st.spinner(f"Running {num_sims:,} simulations over {horizon} days..."):
            try:
                mc_result = run_monte_carlo(
                    returns=augmented_returns,
                    target=target_col,
                    hedge_instruments=result.hedge_tickers,
                    weights=mc_hedge_weights,
                    strategy="Custom Hedge",
                    horizon=horizon,
                    num_sims=num_sims,
                    initial_value=initial_value,
                    confidence_levels=[0.95, 0.99],
                    seed=MC_DEFAULT_SEED if use_seed else None,
                )
                st.session_state["cha_mc_result"] = mc_result
            except Exception as e:
                st.error(f"Monte Carlo failed: {e}")
                return

    mc_result: MonteCarloResult | None = st.session_state.get("cha_mc_result")
    if mc_result is None:
        st.info("Click **Run Monte Carlo** to simulate forward paths.")
        return

    # Metrics table
    pct_rows = {"Mean Return", "Median Return", "Best Case", "Worst Case"}
    var_rows = {r for r in mc_result.metrics.index if r.startswith("VaR") or r.startswith("CVaR")}
    prob_rows = {r for r in mc_result.metrics.index if r.startswith("P(loss")}
    fmt_metrics = mc_result.metrics.copy()
    for col in fmt_metrics.columns:
        fmt_metrics[col] = mc_result.metrics.index.map(
            lambda idx, c=col: (
                f"{mc_result.metrics.loc[idx, c]:.2%}"
                if idx in pct_rows | var_rows | prob_rows
                else f"{mc_result.metrics.loc[idx, c]:.4f}"
            )
        )
    render_metrics_table(fmt_metrics)

    # Charts
    st.plotly_chart(_distribution_chart(mc_result), use_container_width=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(_fan_chart(mc_result, "hedged"), use_container_width=True)
    with ch2:
        st.plotly_chart(_fan_chart(mc_result, "unhedged"), use_container_width=True)

    with st.expander("Individual Simulated Paths", expanded=False):
        st.plotly_chart(
            _spaghetti_chart(mc_result, MC_SPAGHETTI_PATHS), use_container_width=True,
        )


# ── Factor Exposure Section ──────────────────────────────────────────────


def _render_factor_section(
    returns: pd.DataFrame,
    result: CustomHedgeResult,
    factor_data: FactorData,
    params: dict,
):
    """Factor exposure analysis for the custom hedge combined position."""
    st.divider()
    st.subheader("Factor Exposure Analysis")
    st.caption(
        "OLS regression of the combined (hedged) position against a market index "
        "and GS factor price series. Shows how much of the portfolio's return and risk "
        "is explained by systematic factor exposures."
    )

    factor_display_names = list(factor_data.prices.columns)
    benchmarks = [b for b in params.get("benchmarks", []) if b in returns.columns]

    # Controls
    f1, f2, f3 = st.columns(3)
    with f1:
        market_options = benchmarks if benchmarks else list(returns.columns)[:1]
        fa_market = st.selectbox(
            "Market index",
            options=market_options,
            index=market_options.index("SPY") if "SPY" in market_options else 0,
            key="cha_fa_market",
            help="Market benchmark for the factor regression.",
        )
    with f2:
        default_factors = [
            f for f in [
                "Momentum", "Value", "Quality", "Leverage",
                "Profitability", "10Yr Sensitivity", "Size", "Growth", "Crowding",
            ] if f in factor_display_names
        ]
        selected_factors = st.multiselect(
            "GS Factors",
            options=factor_display_names,
            default=default_factors,
            key="cha_fa_factors",
            help="Select GS factor indices to include as regressors.",
        )
    with f3:
        p_threshold = st.number_input(
            "P-value threshold",
            min_value=0.001,
            max_value=0.50,
            value=FA_DEFAULT_P_THRESHOLD,
            step=0.01,
            format="%.3f",
            key="cha_fa_p_threshold",
        )

    if not selected_factors:
        st.info("Select at least one GS factor.")
        return

    run_fa = st.button(
        "Run Factor Analysis", type="secondary", use_container_width=True, key="cha_fa_run",
    )

    if run_fa:
        try:
            fa_result = _run_cha_factor_analysis(
                returns, result, factor_data, fa_market, selected_factors, p_threshold,
            )
            st.session_state["cha_fa_result"] = fa_result
        except Exception as e:
            st.error(f"Factor analysis failed: {e}")
            return

    fa_result: FactorAnalyticsResult | None = st.session_state.get("cha_fa_result")
    if fa_result is None:
        st.info("Click **Run Factor Analysis** to see factor exposures.")
        return

    _render_fa_results(fa_result)


def _run_cha_factor_analysis(
    returns: pd.DataFrame,
    result: CustomHedgeResult,
    factor_data: FactorData,
    market_index: str,
    selected_factors: list[str],
    p_threshold: float,
) -> FactorAnalyticsResult:
    """Prepare data and run factor analytics for the custom hedge position."""
    long_ret = result.daily_standalone
    short_ret = result.daily_hedge_basket
    combined_ret = result.daily_hedged

    market_ret = returns[market_index].loc[long_ret.index]
    fr = factor_data.returns[selected_factors]

    # Align each leg with factors and market
    fr_aligned, long_aligned, market_aligned = align_factor_returns(fr, long_ret, market_ret)
    fr_s, short_aligned, _ = align_factor_returns(fr, short_ret, market_ret)
    fr_c, combined_aligned, _ = align_factor_returns(fr, combined_ret, market_ret)

    # Intersect all date sets
    common = fr_aligned.index.intersection(fr_s.index).intersection(fr_c.index)
    fr_aligned = fr_aligned.loc[common]
    long_aligned = long_aligned.loc[common]
    market_aligned = market_aligned.loc[common]
    short_aligned = short_aligned.loc[common]
    combined_aligned = combined_aligned.loc[common]

    if len(common) < 30:
        raise ValueError(f"Only {len(common)} overlapping observations — need at least 30.")

    return run_factor_analytics(
        long_returns=long_aligned,
        market_returns=market_aligned,
        factor_returns=fr_aligned,
        market_index=market_index,
        factor_names=selected_factors,
        dates=common,
        short_returns=short_aligned,
        combined_returns=combined_aligned,
        p_threshold=p_threshold,
    )


def _render_fa_results(result: FactorAnalyticsResult):
    """Render factor analytics results, reusing chart functions from ui/factor_analytics."""
    legs = _active_legs(result)

    # Summary R² cards
    cols = st.columns(2 * len(legs))
    for i, (name, leg) in enumerate(legs):
        cols[2 * i].metric(f"{name} R\u00b2", f"{leg.ols.r_squared:.3f}")
        cols[2 * i + 1].metric(f"{name} Adj R\u00b2", f"{leg.ols.r_squared_adj:.3f}")

    # Regression tables
    if len(legs) == 1:
        name, leg = legs[0]
        _render_regression_table(leg.table, leg.ols, name)
    else:
        tab_widgets = st.tabs([name for name, _ in legs])
        for tab, (name, leg) in zip(tab_widgets, legs):
            with tab:
                _render_regression_table(leg.table, leg.ols, name)

    # Beta heatmap
    st.plotly_chart(_beta_heatmap_chart(result), use_container_width=True, key="cha_fa_heatmap")

    # Return decomposition
    ret_mode = st.radio(
        "View",
        options=["Compounded (growth of $1)", "Additive (cumulative sum)"],
        horizontal=True,
        key="cha_fa_return_mode",
        help="Compounded shows (1+r).cumprod(); additive shows r.cumsum() where factor + idio = total exactly.",
    )
    _render_return_decomposition(legs, additive=ret_mode.startswith("Additive"), key_prefix="cha_fa")

    # Vol decomposition
    vol_window = st.select_slider(
        "Rolling window (days)",
        options=[20, 30, 60, 90, 120, 252],
        value=60,
        key="cha_fa_vol_window",
    )
    _render_vol_decomposition(legs, vol_window, key_prefix="cha_fa")
