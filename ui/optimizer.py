"""Hedge Optimizer tab — controls + results display."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.optimization import HedgeResult, optimize_hedge
from analytics.rolling_optimization import RollingOptResult, rolling_optimize
from config import (
    DEFAULT_CVAR_CONFIDENCE,
    DEFAULT_MAX_GROSS_RATIO,
    DEFAULT_MIN_HEDGE_NAMES,
    DEFAULT_NOTIONAL,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_STRATEGY,
    DEFAULT_WEIGHT_BOUNDS,
    ROLLING_OPT_STEP,
    ROLLING_OPT_WINDOW,
    STRATEGY_OPTIONS,
)
from ui.style import PLOTLY_LAYOUT, render_metrics_table


def _params_hash(params: dict) -> str:
    """Hash sidebar params to detect staleness."""
    key = json.dumps({
        "stocks": sorted(params.get("stock_tickers", [])),
        "factors": sorted(params.get("factor_tickers", [])),
        "start": str(params.get("start_date")),
        "end": str(params.get("end_date")),
        "method": params.get("return_method"),
    }, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def _rolling_corr_chart(result: HedgeResult, window: int) -> go.Figure:
    series = result.rolling_correlation.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=f"{result.target_ticker} vs Hedge Basket",
        line=dict(width=2, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>Corr: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=result.portfolio_correlation, line_dash="dash", line_color="#dc2626", line_width=1,
                  annotation_text=f"Full-period: {result.portfolio_correlation:.3f}",
                  annotation_position="top left")
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{window}-Day Rolling Correlation: {result.target_ticker} vs Hedge Basket",
        xaxis_title="",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
        height=380,
        showlegend=False,
    )
    return fig


def _weight_bar_chart(result: HedgeResult) -> go.Figure:
    weight_sum = float(np.sum(np.abs(result.weights)))
    abs_pcts = np.abs(result.weights) / weight_sum * 100 if weight_sum > 0 else np.abs(result.weights) * 100
    # Filter out near-zero weights
    mask = abs_pcts > 0.01
    tickers = [t for t, m in zip(result.hedge_instruments, mask) if m]
    pcts = abs_pcts[mask]

    fig = go.Figure(
        data=go.Bar(
            x=tickers,
            y=pcts,
            marker_color="#2563eb",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Hedge Weight Allocation",
        xaxis_title="",
        yaxis_title="Weight (%)",
        height=350,
        showlegend=False,
    )
    return fig


HEDGE_UNIVERSE_OPTIONS = ["All Tickers", "Stocks Only", "Factors / Indices Only"]


def render_optimizer_tab(returns: pd.DataFrame, params: dict):
    """Render the Hedge Optimizer tab."""
    all_tickers = [c for c in returns.columns]
    stock_tickers = [t for t in params.get("stock_tickers", []) if t in returns.columns]
    factor_tickers = [t for t in params.get("factor_tickers", []) if t in returns.columns]

    st.caption(
        "Find the optimal hedge portfolio to reduce the volatility of a target position. "
        "Choose a strategy, set weight bounds, and the optimizer will allocate across your hedge universe "
        "to minimize risk. You can also run a walk-forward rolling optimization to see how hedge weights evolve over time."
    )

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Controls")

        target = st.selectbox(
            "Target (long position)",
            options=all_tickers,
            index=0,
            key="opt_target",
            help="The ticker you hold long and want to hedge against.",
        )

        notional = st.number_input(
            f"Notional (default ${DEFAULT_NOTIONAL:,.0f})",
            min_value=1000.0,
            value=DEFAULT_NOTIONAL,
            step=1_000_000.0,
            format="%.0f",
            key="opt_notional",
            help="Dollar value of your long position. Used to calculate hedge notional amounts.",
        )

        max_gross = st.number_input(
            f"Max hedge notional (default ${DEFAULT_NOTIONAL * DEFAULT_MAX_GROSS_RATIO:,.0f})",
            min_value=0.0,
            value=DEFAULT_NOTIONAL * DEFAULT_MAX_GROSS_RATIO,
            step=1_000_000.0,
            format="%.0f",
            key="opt_max_gross",
            help="Maximum total dollar value of the hedge portfolio. "
                 "When set different from notional, the optimizer uses an inequality constraint "
                 "to find the truly optimal hedge size within this budget — including deploying more "
                 "than 100% of notional if needed (e.g., for beta-neutral hedging). "
                 "Set equal to notional for the default fully-allocated hedge (weights sum to -1).",
        )

        hedge_universe = st.selectbox(
            "Hedge universe",
            options=HEDGE_UNIVERSE_OPTIONS,
            index=0,
            key="opt_universe",
            help="Which instruments can be used as hedges.",
        )

        strategy = st.selectbox(
            "Strategy",
            options=STRATEGY_OPTIONS,
            index=STRATEGY_OPTIONS.index(DEFAULT_STRATEGY),
            key="opt_strategy",
            help="Minimum Variance: minimize portfolio volatility. Beta-Neutral: zero out market beta exposure. "
                 "Tail Risk (CVaR): minimize expected losses in the worst scenarios. Risk Parity: allocate inversely to volatility.",
        )

        # Strategy-specific controls
        selected_factors = []
        confidence = DEFAULT_CVAR_CONFIDENCE

        if strategy == "Beta-Neutral":
            available_factors = [f for f in factor_tickers if f in returns.columns and f != target]
            if available_factors:
                selected_factors = st.multiselect(
                    "Neutralize against factors",
                    options=available_factors,
                    default=available_factors,
                    key="opt_factors",
                    help="Select which market factors to neutralize beta against. The optimizer will try to make the hedged portfolio beta-zero to each selected factor.",
                )
            else:
                st.warning("No factor tickers available. Add factors in the sidebar.")

        elif strategy == "Tail Risk (CVaR)":
            confidence = st.slider(
                "Confidence level",
                min_value=0.90,
                max_value=0.99,
                value=DEFAULT_CVAR_CONFIDENCE,
                step=0.01,
                key="opt_confidence",
                help="The probability threshold for CVaR (Conditional Value at Risk). Higher values focus on more extreme tail losses (e.g., 99% looks at the worst 1% of outcomes).",
            )

        st.divider()

        min_names = st.number_input(
            "Min hedge names",
            min_value=0,
            max_value=20,
            value=DEFAULT_MIN_HEDGE_NAMES,
            step=1,
            key="opt_min_names",
            help="Minimum number of instruments in the hedge basket. "
                 "Caps max weight per name to 1/N, which may conflict with tight weight bounds — "
                 "if total capacity (N instruments × effective max weight) < 100%, "
                 "the optimizer cannot fully allocate the hedge. Set 0 to disable.",
        )

        lb = st.number_input("Weight lower bound", value=DEFAULT_WEIGHT_BOUNDS[0], step=0.1, format="%.2f", key="opt_lb",
                             help="Minimum weight per hedge instrument. Negative = short positions (typical for hedging). "
                                  "E.g., -1.0 allows full short allocation to a single name. "
                                  "Tightened automatically by min hedge names (capped to -1/N). "
                                  "Auto-scales when max hedge notional exceeds notional.")
        ub = st.number_input("Weight upper bound", value=DEFAULT_WEIGHT_BOUNDS[1], step=0.1, format="%.2f", key="opt_ub",
                             help="Maximum weight per hedge instrument. Set to 0.0 for short-only hedges, "
                                  "or positive to allow long hedge positions. "
                                  "Auto-scales when max hedge notional exceeds notional.")

        if lb >= ub:
            st.error("Lower bound must be less than upper bound.")
            return

        bounds = (lb, ub)

        if hedge_universe == "Stocks Only":
            hedge_instruments = [t for t in stock_tickers if t != target]
        elif hedge_universe == "Factors / Indices Only":
            hedge_instruments = [t for t in factor_tickers if t != target]
        else:
            hedge_instruments = [t for t in all_tickers if t != target]

        if not hedge_instruments:
            st.error("No hedge instruments available for the selected universe.")
            return

        if min_names > len(hedge_instruments):
            st.error(f"Min hedge names ({min_names}) exceeds available instruments ({len(hedge_instruments)}).")
            return

        # Warn if bounds + min_names make it impossible to fill the hedge budget
        if min_names > 0:
            max_abs_per_name = 1.0 / min_names
            effective_max = min(max_abs_per_name, max(abs(lb), abs(ub)))
            total_capacity = len(hedge_instruments) * effective_max
            if total_capacity < 1.0:
                st.warning(
                    f"Weight bounds (max {effective_max:.2f} per name) with {len(hedge_instruments)} "
                    f"instruments can only reach {total_capacity:.0%} of the hedge budget. "
                    f"The optimizer may not find a feasible solution. Widen the bounds or reduce min hedge names."
                )

        st.caption(f"Hedge universe ({len(hedge_instruments)}): {', '.join(hedge_instruments)}")

        run_opt = st.button("Optimize", type="primary", use_container_width=True, key="opt_run")

    with col_results:
        st.subheader("Results")

        if run_opt:
            try:
                result = optimize_hedge(
                    returns=returns,
                    target=target,
                    hedge_instruments=hedge_instruments,
                    strategy=strategy,
                    notional=notional,
                    bounds=bounds,
                    factors=selected_factors if strategy == "Beta-Neutral" else factor_tickers,
                    confidence=confidence,
                    min_names=min_names,
                    rolling_window=params["window"],
                    max_gross_notional=max_gross,
                )
                st.session_state["hedge_result"] = result
                st.session_state["hedge_params_hash"] = _params_hash(params)
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                return

        result: HedgeResult | None = st.session_state.get("hedge_result")

        if result is None:
            st.info("Configure parameters and click **Optimize** to see results.")
            return

        # Check staleness
        stored_hash = st.session_state.get("hedge_params_hash")
        current_hash = _params_hash(params)
        if stored_hash and stored_hash != current_hash:
            st.warning("Sidebar parameters have changed since last optimization. Results may be stale.")

        # Weights table — show allocation percentages (normalized to 100% of hedge basket)
        weight_sum = float(np.sum(np.abs(result.weights)))
        display_weights = np.abs(result.weights) / weight_sum if weight_sum > 0 else np.abs(result.weights)
        side = "Short" if result.weights[0] <= 0 else "Long"
        weights_df = pd.DataFrame({
            "Ticker": result.hedge_instruments,
            "Weight (%)": np.round(display_weights * 100, 2),
            "Notional ($)": [f"{v:,.0f}" for v in np.abs(result.notionals)],
            "Side": [side] * len(result.hedge_instruments),
        })
        # Filter out zero-weight instruments for cleaner display
        weights_df = weights_df[weights_df["Weight (%)"] > 0.01].reset_index(drop=True)
        # Add totals row
        totals = pd.DataFrame({
            "Ticker": ["**Total**"],
            "Weight (%)": [weights_df["Weight (%)"].sum()],
            "Notional ($)": [f"{result.total_hedge_notional:,.0f}"],
            "Side": [""],
        })
        weights_df = pd.concat([weights_df, totals], ignore_index=True)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

        # Summary metrics
        vol_reduction = (1 - result.hedged_volatility / result.unhedged_volatility) * 100 if result.unhedged_volatility > 0 else 0
        hedge_ratio = result.total_hedge_notional / result.target_notional * 100 if result.target_notional > 0 else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hedged Vol (ann.)", f"{result.hedged_volatility:.1%}")
        m2.metric("Unhedged Vol (ann.)", f"{result.unhedged_volatility:.1%}")
        m3.metric("Vol Reduction", f"{vol_reduction:.1f}%")
        m4.metric("Hedge Ratio", f"{hedge_ratio:.0f}%", help="Total hedge notional / target notional")

        m5, m6 = st.columns(2)
        m5.metric(
            "Correlation to Hedge Basket",
            f"{result.portfolio_correlation:.3f}",
            help="Full-period Pearson correlation between the target and the weighted hedge basket",
        )

        if result.cvar is not None:
            st.metric("CVaR (daily)", f"{result.cvar:.4f}", help=f"At {result.confidence_level:.0%} confidence")

        if result.beta_neutral_feasible is False:
            st.warning(
                "Beta-neutral constraints were **infeasible** with current weight bounds "
                "and hedge universe. The optimizer minimized beta exposure as far as "
                "possible. Consider using Factors/Indices in the hedge universe, or "
                "reducing the number of neutralization factors."
            )

        if result.portfolio_betas:
            # Show unhedged vs hedged betas side by side (multivariate)
            factors = list(result.portfolio_betas.keys())
            beta_df = pd.DataFrame({
                "Factor": factors,
                "Unhedged Beta": [result.unhedged_betas.get(f, None) for f in factors],
                "Hedged Beta": [result.portfolio_betas[f] for f in factors],
            })
            st.caption("Portfolio Betas (multivariate)")
            st.dataframe(beta_df, use_container_width=True, hide_index=True)

        # Charts side by side
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(_weight_bar_chart(result), use_container_width=True)
        with ch2:
            st.plotly_chart(
                _rolling_corr_chart(result, params["window"]),
                use_container_width=True,
            )

        # --- Rolling Optimization Section ---
        st.divider()
        opt_mode = st.radio(
            "Optimization mode",
            options=["Static Optimization", "Rolling Optimization (Walk-Forward)"],
            index=0,
            horizontal=True,
            key="opt_mode",
            help="Static uses a single optimization over the full period. Rolling re-optimizes at regular intervals using a sliding window, showing how optimal hedge weights change over time.",
        )

        if opt_mode == "Rolling Optimization (Walk-Forward)":
            _render_rolling_optimization(returns, params, result)


def _render_rolling_optimization(returns, params, hedge_result):
    """Render rolling optimization controls and results."""
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        ro_window = st.number_input(
            "Rolling window (days)", min_value=30, max_value=504,
            value=ROLLING_OPT_WINDOW, step=10, key="ro_window",
            help="Number of trading days used for each optimization. Larger windows are more stable but less responsive to recent changes.",
        )
    with rc2:
        ro_step = st.number_input(
            "Step (days)", min_value=5, max_value=120,
            value=ROLLING_OPT_STEP, step=5, key="ro_step",
            help="Number of trading days between each re-optimization. Smaller steps give more frequent updates but take longer to compute.",
        )
    with rc3:
        ro_risk_free = st.number_input(
            "Risk-free rate", min_value=0.0, max_value=0.20,
            value=DEFAULT_RISK_FREE_RATE, step=0.01, format="%.2f", key="ro_rf",
            help="Annual risk-free rate used to calculate Sharpe and Sortino ratios.",
        )

    run_ro = st.button("Run Rolling Optimization", type="primary", key="ro_run")

    if run_ro:
        progress = st.progress(0, text="Running walk-forward optimization...")
        try:
            ro_result = rolling_optimize(
                returns=returns,
                target=hedge_result.target_ticker,
                hedge_instruments=hedge_result.hedge_instruments,
                strategy=hedge_result.strategy,
                bounds=hedge_result.bounds,
                static_weights=hedge_result.weights,
                window=ro_window,
                step=ro_step,
                factors=list(hedge_result.portfolio_betas.keys()) if hedge_result.portfolio_betas else None,
                confidence=hedge_result.confidence_level or 0.95,
                min_names=hedge_result.min_names,
                notional=hedge_result.target_notional,
                rolling_window=params["window"],
                risk_free=ro_risk_free,
                progress_callback=lambda p: progress.progress(p, text=f"Optimizing... {p:.0%}"),
                max_gross_notional=hedge_result.max_gross_notional,
            )
            st.session_state["rolling_opt_result"] = ro_result
            progress.empty()
        except Exception as e:
            progress.empty()
            st.error(f"Rolling optimization failed: {e}")
            return

    ro_result: RollingOptResult | None = st.session_state.get("rolling_opt_result")
    if ro_result is None:
        st.info("Set parameters and click **Run Rolling Optimization**.")
        return

    # 3-line cumulative return chart
    fig_cum = go.Figure()
    for name, series, color in [
        ("Unhedged", ro_result.cumulative_unhedged, "#2563eb"),
        ("Static Hedge", ro_result.cumulative_static, "#16a34a"),
        ("Rolling Hedge", ro_result.cumulative_rolling, "#dc2626"),
    ]:
        fig_cum.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name=name,
            line=dict(width=2, color=color),
        ))
    fig_cum.update_layout(**PLOTLY_LAYOUT)
    fig_cum.update_layout(title="Cumulative Returns: Unhedged vs Static vs Rolling", height=400)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Rolling vol chart (3 lines)
    fig_vol = go.Figure()
    for name, series, color in [
        ("Unhedged", ro_result.rolling_vol_unhedged, "#2563eb"),
        ("Static Hedge", ro_result.rolling_vol_static, "#16a34a"),
        ("Rolling Hedge", ro_result.rolling_vol_rolling, "#dc2626"),
    ]:
        fig_vol.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines", name=name,
            line=dict(width=2, color=color),
        ))
    fig_vol.update_layout(**PLOTLY_LAYOUT)
    fig_vol.update_layout(
        title="Rolling Volatility (Annualized)",
        yaxis_title="Volatility",
        yaxis_tickformat=".0%",
        height=350,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Weight evolution chart
    fig_wt = go.Figure()
    for col in ro_result.weight_history.columns:
        fig_wt.add_trace(go.Scatter(
            x=ro_result.weight_history.index,
            y=ro_result.weight_history[col].values,
            mode="lines",
            name=col,
            stackgroup="one",
        ))
    fig_wt.update_layout(**PLOTLY_LAYOUT)
    fig_wt.update_layout(title="Weight Evolution (Walk-Forward)", height=350)
    st.plotly_chart(fig_wt, use_container_width=True)

    # Turnover bars
    if len(ro_result.turnover) > 0:
        fig_to = go.Figure(data=go.Bar(
            x=ro_result.turnover.index,
            y=ro_result.turnover.values,
            marker_color="#7c3aed",
        ))
        fig_to.update_layout(**PLOTLY_LAYOUT)
        fig_to.update_layout(title="Turnover Between Optimizations", height=300, yaxis_title="Turnover")
        st.plotly_chart(fig_to, use_container_width=True)

    # 3-column metrics table
    st.subheader("Rolling Optimization Metrics")
    pct_rows = {"Total Return", "Ann. Return", "Ann. Volatility", "Max Drawdown", "Tracking Error"}
    int_rows = {"Max DD Duration (days)"}
    fmt = ro_result.metrics.copy()
    for col in fmt.columns:
        fmt[col] = fmt.index.map(
            lambda idx, c=col: f"{ro_result.metrics.loc[idx, c]:.2%}"
            if idx in pct_rows
            else (f"{int(ro_result.metrics.loc[idx, c])}" if idx in int_rows
                  else f"{ro_result.metrics.loc[idx, c]:.2f}")
        )
    render_metrics_table(fmt)

    # Weight stability table
    st.caption("Weight Stability (standard deviation over time)")
    fmt_stab = ro_result.weight_stability.copy()
    fmt_stab["Std Dev"] = fmt_stab["Std Dev"].map("{:.4f}".format)
    fmt_stab["Mean"] = fmt_stab["Mean"].map("{:.4f}".format)
    st.dataframe(fmt_stab, use_container_width=True)
