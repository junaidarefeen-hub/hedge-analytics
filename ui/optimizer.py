"""Hedge Optimizer tab — controls + results display."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.optimization import HedgeResult, optimize_hedge
from config import (
    DEFAULT_CVAR_CONFIDENCE,
    DEFAULT_MIN_HEDGE_NAMES,
    DEFAULT_NOTIONAL,
    DEFAULT_STRATEGY,
    DEFAULT_WEIGHT_BOUNDS,
    STRATEGY_OPTIONS,
)
from ui.style import PLOTLY_LAYOUT


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
    abs_pcts = np.abs(result.weights) * 100
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

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Controls")

        target = st.selectbox(
            "Target (long position)",
            options=all_tickers,
            index=0,
            key="opt_target",
        )

        notional = st.number_input(
            "Notional ($)",
            min_value=1000.0,
            value=DEFAULT_NOTIONAL,
            step=10000.0,
            format="%.0f",
            key="opt_notional",
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
                 "Caps max weight per name to 1/N. Set 0 to disable.",
        )

        lb = st.number_input("Weight lower bound", value=DEFAULT_WEIGHT_BOUNDS[0], step=0.1, format="%.2f", key="opt_lb")
        ub = st.number_input("Weight upper bound", value=DEFAULT_WEIGHT_BOUNDS[1], step=0.1, format="%.2f", key="opt_ub")

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

        # Weights table — show absolute percentages for readability
        abs_weights = np.abs(result.weights)
        side = "Short" if result.weights[0] <= 0 else "Long"
        weights_df = pd.DataFrame({
            "Ticker": result.hedge_instruments,
            "Weight (%)": np.round(abs_weights * 100, 2),
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
