"""Monte Carlo tab — forward-looking simulation of hedged vs unhedged portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.montecarlo import MonteCarloResult, run_monte_carlo
from config import (
    MC_DEFAULT_HORIZON,
    MC_DEFAULT_NUM_SIMS,
    MC_DEFAULT_SEED,
    MC_HORIZON_OPTIONS,
    MC_NUM_SIMS_OPTIONS,
    MC_SPAGHETTI_PATHS,
)
from ui.optimizer import _params_hash
from ui.style import PLOTLY_LAYOUT


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


def _distribution_chart(result: MonteCarloResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=result.unhedged_final,
        name="Unhedged",
        opacity=0.6,
        marker_color="#2563eb",
        nbinsx=80,
        hovertemplate="Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=result.hedged_final,
        name="Hedged",
        opacity=0.6,
        marker_color="#16a34a",
        nbinsx=80,
        hovertemplate="Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=result.initial_value, line_dash="dash", line_color="#dc2626", line_width=1,
        annotation_text="Initial", annotation_position="top left",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution of Portfolio Value at Day {result.horizon}",
        xaxis_title="Portfolio Value ($)",
        yaxis_title="Frequency",
        height=420,
    )
    return fig


def _fan_chart(result: MonteCarloResult, portfolio: str = "hedged") -> go.Figure:
    bands = result.hedged_bands if portfolio == "hedged" else result.unhedged_bands
    days = np.arange(result.horizon + 1)
    color = "#16a34a" if portfolio == "hedged" else "#2563eb"
    label = "Hedged" if portfolio == "hedged" else "Unhedged"
    rgb = _hex_to_rgb(color)

    fig = go.Figure()

    sorted_pctiles = sorted(bands.keys())
    pairs = [(sorted_pctiles[i], sorted_pctiles[-(i + 1)]) for i in range(len(sorted_pctiles) // 2)]
    opacities = [0.15, 0.30]

    for idx, (lo, hi) in enumerate(pairs):
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([bands[hi], bands[lo][::-1]]),
            fill="toself",
            fillcolor=f"rgba({rgb},{opacities[idx]})",
            line=dict(width=0),
            name=f"{lo}th–{hi}th pctile",
            hoverinfo="skip",
        ))

    if 50 in bands:
        fig.add_trace(go.Scatter(
            x=days, y=bands[50],
            mode="lines", name="Median",
            line=dict(width=2.5, color=color),
            hovertemplate="Day %{x}<br>$%{y:,.0f}<extra></extra>",
        ))

    fig.add_hline(y=result.initial_value, line_dash="dot", line_color="#cbd5e1", line_width=1)

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{label} — Percentile Bands ({result.num_sims:,} simulations)",
        xaxis_title="Day",
        yaxis_title="Portfolio Value ($)",
        height=400,
    )
    return fig


def _spaghetti_chart(result: MonteCarloResult, n_paths: int) -> go.Figure:
    fig = go.Figure()
    days = np.arange(result.horizon + 1)
    rng = np.random.default_rng(0)
    indices = rng.choice(result.num_sims, size=min(n_paths, result.num_sims), replace=False)

    for i in indices:
        fig.add_trace(go.Scatter(
            x=days, y=result.hedged_paths[i],
            mode="lines", line=dict(width=0.5, color="rgba(22,163,74,0.15)"),
            showlegend=False, hoverinfo="skip",
        ))

    if 50 in result.hedged_bands:
        fig.add_trace(go.Scatter(
            x=days, y=result.hedged_bands[50],
            mode="lines", name="Median",
            line=dict(width=2.5, color="#16a34a"),
            hovertemplate="Day %{x}<br>$%{y:,.0f}<extra></extra>",
        ))

    fig.add_hline(
        y=result.initial_value, line_dash="dot", line_color="#dc2626", line_width=1,
        annotation_text="Initial Value",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"Simulated Hedged Paths ({n_paths} of {result.num_sims:,})",
        xaxis_title="Day",
        yaxis_title="Portfolio Value ($)",
        height=400,
    )
    return fig


def render_montecarlo_tab(returns: pd.DataFrame, params: dict):
    """Render the Monte Carlo simulation tab."""
    hedge_result = st.session_state.get("hedge_result")

    if hedge_result is None:
        st.info("Run the **Hedge Optimizer** or **Strategy Compare** first, then come back here to simulate.")
        return

    # Staleness check
    stored_hash = st.session_state.get("hedge_params_hash")
    current_hash = _params_hash(params)
    if stored_hash and stored_hash != current_hash:
        st.warning("Sidebar parameters have changed since last optimization. Simulation may use stale hedge weights.")

    st.caption(
        "Simulate thousands of possible future outcomes for your hedged and unhedged portfolios using Monte Carlo methods. "
        "Returns are randomly sampled from the historical distribution to estimate the range of possible gains and losses."
    )

    st.caption(f"Strategy: **{hedge_result.strategy}** | Target: **{hedge_result.target_ticker}**")

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Controls")

        horizon = st.selectbox(
            "Horizon (days)", options=MC_HORIZON_OPTIONS,
            index=MC_HORIZON_OPTIONS.index(MC_DEFAULT_HORIZON), key="mc_horizon",
            help="Number of trading days to simulate forward. 252 days ≈ 1 year.",
        )

        num_sims = st.selectbox(
            "Number of simulations", options=MC_NUM_SIMS_OPTIONS,
            index=MC_NUM_SIMS_OPTIONS.index(MC_DEFAULT_NUM_SIMS), key="mc_num_sims",
            help="More simulations give smoother results but take longer to compute.",
        )

        initial_value = st.number_input(
            "Initial portfolio value ($)", min_value=1000.0,
            value=hedge_result.target_notional, step=10000.0, format="%.0f", key="mc_initial",
            help="Starting dollar value of the portfolio for the simulation.",
        )

        conf_options = [0.90, 0.95, 0.99]
        conf_levels = st.multiselect(
            "VaR confidence levels", options=conf_options, default=[0.95, 0.99], key="mc_conf",
            help="Confidence levels for Value at Risk (VaR) and Conditional VaR calculations. E.g., 95% VaR is the loss exceeded only 5% of the time.",
        )
        if not conf_levels:
            conf_levels = [0.95]

        use_seed = st.checkbox("Reproducible (fixed seed)", value=True, key="mc_seed_toggle",
                               help="When checked, simulations use a fixed random seed so results are the same each time you run. Uncheck for different random outcomes each run.")

        run_mc = st.button("Run Simulation", type="primary", use_container_width=True, key="mc_run")

    with col_results:
        st.subheader("Monte Carlo Simulation")

        if run_mc:
            with st.spinner(f"Running {num_sims:,} simulations over {horizon} days..."):
                try:
                    result = run_monte_carlo(
                        returns=returns,
                        target=hedge_result.target_ticker,
                        hedge_instruments=hedge_result.hedge_instruments,
                        weights=hedge_result.weights,
                        strategy=hedge_result.strategy,
                        horizon=horizon,
                        num_sims=num_sims,
                        initial_value=initial_value,
                        confidence_levels=sorted(conf_levels),
                        seed=MC_DEFAULT_SEED if use_seed else None,
                    )
                    st.session_state["mc_result"] = result
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    return

        result: MonteCarloResult | None = st.session_state.get("mc_result")

        if result is None:
            st.info("Configure parameters and click **Run Simulation** to see results.")
            return

        # Metrics table
        st.caption("Risk Metrics")
        pct_rows = {"Mean Return", "Median Return", "Best Case", "Worst Case"}
        var_rows = {r for r in result.metrics.index if r.startswith("VaR") or r.startswith("CVaR")}
        prob_rows = {r for r in result.metrics.index if r.startswith("P(loss")}
        fmt_metrics = result.metrics.copy()
        for col in fmt_metrics.columns:
            fmt_metrics[col] = result.metrics.index.map(
                lambda idx, c=col: (
                    f"{result.metrics.loc[idx, c]:.2%}" if idx in pct_rows | var_rows | prob_rows
                    else f"{result.metrics.loc[idx, c]:.4f}"
                )
            )
        st.dataframe(fmt_metrics, use_container_width=True)

        # Distribution histogram
        st.plotly_chart(_distribution_chart(result), use_container_width=True)

        # Fan charts side by side
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(_fan_chart(result, "hedged"), use_container_width=True)
        with ch2:
            st.plotly_chart(_fan_chart(result, "unhedged"), use_container_width=True)

        # Spaghetti plot in expander
        with st.expander("Individual Simulated Paths", expanded=False):
            st.plotly_chart(
                _spaghetti_chart(result, MC_SPAGHETTI_PATHS), use_container_width=True,
            )
