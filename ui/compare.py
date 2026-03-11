"""Strategy Compare tab — run all strategies, rank, recommend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.compare import CompareResult, compare_strategies
from config import (
    DEFAULT_CVAR_CONFIDENCE,
    DEFAULT_MAX_GROSS_RATIO,
    DEFAULT_MIN_HEDGE_NAMES,
    DEFAULT_NOTIONAL,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_ROLLING_VOL_WINDOW,
    DEFAULT_STRATEGY,
    DEFAULT_WEIGHT_BOUNDS,
)
from ui.optimizer import HEDGE_UNIVERSE_OPTIONS, _params_hash
from ui.style import PLOTLY_LAYOUT, render_metrics_table
from ui.weight_helpers import handle_normalize, render_weight_inputs, sync_weights, weights_array
from utils.basket import basket_display_name, exclude_basket_constituents, inject_basket_column


def _metrics_table(result: CompareResult) -> pd.DataFrame:
    """Format the metrics DataFrame for display."""
    df = result.metrics_df.copy()
    pct_rows = {"Total Return", "Ann. Return", "Ann. Volatility", "Max Drawdown", "Vol Reduction", "Tracking Error"}
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = df.index.map(
            lambda idx, c=col: f"{df.loc[idx, c]:.2%}" if idx in pct_rows else f"{df.loc[idx, c]:.2f}"
        )
    return formatted


def _radar_chart(result: CompareResult) -> go.Figure:
    """Radar chart: each strategy as a polygon, metrics normalized 0-1."""
    strategy_cols = [c.strategy for c in result.comparisons]
    metrics_df = result.metrics_df[strategy_cols]

    # Metrics where lower is better (Max Drawdown is negative, so higher = less severe = better; it uses the default path)
    _LOWER_BETTER = {"Ann. Volatility", "Tracking Error"}

    # Normalize: for each metric, scale so 1 = best across strategies
    normalized = metrics_df.copy()
    for metric in normalized.index:
        vals = normalized.loc[metric].astype(float)
        vmin, vmax = vals.min(), vals.max()

        if vmax == vmin:
            normalized.loc[metric] = 1.0
        elif metric in _LOWER_BETTER:
            # Lower is better: lowest value → 1.0, highest → 0.0
            normalized.loc[metric] = (vmax - vals) / (vmax - vmin)
        else:
            # Higher is better: highest value → 1.0, lowest → 0.0
            normalized.loc[metric] = (vals - vmin) / (vmax - vmin)

    categories = list(normalized.index)
    fig = go.Figure()

    for strategy in strategy_cols:
        values = normalized[strategy].tolist()
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=strategy,
            opacity=0.6,
        ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Strategy Profile (normalized, 1 = best)",
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def _bar_charts(result: CompareResult) -> go.Figure:
    """2x2 bar chart grid for key metrics."""
    from plotly.subplots import make_subplots

    strategy_cols = [c.strategy for c in result.comparisons]
    # Short labels for x-axis
    short_labels = [s.replace("Tail Risk (CVaR)", "CVaR").replace("Minimum Variance", "Min Var").replace("Beta-Neutral", "Beta-Neut") for s in strategy_cols]

    chart_metrics = [
        ("Sharpe Ratio", "", ".2f"),
        ("Vol Reduction", "%", ".1%"),
        ("Max Drawdown", "%", ".1%"),
        ("Sortino Ratio", "", ".2f"),
    ]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[m[0] for m in chart_metrics])

    for i, (metric, suffix, fmt) in enumerate(chart_metrics):
        row, col = divmod(i, 2)
        values = [float(result.metrics_df.loc[metric, s]) for s in strategy_cols]
        # Highlight recommended
        colors = ["#2563eb" if s == result.recommended_strategy else "#94a3b8" for s in strategy_cols]
        hover = [f"{s}: {v:{fmt}}" for s, v in zip(strategy_cols, values)]
        fig.add_trace(
            go.Bar(x=short_labels, y=values, marker_color=colors, hovertext=hover, hoverinfo="text", showlegend=False),
            row=row + 1, col=col + 1,
        )

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(height=500, title_text="Key Metrics by Strategy", showlegend=False)
    return fig


def render_compare_tab(returns: pd.DataFrame, params: dict):
    """Render the Strategy Compare tab."""
    all_tickers = list(returns.columns)
    stock_tickers = [t for t in params.get("stock_tickers", []) if t in returns.columns]
    factor_tickers = [t for t in params.get("factor_tickers", []) if t in returns.columns]

    st.caption(
        "Run all four hedging strategies side-by-side and compare their performance. "
        "The app ranks each strategy across multiple metrics and recommends the best overall approach. "
        "You can then load any strategy into the Optimizer and Backtest tabs for deeper analysis."
    )

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Controls")

        long_tickers = st.multiselect(
            "Target (long position)", options=all_tickers,
            default=[all_tickers[0]] if all_tickers else [],
            key="cmp_target_tickers",
            help="Select one or more tickers for your long position. Multiple tickers create a weighted basket.",
        )

        if not long_tickers:
            st.warning("Select at least one long ticker.")
            return

        # Weight inputs for multi-ticker basket
        sync_weights("cmp_lw", long_tickers)
        handle_normalize("cmp_lw", long_tickers)
        if len(long_tickers) > 1:
            long_weights_raw = render_weight_inputs("cmp_lw", long_tickers, "cmp_norm_long")
            long_w_arr = weights_array(long_weights_raw, long_tickers)
        else:
            long_weights_raw = None
            long_w_arr = np.array([1.0])

        notional = st.number_input(
            f"Notional (default ${DEFAULT_NOTIONAL:,.0f})", min_value=1000.0, value=DEFAULT_NOTIONAL, step=1_000_000.0, format="%.0f", key="cmp_notional",
            help="Dollar value of your long position.",
        )

        max_gross = st.number_input(
            f"Max hedge notional (default ${DEFAULT_NOTIONAL * DEFAULT_MAX_GROSS_RATIO:,.0f})", min_value=0.0, value=DEFAULT_NOTIONAL * DEFAULT_MAX_GROSS_RATIO,
            step=1_000_000.0, format="%.0f", key="cmp_max_gross",
            help="Maximum total dollar value of the hedge portfolio. "
                 "When set different from notional, the optimizer uses an inequality constraint "
                 "to find the truly optimal hedge size within this budget — including deploying more "
                 "than 100% of notional if needed (e.g., for beta-neutral hedging). "
                 "Set equal to notional for the default fully-allocated hedge (weights sum to -1).",
        )

        hedge_universe = st.selectbox(
            "Hedge universe", options=HEDGE_UNIVERSE_OPTIONS, index=0, key="cmp_universe",
            help="Which instruments can be used as hedges.",
        )

        min_names = st.number_input(
            "Min hedge names", min_value=0, max_value=20, value=DEFAULT_MIN_HEDGE_NAMES, step=1, key="cmp_min_names",
            help="Minimum number of instruments in the hedge basket. "
                 "Caps max weight per name to 1/N, which may conflict with tight weight bounds — "
                 "if total capacity (N instruments × effective max weight) < 100%, "
                 "the optimizer cannot fully allocate the hedge. Set 0 to disable.",
        )

        lb = st.number_input("Weight lower bound", value=DEFAULT_WEIGHT_BOUNDS[0], step=0.1, format="%.2f", key="cmp_lb",
                             help="Minimum weight per hedge instrument. Negative = short positions (typical for hedging). "
                                  "Tightened automatically by min hedge names (capped to -1/N). "
                                  "Auto-scales when max hedge notional exceeds notional.")
        ub = st.number_input("Weight upper bound", value=DEFAULT_WEIGHT_BOUNDS[1], step=0.1, format="%.2f", key="cmp_ub",
                             help="Maximum weight per hedge instrument. Set to 0.0 for short-only hedges, "
                                  "or positive to allow long hedge positions. "
                                  "Auto-scales when max hedge notional exceeds notional.")

        if lb >= ub:
            st.error("Lower bound must be less than upper bound.")
            return

        bounds = (lb, ub)

        st.divider()
        st.caption("Beta-Neutral settings")
        long_set = set(long_tickers)
        available_factors = [f for f in factor_tickers if f in returns.columns and f not in long_set]
        selected_factors = st.multiselect(
            "Neutralize against factors", options=available_factors, default=available_factors, key="cmp_factors",
            help="Factors used by the Beta-Neutral strategy to neutralize market exposure.",
        ) if available_factors else []

        st.caption("CVaR settings")
        confidence = st.slider(
            "Confidence level", min_value=0.90, max_value=0.99, value=DEFAULT_CVAR_CONFIDENCE, step=0.01, key="cmp_conf",
            help="Confidence level for the Tail Risk (CVaR) strategy. Higher values target more extreme tail losses.",
        )

        st.divider()
        st.caption("Backtest settings")

        min_date = returns.index.min().date()
        max_date = returns.index.max().date()
        c1, c2 = st.columns(2)
        with c1:
            bt_start = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date, key="cmp_start")
        with c2:
            bt_end = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date, key="cmp_end")

        risk_free = st.number_input(
            "Risk-free rate", min_value=0.0, max_value=0.20, value=DEFAULT_RISK_FREE_RATE, step=0.01, format="%.2f", key="cmp_rf",
            help="Annual risk-free rate for Sharpe and Sortino ratio calculations.",
        )

        roll_window = st.number_input(
            "Rolling vol window", min_value=10, max_value=252, value=DEFAULT_ROLLING_VOL_WINDOW, step=10, key="cmp_roll",
            help="Window for computing rolling volatility in backtest charts.",
        )

        # Build hedge instruments
        if hedge_universe == "Stocks Only":
            hedge_instruments = exclude_basket_constituents(stock_tickers, long_tickers)
        elif hedge_universe == "Factors / Indices Only":
            hedge_instruments = exclude_basket_constituents(factor_tickers, long_tickers)
        else:
            hedge_instruments = exclude_basket_constituents(all_tickers, long_tickers)

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

        run_compare = st.button("Compare All Strategies", type="primary", use_container_width=True, key="cmp_run")

    with col_results:
        st.subheader("Strategy Comparison")

        if run_compare:
            with st.spinner("Running all strategies and backtesting..."):
                try:
                    # Inject basket column for multi-ticker long positions
                    if len(long_tickers) > 1:
                        augmented_returns, target = inject_basket_column(returns, long_tickers, long_w_arr)
                    else:
                        augmented_returns, target = returns, long_tickers[0]

                    result = compare_strategies(
                        returns=augmented_returns,
                        target=target,
                        hedge_instruments=hedge_instruments,
                        notional=notional,
                        bounds=bounds,
                        factors=selected_factors if selected_factors else factor_tickers,
                        confidence=confidence,
                        min_names=min_names,
                        rolling_window=params["window"],
                        risk_free=risk_free,
                        start_date=pd.Timestamp(bt_start),
                        end_date=pd.Timestamp(bt_end),
                        max_gross_notional=max_gross,
                    )
                    # Stamp basket metadata on each hedge result
                    for comp in result.comparisons:
                        comp.hedge_result.target_tickers = long_tickers
                        comp.hedge_result.target_weights = long_w_arr
                    st.session_state["compare_result"] = result
                    st.session_state["compare_params_hash"] = _params_hash(params)
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    return

        result: CompareResult | None = st.session_state.get("compare_result")

        if result is None:
            st.info("Configure parameters and click **Compare All Strategies** to see results.")
            return

        # Staleness check
        stored_hash = st.session_state.get("compare_params_hash")
        current_hash = _params_hash(params)
        if stored_hash and stored_hash != current_hash:
            st.warning("Sidebar parameters have changed since last comparison. Results may be stale.")

        # Failed strategies
        if result.failed_strategies:
            for strat, reason in result.failed_strategies.items():
                st.warning(f"{strat} failed: {reason}")

        # Beta-neutral infeasibility warning
        for comp in result.comparisons:
            if comp.strategy == "Beta-Neutral" and comp.hedge_result.beta_neutral_feasible is False:
                st.warning(
                    "Beta-neutral constraints were **infeasible** with current weight bounds "
                    "and hedge universe. The optimizer minimized beta exposure as far as "
                    "possible. Consider using Factors/Indices in the hedge universe, or "
                    "reducing the number of neutralization factors."
                )
                break

        # Recommendation
        st.success(f"**Recommended: {result.recommended_strategy}** — best composite rank across all metrics")

        # Metrics table
        st.caption("Performance Metrics (Hedged portfolio for each strategy)")
        formatted = _metrics_table(result)
        render_metrics_table(formatted)

        # Ranking table
        rank_df = result.ranking_df.copy()
        # Format composite row differently
        display_rank = rank_df.copy()
        for col in display_rank.columns:
            display_rank[col] = rank_df.index.map(
                lambda idx, c=col: f"{rank_df.loc[idx, c]:.1f}" if idx == "Composite" else f"{int(rank_df.loc[idx, c])}"
            )
        st.caption("Ranking (1 = best per metric)")
        st.dataframe(display_rank, use_container_width=True)

        # Charts
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(_radar_chart(result), use_container_width=True)
        with ch2:
            st.plotly_chart(_bar_charts(result), use_container_width=True)

        # Use strategy buttons
        st.divider()
        st.caption("Load a strategy into the Hedge Optimizer & Backtest tabs:")
        btn_cols = st.columns(len(result.comparisons))
        for i, comp in enumerate(result.comparisons):
            label = f"{'** ' if comp.strategy == result.recommended_strategy else ''}{comp.strategy}{'**' if comp.strategy == result.recommended_strategy else ''}"
            with btn_cols[i]:
                if st.button(
                    comp.strategy,
                    key=f"cmp_use_{i}",
                    use_container_width=True,
                    type="primary" if comp.strategy == result.recommended_strategy else "secondary",
                ):
                    st.session_state["hedge_result"] = comp.hedge_result
                    st.session_state["backtest_result"] = comp.backtest_result
                    st.session_state["hedge_params_hash"] = _params_hash(params)
                    st.session_state["bt_roll_window"] = roll_window
                    st.success(f"Loaded **{comp.strategy}**. Switch to the Hedge Optimizer or Backtest tab for details.")
