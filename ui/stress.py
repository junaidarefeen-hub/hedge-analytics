"""Stress Test tab — historical scenario replay and custom shock analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.stress import ScenarioResult, StressTestResult, run_stress_test
from config import HISTORICAL_SCENARIOS, STRESS_CUSTOM_SHOCK_RANGE
from ui.optimizer import _params_hash
from ui.style import PLOTLY_LAYOUT


def _pnl_comparison_chart(result: StressTestResult) -> go.Figure:
    names = [s.name for s in result.scenarios]
    unhedged = [s.unhedged_pnl for s in result.scenarios]
    hedged = [s.hedged_pnl for s in result.scenarios]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=unhedged, name="Unhedged",
        marker_color="#2563eb", hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=names, y=hedged, name="Hedged",
        marker_color="#16a34a", hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        barmode="group",
        title="P&L Impact by Scenario",
        xaxis_title="",
        yaxis_title="P&L ($)",
        height=420,
    )
    return fig


def _hedge_benefit_chart(result: StressTestResult) -> go.Figure:
    names = [s.name for s in result.scenarios]
    benefits = [s.hedge_benefit for s in result.scenarios]
    colors = ["#16a34a" if b >= 0 else "#dc2626" for b in benefits]

    fig = go.Figure(go.Bar(
        x=names, y=benefits, marker_color=colors,
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Hedge Benefit by Scenario ($)",
        xaxis_title="",
        yaxis_title="Benefit ($)",
        height=380,
        showlegend=False,
    )
    return fig


def _scenario_drawdown_chart(scenario: ScenarioResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scenario.daily_cumulative_unhedged.index,
        y=scenario.daily_cumulative_unhedged.values,
        mode="lines", name="Unhedged",
        line=dict(width=2, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=scenario.daily_cumulative_hedged.index,
        y=scenario.daily_cumulative_hedged.values,
        mode="lines", name="Hedged",
        line=dict(width=2, color="#16a34a"),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{scenario.name}",
        xaxis_title="",
        yaxis_title="Cumulative Return",
        height=320,
    )
    return fig


def render_stress_tab(returns: pd.DataFrame, params: dict):
    """Render the Stress Test tab."""
    hedge_result = st.session_state.get("hedge_result")

    if hedge_result is None:
        st.info("Run the **Hedge Optimizer** or **Strategy Compare** first, then come back here for stress testing.")
        return

    # Staleness check
    stored_hash = st.session_state.get("hedge_params_hash")
    current_hash = _params_hash(params)
    if stored_hash and stored_hash != current_hash:
        st.warning("Sidebar parameters have changed since last optimization. Stress test may use stale hedge weights.")

    st.caption(
        "See how your hedge performs under extreme market conditions. Replay historical crises (e.g., COVID crash, GFC) "
        "or define custom shock scenarios to estimate the P&L impact on your hedged vs unhedged portfolio."
    )

    st.caption(f"Strategy: **{hedge_result.strategy}** | Target: **{hedge_result.target_ticker}**")

    col_ctrl, col_results = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Controls")

        mode = st.radio("Mode", ["Historical", "Custom", "Both"], horizontal=True, key="stress_mode",
                        help="Historical: replay actual market crises from your data. Custom: define your own shock scenarios. Both: run all together.")

        selected_scenarios = []
        custom_scenarios_list = []

        # Historical scenarios
        if mode in ("Historical", "Both"):
            st.caption("Historical Scenarios")
            data_start = returns.index.min()
            data_end = returns.index.max()
            scenario_names = []
            for s in HISTORICAL_SCENARIOS:
                s_start = pd.Timestamp(s["start"])
                s_end = pd.Timestamp(s["end"])
                in_range = s_start >= data_start and s_end <= data_end
                label = f"{s['name']} {'' if in_range else '(no data)'}"
                scenario_names.append(label)

            # Default to scenarios within data range
            defaults = [
                scenario_names[i] for i, s in enumerate(HISTORICAL_SCENARIOS)
                if pd.Timestamp(s["start"]) >= data_start and pd.Timestamp(s["end"]) <= data_end
            ]
            chosen = st.multiselect(
                "Select scenarios", options=scenario_names, default=defaults, key="stress_scenarios",
            )
            selected_scenarios = [
                HISTORICAL_SCENARIOS[scenario_names.index(c)] for c in chosen
            ]

            if not defaults:
                st.info(
                    f"Data covers {data_start.date()} to {data_end.date()}. "
                    "Expand sidebar date range to include more historical scenarios."
                )

        # Custom scenarios
        if mode in ("Custom", "Both"):
            st.caption("Custom Scenarios")
            all_tickers = [hedge_result.target_ticker] + hedge_result.hedge_instruments

            if "stress_custom_count" not in st.session_state:
                st.session_state["stress_custom_count"] = 1

            n_custom = st.session_state["stress_custom_count"]

            for idx in range(n_custom):
                with st.expander(f"Custom Scenario {idx + 1}", expanded=(idx == 0)):
                    name = st.text_input("Name", value=f"Custom {idx + 1}", key=f"stress_cname_{idx}")

                    # Preset buttons
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        if st.button("Market Crash", key=f"stress_preset_crash_{idx}", use_container_width=True):
                            for t in all_tickers:
                                st.session_state[f"stress_shock_{idx}_{t}"] = -20.0
                    with p2:
                        if st.button("Mild Selloff", key=f"stress_preset_mild_{idx}", use_container_width=True):
                            for t in all_tickers:
                                st.session_state[f"stress_shock_{idx}_{t}"] = -10.0
                    with p3:
                        if st.button("Reset", key=f"stress_preset_reset_{idx}", use_container_width=True):
                            for t in all_tickers:
                                st.session_state[f"stress_shock_{idx}_{t}"] = 0.0

                    shocks = {}
                    for t in all_tickers:
                        default_val = st.session_state.get(f"stress_shock_{idx}_{t}", 0.0)
                        shock = st.number_input(
                            f"{t} shock (%)",
                            min_value=STRESS_CUSTOM_SHOCK_RANGE[0],
                            max_value=STRESS_CUSTOM_SHOCK_RANGE[1],
                            value=default_val,
                            step=1.0,
                            format="%.1f",
                            key=f"stress_shock_{idx}_{t}",
                        )
                        shocks[t] = shock

                    custom_scenarios_list.append({"name": name, "shocks": shocks})

            if st.button("+ Add Custom Scenario", key="stress_add_custom"):
                st.session_state["stress_custom_count"] = n_custom + 1
                st.rerun()

        st.divider()

        notional = st.number_input(
            "Portfolio notional ($)", min_value=1000.0,
            value=hedge_result.target_notional, step=10000.0, format="%.0f", key="stress_notional",
            help="Dollar value of the portfolio used to calculate P&L impact in dollar terms.",
        )

        run_stress = st.button("Run Stress Test", type="primary", use_container_width=True, key="stress_run")

    with col_results:
        st.subheader("Stress Test Results")

        if run_stress:
            if not selected_scenarios and not custom_scenarios_list:
                st.error("Select at least one scenario to run.")
                return
            with st.spinner("Running stress scenarios..."):
                try:
                    result = run_stress_test(
                        returns=returns,
                        target=hedge_result.target_ticker,
                        hedge_instruments=hedge_result.hedge_instruments,
                        weights=hedge_result.weights,
                        notional=notional,
                        strategy=hedge_result.strategy,
                        selected_scenarios=selected_scenarios,
                        custom_scenarios=custom_scenarios_list,
                    )
                    st.session_state["stress_result"] = result
                except Exception as e:
                    st.error(f"Stress test failed: {e}")
                    return

        result: StressTestResult | None = st.session_state.get("stress_result")

        if result is None:
            st.info("Configure scenarios and click **Run Stress Test** to see results.")
            return

        # Skipped scenarios
        if result.skipped:
            for name, reason in result.skipped.items():
                st.warning(f"Skipped **{name}**: {reason}")

        # Summary table
        st.caption("Summary")
        fmt_df = result.summary_df.copy()
        pct_cols = ["Unhedged Return", "Hedged Return"]
        dollar_cols = ["Unhedged P&L ($)", "Hedged P&L ($)", "Hedge Benefit ($)"]
        for col in pct_cols:
            fmt_df[col] = result.summary_df[col].apply(lambda v: f"{v:.2%}")
        for col in dollar_cols:
            fmt_df[col] = result.summary_df[col].apply(lambda v: f"${v:,.0f}")
        fmt_df["Hedge Benefit (pp)"] = result.summary_df["Hedge Benefit (pp)"].apply(lambda v: f"{v:+.1f}pp")
        st.dataframe(fmt_df, use_container_width=True, hide_index=True)

        # Charts
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(_pnl_comparison_chart(result), use_container_width=True)
        with ch2:
            st.plotly_chart(_hedge_benefit_chart(result), use_container_width=True)

        # Historical scenario drawdown charts
        historical = [s for s in result.scenarios if not s.is_custom and s.daily_cumulative_hedged is not None]
        if historical:
            st.caption("Historical Scenario Drawdowns")
            # Show charts in pairs
            for i in range(0, len(historical), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(historical):
                        with col:
                            st.plotly_chart(
                                _scenario_drawdown_chart(historical[i + j]),
                                use_container_width=True,
                            )

        # Instrument P&L breakdown
        with st.expander("Instrument P&L Breakdown"):
            breakdown_rows = []
            for s in result.scenarios:
                row = {"Scenario": s.name}
                for inst, pnl in s.instrument_pnl.items():
                    row[inst] = pnl
                breakdown_rows.append(row)
            breakdown_df = pd.DataFrame(breakdown_rows).set_index("Scenario")
            # Format
            fmt_breakdown = breakdown_df.map(lambda v: f"${v:,.0f}")
            st.dataframe(fmt_breakdown, use_container_width=True)
