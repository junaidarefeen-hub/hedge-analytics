"""Factor Monitor tab: factor performance, trends, and stock-level factor exposures."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from analytics.factor_monitor import compute_factor_monitor
from data.market_monitor.constituents import get_name_map, get_sector_map
from ui.style import PLOTLY_LAYOUT


def _factor_cumulative_chart(factor_returns, title: str = "Factor Cumulative Returns") -> go.Figure:
    """Line chart of cumulative factor returns."""
    if factor_returns.empty:
        return go.Figure()

    cumulative = (1 + factor_returns).cumprod() - 1

    fig = go.Figure()
    for col in cumulative.columns:
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[col] * 100,
            name=col,
            mode="lines",
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>Return: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        yaxis_title="Cumulative Return (%)",
        height=400,
        hovermode="x unified",
    )
    return fig


def _factor_trend_table_html(trends) -> str:
    """Render factor trends as styled HTML."""
    if trends.empty:
        return "<p>No factor trend data available.</p>"

    rows = []
    for factor in trends.index:
        trend = trends.loc[factor, "Trend"]
        sharpe = trends.loc[factor, "Rolling Sharpe"]
        diff = trends.loc[factor, "Diff %"]

        if trend == "Trending Up":
            color = "#16a34a"
            icon = "▲"
        elif trend == "Mean Reverting":
            color = "#dc2626"
            icon = "▼"
        else:
            color = "#64748b"
            icon = "●"

        rows.append(
            f'<tr>'
            f'<td style="font-weight:500;padding:8px 12px">{factor}</td>'
            f'<td style="color:{color};padding:8px 12px;font-weight:600">{icon} {trend}</td>'
            f'<td style="padding:8px 12px;text-align:right">{diff:.2f}%</td>'
            f'<td style="padding:8px 12px;text-align:right">{sharpe:.2f}</td>'
            f'</tr>'
        )

    return f"""
    <div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-family:Inter,system-ui,sans-serif;font-size:13px">
    <thead>
        <tr style="border-bottom:2px solid #cbd5e1">
            <th style="padding:8px 12px;text-align:left;color:#475569;font-weight:600">Factor</th>
            <th style="padding:8px 12px;text-align:left;color:#475569;font-weight:600">Trend</th>
            <th style="padding:8px 12px;text-align:right;color:#475569;font-weight:600">MA Diff %</th>
            <th style="padding:8px 12px;text-align:right;color:#475569;font-weight:600">60d Sharpe</th>
        </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    """


def render_factor_monitor_tab(prices, factor_data, mm_data_available: bool) -> None:
    """Render the Factor Monitor tab."""
    st.caption(
        "Monitor GS factor performance (Momentum, Value, Size, Quality, etc.), "
        "identify trending vs mean-reverting factors, and screen stocks by factor exposure."
    )

    if factor_data is None:
        st.warning("Factor data unavailable. Reload via sidebar or check Prismatic connection.")
        return

    if not mm_data_available or prices is None or prices.empty:
        st.info("No market data loaded. Factor performance is shown below; stock-level betas require market data.")

    sector_map = get_sector_map()
    name_map = get_name_map()

    factor_returns = factor_data.returns

    # Compute stock-level betas if market data available
    stock_returns = None
    market_returns = None
    if mm_data_available and prices is not None and not prices.empty:
        constituent_cols = [c for c in prices.columns if c != "SPX" and c in sector_map]
        stock_returns = prices[constituent_cols].pct_change().dropna(how="all")
        if "SPX" in prices.columns:
            market_returns = prices["SPX"].pct_change().dropna()

    result = compute_factor_monitor(factor_returns, stock_returns, market_returns)

    # --- Factor performance cards ---
    st.subheader("Factor Performance")
    if not result.factor_perf_summary.empty:
        factor_cols = st.columns(min(len(result.factor_perf_summary.index), 4))
        for i, factor in enumerate(result.factor_perf_summary.index):
            with factor_cols[i % len(factor_cols)]:
                perf = result.factor_perf_summary.loc[factor]
                ret_1d = perf.get("1D", 0)
                ret_1m = perf.get("1M", 0)
                st.metric(
                    factor,
                    f"{ret_1m:+.2%} (1M)",
                    delta=f"{ret_1d:+.2%} (1D)",
                )

    # --- Cumulative returns chart ---
    period = st.radio(
        "Chart Period", ["1M", "3M", "6M", "1Y", "All"],
        horizontal=True, key="mm_factor_period",
    )
    period_days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "All": len(factor_returns)}
    n_days = period_days.get(period, len(factor_returns))
    chart_data = factor_returns.iloc[-n_days:]

    st.plotly_chart(
        _factor_cumulative_chart(chart_data, f"Factor Cumulative Returns ({period})"),
        use_container_width=True,
    )

    # --- Factor trends ---
    st.subheader("Factor Trend Classification")
    st.caption("20-day MA vs 60-day MA of cumulative returns. Diff > 0.5% = Trending Up, < -0.5% = Mean Reverting.")
    if not result.factor_trends.empty:
        st.html(_factor_trend_table_html(result.factor_trends))
    else:
        st.info("Insufficient data for trend classification.")

    # --- Stock factor betas ---
    st.divider()
    st.subheader("Stock Factor Exposures")

    if result.stock_factor_betas.empty:
        st.info("Stock-level factor betas require S&P 500 price data. Refresh market data first.")
        return

    # Factor selector
    available_factors = [c for c in result.stock_factor_betas.columns if c != "Market"]
    if not available_factors:
        st.info("No factor beta data available.")
        return

    selected_factor = st.selectbox(
        "Select Factor", available_factors, key="mm_factor_select",
    )

    # Top exposures
    betas = result.stock_factor_betas[selected_factor].dropna().sort_values(ascending=False)

    col_long, col_short = st.columns(2)

    with col_long:
        st.subheader(f"Top 20: Highest {selected_factor} Beta")
        top = betas.head(20)
        top_df = top.reset_index()
        top_df.columns = ["Ticker", "Beta"]
        top_df["Name"] = top_df["Ticker"].map(name_map)
        top_df["Sector"] = top_df["Ticker"].map(sector_map)
        top_df = top_df[["Ticker", "Name", "Sector", "Beta"]]
        st.dataframe(
            top_df.style.format({"Beta": "{:.3f}"}),
            use_container_width=True, hide_index=True, height=500,
        )

    with col_short:
        st.subheader(f"Bottom 20: Lowest {selected_factor} Beta")
        bottom = betas.tail(20).sort_values()
        bottom_df = bottom.reset_index()
        bottom_df.columns = ["Ticker", "Beta"]
        bottom_df["Name"] = bottom_df["Ticker"].map(name_map)
        bottom_df["Sector"] = bottom_df["Ticker"].map(sector_map)
        bottom_df = bottom_df[["Ticker", "Name", "Sector", "Beta"]]
        st.dataframe(
            bottom_df.style.format({"Beta": "{:.3f}"}),
            use_container_width=True, hide_index=True, height=500,
        )
