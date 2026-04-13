"""Pairs / Spread Analysis tab — cointegration, z-scores, mean-reversion."""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from analytics.pairs import compute_pairs_analysis
from config import DEFAULT_ROLLING_WINDOW, ROLLING_WINDOW_OPTIONS
from ui.style import PLOTLY_LAYOUT, render_metrics_table


# ---------------------------------------------------------------------------
# Quick-period helpers (same pattern as ui/performance.py)
# ---------------------------------------------------------------------------

_PERIODS = {
    "1M": timedelta(days=30),
    "3M": timedelta(days=91),
    "6M": timedelta(days=182),
    "1Y": timedelta(days=365),
}


def _apply_pending_dates() -> None:
    pending_start = st.session_state.pop("pairs_pending_start", None)
    pending_end = st.session_state.pop("pairs_pending_end", None)
    if pending_start is not None:
        st.session_state["pairs_start"] = pending_start
    if pending_end is not None:
        st.session_state["pairs_end"] = pending_end


# ---------------------------------------------------------------------------
# Endpoint label helper
# ---------------------------------------------------------------------------

def _add_label(fig: go.Figure, series, name: str, fmt: str = ".2f") -> None:
    s = series.dropna()
    if s.empty:
        return
    val = float(s.iloc[-1])
    fig.add_annotation(
        x=s.index[-1], y=val,
        text=f"  {name} {val:{fmt}}",
        showarrow=False, xanchor="left",
        font=dict(size=11, color="#334155"),
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _spread_chart(result) -> go.Figure:
    """Log spread with rolling mean ± 1σ/2σ bands."""
    fig = go.Figure()

    # ±2σ band (wider, lighter)
    upper2 = result.rolling_mean + 2 * result.rolling_std
    lower2 = result.rolling_mean - 2 * result.rolling_std
    fig.add_trace(go.Scatter(
        x=upper2.index, y=upper2.values, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=lower2.index, y=lower2.values, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(37,99,235,0.08)", showlegend=False, hoverinfo="skip",
    ))

    # ±1σ band (narrower, darker)
    upper1 = result.rolling_mean + result.rolling_std
    lower1 = result.rolling_mean - result.rolling_std
    fig.add_trace(go.Scatter(
        x=upper1.index, y=upper1.values, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=lower1.index, y=lower1.values, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(37,99,235,0.15)", showlegend=False, hoverinfo="skip",
    ))

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=result.rolling_mean.index, y=result.rolling_mean.values,
        mode="lines", name="Rolling Mean",
        line=dict(width=1.5, dash="dash", color="#64748b"),
        hovertemplate="%{x|%b %d, %Y}<br>Mean: %{y:.4f}<extra></extra>",
    ))

    # Spread
    fig.add_trace(go.Scatter(
        x=result.spread.index, y=result.spread.values,
        mode="lines", name="Log Spread",
        line=dict(width=2, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>Spread: %{y:.4f}<extra></extra>",
    ))
    _add_label(fig, result.spread, "Spread", ".4f")

    is_log = "log" in result.spread.name
    label = "Log Price Spread" if is_log else "Price Spread"
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{label} with Bollinger Bands",
        xaxis_title="", yaxis_title=label,
        height=350, margin=dict(r=120),
    )
    return fig


def _zscore_chart(result) -> go.Figure:
    """Rolling z-score with ±1 and ±2 threshold lines."""
    fig = go.Figure()

    zs = result.zscore.dropna()

    # Shade extreme regions (|z| > 2)
    fig.add_hrect(y0=2, y1=4, fillcolor="rgba(220,38,38,0.06)", line_width=0)
    fig.add_hrect(y0=-4, y1=-2, fillcolor="rgba(22,163,74,0.06)", line_width=0)

    # Z-score line
    fig.add_trace(go.Scatter(
        x=zs.index, y=zs.values,
        mode="lines", name="Z-Score",
        line=dict(width=2, color="#7c3aed"),
        hovertemplate="%{x|%b %d, %Y}<br>Z: %{y:.2f}<extra></extra>",
    ))
    _add_label(fig, zs, "Z", ".2f")

    # Threshold lines
    for level, color, dash in [
        (2, "#dc2626", "dot"), (1, "#94a3b8", "dot"),
        (0, "#334155", "dash"),
        (-1, "#94a3b8", "dot"), (-2, "#16a34a", "dot"),
    ]:
        fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1,
                       annotation_text=f"{level:+d}σ" if level != 0 else "0",
                       annotation_position="top right" if level >= 0 else "bottom right",
                       annotation_font_color=color, annotation_font_size=10)

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Rolling Z-Score",
        xaxis_title="", yaxis_title="Z-Score",
        height=300, margin=dict(r=120),
        showlegend=False,
    )
    return fig


def _correlation_chart(result) -> go.Figure:
    """Rolling correlation between the two tickers."""
    fig = go.Figure()

    rc = result.rolling_corr.dropna()
    fig.add_trace(go.Scatter(
        x=rc.index, y=rc.values,
        mode="lines", name="Rolling Correlation",
        line=dict(width=2, color="#0891b2"),
        hovertemplate="%{x|%b %d, %Y}<br>ρ: %{y:.3f}<extra></extra>",
    ))
    _add_label(fig, rc, "ρ", ".3f")

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Rolling Correlation",
        xaxis_title="", yaxis_title="Correlation",
        height=300, margin=dict(r=120),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_pairs_tab(prices, returns, params: dict) -> None:
    """Render the Pairs / Spread Analysis tab."""
    st.caption(
        "Spread analysis and mean-reversion statistics for ticker pairs. "
        "Identifies cointegrated pairs, measures dislocation via z-score, "
        "and estimates mean-reversion half-life."
    )

    _apply_pending_dates()

    all_tickers = params["all_tickers"]
    if len(all_tickers) < 2:
        st.info("Load at least 2 tickers to use the Pairs/Spread tab.")
        return

    # --- Controls ---
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])

    with c1:
        ticker_a = st.selectbox("Ticker A", options=all_tickers, index=0, key="pairs_a")
    with c2:
        default_b = min(1, len(all_tickers) - 1)
        ticker_b = st.selectbox("Ticker B", options=all_tickers, index=default_b, key="pairs_b")
    with c3:
        window = st.selectbox(
            "Rolling Window",
            options=ROLLING_WINDOW_OPTIONS,
            index=ROLLING_WINDOW_OPTIONS.index(DEFAULT_ROLLING_WINDOW),
            key="pairs_window",
        )
    with c4:
        spread_type = st.selectbox(
            "Spread Type",
            options=["log", "price"],
            index=0,
            key="pairs_spread_type",
            help="**Log**: scale-invariant, better for long lookbacks and cointegration tests. "
                 "**Price**: raw dollar spread, more intuitive for short-horizon P&L thinking.",
        )

    min_date = returns.index.min().date()
    max_date = returns.index.max().date()
    default_start = max(params["start_date"], min_date)
    default_end = min(params["end_date"], max_date)

    with c5:
        start = st.date_input("Start", value=default_start, min_value=min_date,
                               max_value=max_date, key="pairs_start")
    with c6:
        end = st.date_input("End", value=default_end, min_value=min_date,
                             max_value=max_date, key="pairs_end")

    # Quick-period buttons
    btn_cols = st.columns(6)
    ref_end = max_date
    for i, (label, delta) in enumerate(_PERIODS.items()):
        with btn_cols[i]:
            if st.button(label, key=f"pairs_qp_{label}", use_container_width=True):
                st.session_state["pairs_pending_start"] = ref_end - delta
                st.session_state["pairs_pending_end"] = ref_end
                st.rerun()
    with btn_cols[4]:
        if st.button("YTD", key="pairs_qp_ytd", use_container_width=True):
            st.session_state["pairs_pending_start"] = date(ref_end.year, 1, 1)
            st.session_state["pairs_pending_end"] = ref_end
            st.rerun()
    with btn_cols[5]:
        if st.button("Max", key="pairs_qp_max", use_container_width=True):
            st.session_state["pairs_pending_start"] = min_date
            st.session_state["pairs_pending_end"] = ref_end
            st.rerun()

    if ticker_a == ticker_b:
        st.warning("Select two different tickers for a meaningful spread analysis.")

    # --- Compute ---
    try:
        result = compute_pairs_analysis(
            prices, returns, ticker_a, ticker_b, window, start, end,
            spread_type=spread_type,
        )
    except ValueError as e:
        st.error(str(e))
        return

    # --- Metric cards ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric(
        "Current Z-Score", f"{result.current_zscore:.2f}",
        help="How far the spread is from its rolling mean, in standard deviations. "
             "±1 = mild, ±2 = notable dislocation, ±3 = extreme. "
             "Positive means A is rich vs B; negative means A is cheap vs B.",
    )
    m2.metric(
        "Z-Score Percentile", f"{result.zscore_percentile:.0f}%",
        help="Where the current z-score ranks in the historical distribution. "
             "95th = A is richer vs B than 95% of history. "
             "5th = A is cheaper vs B than 95% of history.",
    )
    hl_display = f"{result.half_life:.1f}" if result.half_life < 1000 else "∞"
    m3.metric(
        "Half-Life (days)", hl_display,
        help="Expected number of trading days for the spread dislocation to close by half. "
             "Lower = faster mean reversion. 10 days = fast, 50+ = slow/unreliable, ∞ = no evidence of reversion.",
    )
    m4.metric(
        "ADF p-value", f"{result.adf_pvalue:.3f}",
        help="Augmented Dickey-Fuller test for stationarity. "
             "Below 0.05 = the spread is statistically mean-reverting (good for pairs trading). "
             "Above 0.10 = no evidence of mean reversion — the spread may be a random walk.",
    )
    m5.metric(
        "Hedge Ratio (β)", f"{result.hedge_ratio:.3f}",
        help="OLS slope from log(A) ~ log(B). The number of units of B to short per unit of A "
             "to construct a market-neutral spread. β = 1.2 means short 1.2 shares of B per long share of A.",
    )
    rc_clean = result.rolling_corr.dropna()
    latest_corr = f"{rc_clean.iloc[-1]:.3f}" if len(rc_clean) > 0 else "N/A"
    m6.metric(
        "Correlation (latest)", latest_corr,
        help="Latest rolling correlation between the two tickers' returns. "
             "Above 0.7 = strong relationship (spread model is reliable). "
             "A sudden drop means the pair relationship may be breaking down.",
    )

    # --- Charts ---
    st.plotly_chart(_spread_chart(result), use_container_width=True)
    st.plotly_chart(_zscore_chart(result), use_container_width=True)
    st.plotly_chart(_correlation_chart(result), use_container_width=True)

    # --- Details ---
    with st.expander("Spread Statistics", expanded=False):
        import pandas as pd
        stats = pd.DataFrame({
            "Statistic": [
                "Hedge Ratio (β)", "Half-Life (days)", "ADF Statistic",
                "ADF p-value", "Mean Spread", "Spread Std Dev",
                "Current Z-Score", "Z-Score Percentile",
                "Observations", "Date Range",
            ],
            "Value": [
                f"{result.hedge_ratio:.4f}",
                hl_display,
                f"{result.adf_stat:.3f}",
                f"{result.adf_pvalue:.4f}",
                f"{result.mean_spread:.6f}",
                f"{result.spread_std:.6f}",
                f"{result.current_zscore:.3f}",
                f"{result.zscore_percentile:.1f}%",
                str(len(result.spread)),
                f"{result.spread.index[0].strftime('%Y-%m-%d')} to {result.spread.index[-1].strftime('%Y-%m-%d')}",
            ],
            "Description": [
                "OLS slope from log(A) ~ log(B). Units of B to short per unit of A.",
                "Trading days for the spread to close half the dislocation (AR(1) estimate).",
                "Augmented Dickey-Fuller t-statistic. More negative = stronger stationarity.",
                "ADF significance. < 0.05 = mean-reverting, > 0.10 = no evidence.",
                "Long-run average of the log spread over the full period.",
                "Long-run standard deviation of the log spread.",
                "Current spread dislocation in standard deviations from the rolling mean.",
                "Percentage of historical z-scores below the current reading.",
                "Number of overlapping trading days used in the analysis.",
                "Start and end dates of the analysis window.",
            ],
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)
