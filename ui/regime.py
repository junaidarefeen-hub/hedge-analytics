"""Regime Detection tab — volatility regimes + hedge effectiveness."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import numpy as np

from analytics.regime import (
    CorrelationRegimeResult,
    RegimeResult,
    correlation_regime_hedge_metrics,
    detect_correlation_regimes,
    detect_regimes,
    regime_hedge_effectiveness,
)
from config import (
    CORR_REGIME_N_REGIMES,
    REGIME_LABELS,
    REGIME_METHODS,
    REGIME_N_REGIMES,
    REGIME_VOL_WINDOW,
)
from ui.style import PLOTLY_LAYOUT
from utils.basket import basket_display_name, inject_basket_column

_REGIME_COLORS = ["#16a34a", "#2563eb", "#dc2626", "#ca8a04", "#7c3aed"]


def _price_chart_with_regimes(
    returns: pd.DataFrame, ticker: str, regime_result: RegimeResult,
) -> go.Figure:
    """Price chart with colored background for each regime."""
    cum = (1 + returns[ticker]).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values, mode="lines", name=ticker,
        line=dict(width=2.5, color="#0f172a"),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.4f}<extra></extra>",
    ))

    # Add regime background bands (no per-band labels to avoid clutter)
    regime_series = regime_result.regime_series
    labels = regime_result.labels
    prev_regime = None
    band_start = None

    for date, regime in regime_series.items():
        if regime != prev_regime:
            if prev_regime is not None and band_start is not None:
                color = _REGIME_COLORS[prev_regime % len(_REGIME_COLORS)]
                fig.add_vrect(
                    x0=band_start, x1=date,
                    fillcolor=color, opacity=0.15,
                    line_width=0,
                )
            band_start = date
            prev_regime = regime

    # Close last band
    if prev_regime is not None and band_start is not None:
        color = _REGIME_COLORS[prev_regime % len(_REGIME_COLORS)]
        fig.add_vrect(
            x0=band_start, x1=regime_series.index[-1],
            fillcolor=color, opacity=0.15,
            line_width=0,
        )

    # Add invisible traces for the legend to explain regime colors
    unique_regimes = sorted(regime_series.unique())
    for regime_id in unique_regimes:
        color = _REGIME_COLORS[regime_id % len(_REGIME_COLORS)]
        label = labels.get(regime_id, f"Regime {regime_id}")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=label,
            showlegend=True,
        ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"Cumulative Returns with Regime Overlay: {ticker}",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _rolling_vol_with_regimes(regime_result: RegimeResult) -> go.Figure:
    """Rolling vol chart with regime-colored background."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=regime_result.rolling_vol.index,
        y=regime_result.rolling_vol.values,
        mode="lines",
        name="Rolling Vol",
        line=dict(width=2, color="#2563eb"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Rolling Volatility by Regime",
        yaxis_title="Annualized Vol",
        yaxis_tickformat=".0%",
        height=350,
    )
    return fig


def render_regime_tab(returns: pd.DataFrame, params: dict):
    """Render the Regime Detection tab."""
    st.caption(
        "Detect volatility regimes (low, normal, high) in the market based on rolling volatility of a reference ticker. "
        "This helps you understand how your hedge performs differently in calm vs turbulent markets."
    )

    all_tickers = list(returns.columns)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ref_ticker = st.selectbox("Reference ticker", options=all_tickers, index=0, key="reg_ref",
                                  help="The ticker whose rolling volatility is used to define market regimes. Typically a broad index like SPY.")
    with c2:
        vol_window = st.number_input("Vol window", min_value=10, max_value=252, value=REGIME_VOL_WINDOW, step=10, key="reg_window",
                                     help="Number of trading days for computing rolling volatility. Larger windows smooth out short-term noise.")
    with c3:
        n_regimes = st.number_input("# Regimes", min_value=2, max_value=5, value=REGIME_N_REGIMES, step=1, key="reg_n",
                                    help="Number of volatility regimes to detect. 3 regimes = low/normal/high vol environments.")
    with c4:
        method = st.selectbox("Method", options=REGIME_METHODS, index=0, key="reg_method",
                              help="Quantile: split by percentile thresholds (simpler, deterministic). K-Means: cluster by volatility levels (data-driven, may vary between runs).")

    run_regime = st.button("Detect Regimes", type="primary", key="reg_run")

    if run_regime:
        try:
            regime_result = detect_regimes(
                returns=returns,
                reference_ticker=ref_ticker,
                window=vol_window,
                n_regimes=n_regimes,
                method=method,
            )
            st.session_state["regime_result"] = regime_result
            st.session_state["regime_ref_ticker"] = ref_ticker
        except Exception as e:
            st.error(f"Regime detection failed: {e}")
            return

    regime_result: RegimeResult | None = st.session_state.get("regime_result")
    if regime_result is None:
        st.info("Select a reference ticker and click **Detect Regimes**.")
        return

    ref = st.session_state.get("regime_ref_ticker", ref_ticker)

    # Context: date range
    start_dt = returns.index.min().strftime("%Y-%m-%d")
    end_dt = returns.index.max().strftime("%Y-%m-%d")
    st.caption(f"Reference: **{ref}** | Period: **{start_dt}** to **{end_dt}**")

    # Charts
    st.plotly_chart(
        _price_chart_with_regimes(returns, ref, regime_result),
        use_container_width=True,
    )
    st.plotly_chart(
        _rolling_vol_with_regimes(regime_result),
        use_container_width=True,
    )

    # Per-regime stats
    st.subheader("Per-Regime Statistics")
    fmt_stats = regime_result.per_regime_stats.copy()
    fmt_stats["Avg Return (ann.)"] = fmt_stats["Avg Return (ann.)"].map("{:.2%}".format)
    fmt_stats["Avg Vol (ann.)"] = fmt_stats["Avg Vol (ann.)"].map("{:.2%}".format)
    fmt_stats["% of Time"] = fmt_stats["% of Time"].map("{:.1f}%".format)
    st.dataframe(fmt_stats, use_container_width=True)

    # Hedge effectiveness by regime (if hedge_result exists)
    hedge_result = st.session_state.get("hedge_result")
    if hedge_result is not None:
        st.subheader("Hedge Effectiveness by Regime")
        # Reconstruct basket if multi-ticker
        if hedge_result.target_tickers and len(hedge_result.target_tickers) > 1:
            eff_returns, eff_target = inject_basket_column(
                returns, hedge_result.target_tickers, hedge_result.target_weights,
            )
            display_target = basket_display_name(hedge_result.target_tickers, hedge_result.target_weights)
        else:
            eff_returns = returns
            eff_target = hedge_result.target_ticker
            display_target = hedge_result.target_ticker
        st.caption(
            f"Strategy: **{hedge_result.strategy}** | Target: **{display_target}**"
        )
        try:
            eff = regime_hedge_effectiveness(
                returns=eff_returns,
                target=eff_target,
                hedges=hedge_result.hedge_instruments,
                weights=hedge_result.weights,
                regime_series=regime_result.regime_series,
                n_regimes=n_regimes,
                labels=regime_result.labels,
            )
            fmt_eff = eff.copy()
            for col in ["Unhedged Vol", "Hedged Vol"]:
                if col in fmt_eff.columns:
                    fmt_eff[col] = fmt_eff[col].map("{:.2%}".format)
            if "Vol Reduction (%)" in fmt_eff.columns:
                fmt_eff["Vol Reduction (%)"] = fmt_eff["Vol Reduction (%)"].map("{:.1f}%".format)
            if "Avg Hedged Return (ann.)" in fmt_eff.columns:
                fmt_eff["Avg Hedged Return (ann.)"] = fmt_eff["Avg Hedged Return (ann.)"].map("{:.2%}".format)
            if "Correlation" in fmt_eff.columns:
                fmt_eff["Correlation"] = fmt_eff["Correlation"].map("{:.3f}".format)
            st.dataframe(fmt_eff, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute hedge effectiveness: {e}")

        # Correlation regime analysis for optimizer hedges
        _render_optimizer_corr_regimes(returns, hedge_result, eff_returns, eff_target, vol_window)


# ── Shared Correlation Regime Charts ─────────────────────────────────────


def rolling_corr_regime_chart(
    corr_regime: CorrelationRegimeResult,
    full_period_corr: float | None = None,
) -> go.Figure:
    """Rolling correlation chart with regime-colored background bands."""
    series = corr_regime.rolling_correlation
    regime_series = corr_regime.regime_series
    labels = corr_regime.labels

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name="Rolling Correlation",
        line=dict(width=2, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>Corr: %{y:.3f}<extra></extra>",
    ))

    # Regime background bands
    prev_regime = None
    band_start = None
    for date, regime in regime_series.items():
        if regime != prev_regime:
            if prev_regime is not None and band_start is not None:
                color = _REGIME_COLORS[prev_regime % len(_REGIME_COLORS)]
                fig.add_vrect(x0=band_start, x1=date, fillcolor=color, opacity=0.15, line_width=0)
            band_start = date
            prev_regime = regime
    if prev_regime is not None and band_start is not None:
        color = _REGIME_COLORS[prev_regime % len(_REGIME_COLORS)]
        fig.add_vrect(x0=band_start, x1=regime_series.index[-1], fillcolor=color, opacity=0.15, line_width=0)

    # Legend traces
    for regime_id in sorted(regime_series.unique()):
        color = _REGIME_COLORS[regime_id % len(_REGIME_COLORS)]
        label = labels.get(regime_id, f"Regime {regime_id}")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=label, showlegend=True,
        ))

    if full_period_corr is not None:
        fig.add_hline(
            y=full_period_corr, line_dash="dash", line_color="#dc2626", line_width=1,
            annotation_text=f"Full-period: {full_period_corr:.3f}",
            annotation_position="top left",
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Rolling Correlation by Regime",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def regime_boxplot_chart(
    daily_hedged: pd.Series,
    regime_series: pd.Series,
    labels: dict[int, str],
) -> go.Figure:
    """Box plot of daily hedged returns by correlation regime."""
    common = daily_hedged.index.intersection(regime_series.index)
    hedged = daily_hedged.loc[common]
    regimes = regime_series.loc[common]

    fig = go.Figure()
    for regime_id in sorted(regimes.unique()):
        mask = regimes == regime_id
        color = _REGIME_COLORS[regime_id % len(_REGIME_COLORS)]
        label = labels.get(regime_id, f"Regime {regime_id}")
        fig.add_trace(go.Box(
            y=hedged[mask].values,
            name=label,
            marker_color=color,
            boxmean=True,
            hovertemplate=f"{label}<br>Return: %{{y:.4f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Daily Hedged Return Distribution by Correlation Regime",
        yaxis_title="Daily Hedged Return",
        height=400,
        showlegend=False,
    )
    return fig


def render_corr_regime_metrics_table(metrics: pd.DataFrame):
    """Format and render the correlation regime hedge metrics table."""
    fmt = metrics.copy()
    for col in ["Avg Correlation"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map("{:.3f}".format)
    for col in ["Unhedged Vol", "Hedged Vol"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map("{:.2%}".format)
    if "Vol Reduction (%)" in fmt.columns:
        fmt["Vol Reduction (%)"] = fmt["Vol Reduction (%)"].map("{:.1f}%".format)
    for col in ["Avg Unhedged Return (ann.)", "Avg Hedged Return (ann.)"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map("{:.2%}".format)
    st.dataframe(fmt, use_container_width=True)


# ── Optimizer Correlation Regime Section ─────────────────────────────────


def _render_optimizer_corr_regimes(returns, hedge_result, eff_returns, eff_target, vol_window):
    """Correlation regime analysis using optimizer hedge result."""
    st.subheader("Correlation Regime Analysis")
    st.caption(
        "Classify the rolling correlation between your long portfolio and hedge basket into regimes. "
        "Shows how hedge effectiveness varies when correlations are low vs high."
    )

    # Reconstruct daily series from optimizer result
    daily_standalone = eff_returns[eff_target].dropna()
    daily_hedge_basket = pd.Series(
        eff_returns[hedge_result.hedge_instruments].dropna().values @ np.abs(hedge_result.weights),
        index=eff_returns[hedge_result.hedge_instruments].dropna().index,
    )
    common = daily_standalone.index.intersection(daily_hedge_basket.index)
    daily_standalone = daily_standalone.loc[common]
    daily_hedge_basket = daily_hedge_basket.loc[common]
    daily_hedged = daily_standalone + pd.Series(
        eff_returns[hedge_result.hedge_instruments].loc[common].values @ hedge_result.weights,
        index=common,
    )

    rolling_corr = daily_standalone.rolling(window=vol_window, min_periods=vol_window).corr(daily_hedge_basket)

    c1, c2 = st.columns(2)
    with c1:
        cr_n = st.number_input(
            "# Correlation regimes", min_value=2, max_value=5,
            value=CORR_REGIME_N_REGIMES, step=1, key="reg_cr_n",
        )
    with c2:
        cr_method = st.selectbox(
            "Method", options=REGIME_METHODS, index=0, key="reg_cr_method",
        )

    if st.button("Analyze Correlation Regimes", type="secondary", use_container_width=True, key="reg_cr_run"):
        try:
            cr_result = detect_correlation_regimes(rolling_corr, cr_n, cr_method)
            cr_metrics = correlation_regime_hedge_metrics(
                daily_standalone, daily_hedged, daily_hedge_basket,
                cr_result.regime_series, cr_result.labels,
            )
            st.session_state["reg_cr_result"] = cr_result
            st.session_state["reg_cr_metrics"] = cr_metrics
        except Exception as e:
            st.error(f"Correlation regime analysis failed: {e}")
            return

    cr_result: CorrelationRegimeResult | None = st.session_state.get("reg_cr_result")
    cr_metrics = st.session_state.get("reg_cr_metrics")
    if cr_result is None:
        st.info("Click **Analyze Correlation Regimes** to see results.")
        return

    full_corr = float(daily_standalone.corr(daily_hedge_basket))
    st.plotly_chart(
        rolling_corr_regime_chart(cr_result, full_corr),
        use_container_width=True,
        key="reg_cr_corr_chart",
    )

    if cr_metrics is not None:
        render_corr_regime_metrics_table(cr_metrics)

    st.plotly_chart(
        regime_boxplot_chart(daily_hedged, cr_result.regime_series, cr_result.labels),
        use_container_width=True,
        key="reg_cr_boxplot",
    )
