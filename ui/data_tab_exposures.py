"""Per-ticker factor exposure drill-down — standalone tab.

Lives in the Analysis tab group as the 6th tab. For each user-inserted stock
ticker, runs an OLS against a chosen market index and a user-selected subset
of the GS factor indices, then renders:
  * a tickers x factors beta heatmap with significance stars and R² alongside
  * a per-ticker expander with the full regression table and a cumulative
    factor-vs-idiosyncratic decomposition mini-chart
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.factor_analytics import (
    PerTickerExposures,
    _significance_star,
    compute_per_ticker_exposures,
)
from config import FA_DEFAULT_PER_TICKER_FACTORS
from data.factor_loader import FactorData
from ui.style import PLOTLY_LAYOUT


def _exposure_heatmap(result: PerTickerExposures) -> go.Figure:
    """Tickers x factors beta heatmap with significance stars baked into cells."""
    df = result.betas
    vals = df.values.astype(float)
    abs_max = max(
        abs(float(np.nanmin(vals))),
        abs(float(np.nanmax(vals))),
        0.01,
    )

    colorscale = [
        [0.0, "#b91c1c"],
        [0.25, "#fca5a5"],
        [0.5, "#ffffff"],
        [0.75, "#93c5fd"],
        [1.0, "#1d4ed8"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=vals,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale=colorscale,
            zmin=-abs_max,
            zmax=abs_max,
            showscale=True,
            colorbar=dict(title="Beta", thickness=14, len=0.8),
            hovertemplate="%{y} — %{x}: %{z:.4f}<extra></extra>",
        )
    )

    annotations = []
    for i, ticker in enumerate(df.index):
        for j, factor in enumerate(df.columns):
            val = df.iloc[i, j]
            if pd.isna(val):
                continue
            p = result.pvalues.iloc[i, j]
            star = _significance_star(p) if pd.notna(p) else ""
            norm = (val - (-abs_max)) / (2 * abs_max) if abs_max > 0 else 0.5
            norm = max(0, min(1, norm))
            use_white = norm < 0.15 or norm > 0.85
            annotations.append(dict(
                x=factor, y=ticker,
                text=f"{val:.2f}{star}",
                showarrow=False,
                font=dict(size=11, color="#ffffff" if use_white else "#1e293b"),
            ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Per-Ticker Factor Beta Heatmap",
        annotations=annotations,
        height=max(280, 35 * len(df) + 120),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _decomposition_chart(ticker: str, leg) -> go.Figure:
    """Cumulative additive decomposition: factor + idio = total."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=leg.cumsum_total.index,
        y=leg.cumsum_total.values,
        name="Total",
        line=dict(color="#0f172a", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=leg.cumsum_factor.index,
        y=leg.cumsum_factor.values,
        name="Factor (systematic)",
        line=dict(color="#2563eb", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=leg.cumsum_idio.index,
        y=leg.cumsum_idio.values,
        name="Idiosyncratic (alpha + ε)",
        line=dict(color="#16a34a", width=1.5, dash="dot"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{ticker} — Cumulative additive decomposition",
        yaxis_title="Cumulative return",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _summary_table(result: PerTickerExposures) -> pd.DataFrame:
    """Compact summary: R², annualised alpha, observations per ticker."""
    summary = pd.DataFrame({
        "R²": result.r_squared,
        "Ann. Alpha": result.alpha_ann,
        "Obs": result.n_obs,
    })
    return summary.loc[result.tickers]


def render_per_ticker_exposures(
    *,
    valid_stocks: list[str],
    returns: pd.DataFrame,
    factor_data: FactorData | None,
    benchmarks: list[str],
) -> None:
    """Render the standalone Factor Exposure tab."""
    st.caption(
        "Per-ticker OLS against a market regressor and a user-selected set of "
        "GS factor indices. Cells show beta with significance stars (***/**/*); "
        "hover for the exact value. Expand any ticker for the full regression "
        "table and a cumulative factor-vs-idiosyncratic decomposition."
    )

    if factor_data is None:
        st.info("Factor data unavailable — this tab requires the GS factor cache.")
        return
    if not valid_stocks:
        st.info("Add stock tickers in the sidebar to see per-ticker factor exposures.")
        return
    if not benchmarks:
        st.info(
            "Add at least one factor / index ticker (e.g. SPY) in the sidebar — "
            "the first one becomes the market regressor for this tab."
        )
        return

    available_benchmarks = [b for b in benchmarks if b in returns.columns]
    if not available_benchmarks:
        st.info("None of the sidebar benchmarks have data — cannot run regression.")
        return

    available_factors = list(factor_data.returns.columns)
    default_factors = [f for f in FA_DEFAULT_PER_TICKER_FACTORS if f in available_factors]

    col_market, col_factors = st.columns([1, 3])
    with col_market:
        market_index = st.selectbox(
            "Market regressor",
            available_benchmarks,
            key="exposure_market",
            help=(
                "The benchmark that serves as the 'market' factor. The "
                "remaining regressors come from the GS factor indices selected "
                "to the right."
            ),
        )
    with col_factors:
        selected_factors = st.multiselect(
            "GS factor regressors",
            options=available_factors,
            default=default_factors,
            key="exposure_factors",
            help=(
                "Pick which GS factor indices enter the regression alongside "
                "the market regressor. Default excludes Equal Weight (overlaps "
                "with market) and Beta (double-counts market exposure)."
            ),
        )

    if not selected_factors:
        st.info(
            "Select at least one GS factor regressor above to run the regression."
        )
        return

    stock_returns = returns[[t for t in valid_stocks if t in returns.columns]].dropna(how="all")
    if stock_returns.empty:
        st.info("No usable return history for the selected stocks.")
        return

    market_returns = returns[market_index].dropna()
    factor_returns = factor_data.returns[selected_factors]

    result = compute_per_ticker_exposures(
        stock_returns=stock_returns,
        market_returns=market_returns,
        factor_returns=factor_returns,
        market_index=market_index,
        factor_names=selected_factors,
    )

    if not result.tickers:
        st.info(
            "No tickers had enough overlapping history with the factor data "
            "to run a regression. Try a longer date range in the sidebar."
        )
        return

    col_chart, col_summary = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(
            _exposure_heatmap(result),
            use_container_width=True,
            key="exposure_heatmap",
        )
    with col_summary:
        st.markdown("**Fit summary**")
        st.dataframe(
            _summary_table(result).style.format({
                "R²": "{:.2f}",
                "Ann. Alpha": "{:+.1%}",
                "Obs": "{:.0f}",
            }),
            use_container_width=True,
            height=max(140, 35 * len(result.tickers) + 60),
        )

    st.caption(
        "Significance: *** p<0.01 · ** p<0.05 · * p<0.10 · "
        "no star = not statistically distinguishable from zero."
    )

    for ticker in result.tickers:
        leg = result.per_ticker_legs[ticker]
        with st.expander(f"{ticker} — full regression & decomposition", expanded=False):
            col_t, col_c = st.columns([1, 1])
            with col_t:
                st.markdown(f"**OLS table** (R² = {leg.ols.r_squared:.3f}, "
                            f"adj-R² = {leg.ols.r_squared_adj:.3f})")
                styled = leg.table.style.format({
                    "Beta": "{:+.4f}",
                    "Std Error": "{:.4f}",
                    "t-stat": "{:+.2f}",
                    "p-value": "{:.3f}",
                })
                st.dataframe(styled, hide_index=True, use_container_width=True)
            with col_c:
                st.plotly_chart(
                    _decomposition_chart(ticker, leg),
                    use_container_width=True,
                    key=f"exposure_decomp_{ticker}",
                )
