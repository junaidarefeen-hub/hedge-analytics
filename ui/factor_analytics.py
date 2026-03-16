"""Factor Analytics tab — regress basket returns against market + GS factor indices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.factor_analytics import (
    FactorAnalyticsResult,
    LegDecomposition,
    run_factor_analytics,
)
from config import (
    ANNUALIZATION_FACTOR,
    FA_DEFAULT_LONG_NOTIONAL,
    FA_DEFAULT_P_THRESHOLD,
    FA_DEFAULT_SHORT_NOTIONAL,
    FA_SIGNIFICANCE_LEVELS,
)
from data.factor_loader import FactorData, align_factor_returns
from ui.style import PLOTLY_LAYOUT
from ui.weight_helpers import equal_weight, handle_normalize, sync_weights



def render_factor_analytics_tab(
    returns: pd.DataFrame, params: dict, factor_data: FactorData | None,
):
    """Render the Factor Analytics tab."""
    all_tickers = list(returns.columns)

    st.caption(
        "Regress basket returns against a market index and GS factor price series. "
        "Produces OLS regression tables with significance testing, return/volatility "
        "decomposition, and a factor beta heatmap. Short basket is optional."
    )

    if factor_data is None:
        st.warning("Factor data not available. Check that Factor Prices.xlsx exists.")
        return

    # ── Portfolio Definition ──────────────────────────────────────────────
    col_long, col_short = st.columns(2)

    with col_long:
        st.subheader("Long Basket")
        long_notional = st.number_input(
            "Long notional ($)",
            min_value=1000.0,
            value=FA_DEFAULT_LONG_NOTIONAL,
            step=1_000_000.0,
            format="%.0f",
            key="fa_long_notional",
        )
        long_tickers = st.multiselect(
            "Long tickers",
            options=all_tickers,
            default=[all_tickers[0]] if all_tickers else [],
            key="fa_long_tickers",
        )

    with col_short:
        st.subheader("Short Basket (optional)")
        short_notional = st.number_input(
            "Short notional ($)",
            min_value=1000.0,
            value=FA_DEFAULT_SHORT_NOTIONAL,
            step=1_000_000.0,
            format="%.0f",
            key="fa_short_notional",
        )
        short_tickers = st.multiselect(
            "Short tickers",
            options=all_tickers,
            default=[],
            key="fa_short_tickers",
        )

    # Pre-render: sync weights on ticker change + apply pending normalizes
    sync_weights("fa_lw", long_tickers)
    handle_normalize("fa_lw", long_tickers)
    sync_weights("fa_sw", short_tickers)
    handle_normalize("fa_sw", short_tickers)

    # Render weight inputs
    with col_long:
        long_weights_raw: dict[str, float] = {}
        if long_tickers:
            eq = equal_weight(len(long_tickers))
            for tk in long_tickers:
                long_weights_raw[tk] = st.number_input(
                    f"{tk} weight (%)",
                    min_value=-100.0,
                    max_value=100.0,
                    value=eq,
                    step=1.0,
                    key=f"fa_lw_{tk}",
                )
            long_sum = sum(long_weights_raw.values())
            st.caption(f"Weight sum: **{long_sum:.1f}%**")
            if st.button("Normalize to 100%", key="fa_norm_long"):
                st.session_state["fa_lw_do_normalize"] = True
                st.rerun()

    with col_short:
        short_weights_raw: dict[str, float] = {}
        if short_tickers:
            eq = equal_weight(len(short_tickers))
            for tk in short_tickers:
                short_weights_raw[tk] = st.number_input(
                    f"{tk} weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=eq,
                    step=1.0,
                    key=f"fa_sw_{tk}",
                )
            short_sum = sum(short_weights_raw.values())
            st.caption(f"Weight sum: **{short_sum:.1f}%**")
            if st.button("Normalize to 100%", key="fa_norm_short"):
                st.session_state["fa_sw_do_normalize"] = True
                st.rerun()

    # ── Settings Row ──────────────────────────────────────────────────────
    st.divider()
    factor_display_names = list(factor_data.prices.columns)
    benchmarks = params.get("benchmarks", [])

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        market_options = [t for t in benchmarks if t in returns.columns]
        if not market_options:
            market_options = all_tickers[:1]
        market_index = st.selectbox(
            "Market index",
            options=market_options,
            index=0,
            key="fa_market_index",
            help="Market benchmark for the regression (from sidebar Factor/Index tickers).",
        )
    with s2:
        _default_factors = [
            f for f in [
                "Momentum", "Value", "Quality", "Leverage", "Profitability",
                "10Yr Sensitivity", "Size", "Growth", "Crowding",
            ] if f in factor_display_names
        ]
        selected_factors = st.multiselect(
            "GS Factors",
            options=factor_display_names,
            default=_default_factors,
            key="fa_factors",
            help="Select GS factor indices to include as regressors.",
        )
    with s3:
        p_threshold = st.number_input(
            "P-value threshold",
            min_value=0.001,
            max_value=0.50,
            value=FA_DEFAULT_P_THRESHOLD,
            step=0.01,
            format="%.3f",
            key="fa_p_threshold",
        )
    with s4:
        min_date = returns.index.min().date()
        max_date = returns.index.max().date()
        fa_start = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="fa_start",
        )
    with s5:
        fa_end = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="fa_end",
        )

    # ── Early exit if inputs incomplete ───────────────────────────────────
    if not long_tickers:
        st.info("Select at least one long ticker to begin.")
        return
    if not selected_factors:
        st.info("Select at least one GS factor.")
        return

    # ── Analyze Button ────────────────────────────────────────────────────
    run_btn = st.button("Analyze", type="primary", use_container_width=True, key="fa_run")

    if run_btn:
        try:
            result = _run_analysis(
                returns=returns,
                factor_data=factor_data,
                long_tickers=long_tickers,
                long_weights_raw=long_weights_raw,
                long_notional=long_notional,
                short_tickers=short_tickers,
                short_weights_raw=short_weights_raw,
                short_notional=short_notional,
                market_index=market_index,
                selected_factors=selected_factors,
                p_threshold=p_threshold,
                start_date=pd.Timestamp(fa_start),
                end_date=pd.Timestamp(fa_end),
            )
            st.session_state["fa_result"] = result
        except Exception as e:
            st.error(f"Factor analysis failed: {e}")
            return

    result: FactorAnalyticsResult | None = st.session_state.get("fa_result")
    if result is None:
        st.info("Configure portfolios and click **Analyze** to see results.")
        return

    # ── Render Results ────────────────────────────────────────────────────
    _render_results(result)


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def _run_analysis(
    returns: pd.DataFrame,
    factor_data: FactorData,
    long_tickers: list[str],
    long_weights_raw: dict[str, float],
    long_notional: float,
    short_tickers: list[str],
    short_weights_raw: dict[str, float],
    short_notional: float,
    market_index: str,
    selected_factors: list[str],
    p_threshold: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> FactorAnalyticsResult:
    """Build basket returns, align with factors, and run regressions."""
    # Slice returns to date range
    mask = (returns.index >= start_date) & (returns.index <= end_date)
    ret = returns[mask]

    # Build weighted long basket returns
    long_w = np.array([long_weights_raw[tk] for tk in long_tickers]) / 100.0
    long_ret = (ret[long_tickers] * long_w).sum(axis=1)

    # Market returns
    market_ret = ret[market_index]

    # Factor returns (selected subset)
    fr = factor_data.returns[selected_factors]

    # Align long + market + factors
    fr_aligned, long_aligned, market_aligned = align_factor_returns(fr, long_ret, market_ret)

    # Build short/combined if short basket provided
    has_short = bool(short_tickers)
    short_final: pd.Series | None = None
    combined_final: pd.Series | None = None

    if has_short:
        short_w = np.array([short_weights_raw[tk] for tk in short_tickers]) / 100.0
        short_ret = (ret[short_tickers] * short_w).sum(axis=1)
        hedge_ratio = short_notional / long_notional if long_notional > 0 else 0.0
        combined_ret = long_ret - hedge_ratio * short_ret

        fr_s, short_aligned, _ = align_factor_returns(fr, short_ret, market_ret)
        fr_c, combined_aligned, _ = align_factor_returns(fr, combined_ret, market_ret)

        # Intersect all date sets
        common = (
            fr_aligned.index
            .intersection(fr_s.index)
            .intersection(fr_c.index)
        )
        fr_aligned = fr_aligned.loc[common]
        long_aligned = long_aligned.loc[common]
        market_aligned = market_aligned.loc[common]
        short_final = short_aligned.loc[common]
        combined_final = combined_aligned.loc[common]

    common_dates = fr_aligned.index
    if len(common_dates) < 30:
        raise ValueError(f"Only {len(common_dates)} overlapping observations — need at least 30.")

    return run_factor_analytics(
        long_returns=long_aligned,
        market_returns=market_aligned,
        factor_returns=fr_aligned,
        market_index=market_index,
        factor_names=selected_factors,
        dates=common_dates,
        short_returns=short_final,
        combined_returns=combined_final,
        p_threshold=p_threshold,
    )


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def _render_results(result: FactorAnalyticsResult):
    """Render all factor analytics result sections."""
    legs = _active_legs(result)

    # a. Summary metric cards
    st.subheader("Regression Summary")
    cols = st.columns(2 * len(legs))
    for i, (name, leg) in enumerate(legs):
        cols[2 * i].metric(f"{name} R²", f"{leg.ols.r_squared:.3f}")
        cols[2 * i + 1].metric(f"{name} Adj R²", f"{leg.ols.r_squared_adj:.3f}")

    # b. Regression tables
    st.subheader("OLS Regression Results")
    if len(legs) == 1:
        name, leg = legs[0]
        _render_regression_table(leg.table, leg.ols, name)
    else:
        tab_widgets = st.tabs([name for name, _ in legs])
        for tab, (name, leg) in zip(tab_widgets, legs):
            with tab:
                _render_regression_table(leg.table, leg.ols, name)

    # c. Factor beta heatmap
    st.subheader("Factor Beta Heatmap")
    st.plotly_chart(_beta_heatmap_chart(result), use_container_width=True, key="fa_heatmap")

    # d. Return decomposition
    st.subheader("Return Decomposition")
    ret_mode = st.radio(
        "View",
        options=["Compounded (growth of $1)", "Additive (cumulative sum)"],
        horizontal=True,
        key="fa_return_mode",
        help="Compounded shows (1+r).cumprod(); additive shows r.cumsum() where factor + idio = total exactly.",
    )
    _render_return_decomposition(legs, additive=ret_mode.startswith("Additive"))

    # e. Volatility decomposition over time
    st.subheader("Volatility Decomposition")
    vol_window = st.select_slider(
        "Rolling window (days)",
        options=[20, 30, 60, 90, 120, 252],
        value=60,
        key="fa_vol_window",
    )
    _render_vol_decomposition(legs, vol_window)


def _active_legs(result: FactorAnalyticsResult) -> list[tuple[str, LegDecomposition]]:
    """Return list of (name, leg) for legs that exist."""
    legs = [("Long", result.long)]
    if result.has_short:
        legs.append(("Short", result.short))
        legs.append(("Combined", result.combined))
    return legs


def _render_regression_table(table: pd.DataFrame, ols: "OLSResult", label: str):
    """Render a single leg's regression table with formatting."""
    from analytics.factor_analytics import OLSResult  # deferred to avoid circular at module level

    fmt = table.copy()
    fmt["Beta"] = fmt["Beta"].map(lambda x: f"{x:.6f}")
    fmt["Std Error"] = fmt["Std Error"].map(lambda x: f"{x:.6f}")
    fmt["t-stat"] = fmt["t-stat"].map(lambda x: f"{x:.3f}")
    fmt["p-value"] = fmt["p-value"].map(lambda x: f"{x:.4f}")
    st.dataframe(fmt, use_container_width=True, hide_index=True)
    st.caption(
        f"**{label}** — F-stat: {ols.f_stat:.2f} (p={ols.f_pvalue:.4f}) | "
        f"R²: {ols.r_squared:.4f} | Adj R²: {ols.r_squared_adj:.4f} | "
        f"N: {ols.n_obs} | "
        f"Sig: \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10"
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _beta_heatmap_chart(result: FactorAnalyticsResult) -> go.Figure:
    """Diverging heatmap of factor betas across active legs."""
    df = result.beta_heatmap
    vals = df.values.astype(float)
    abs_max = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals))), 0.01)

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

    # Build OLS lookup for p-values
    ols_lookup = {"Long": result.long.ols}
    if result.has_short:
        ols_lookup["Short"] = result.short.ols
        ols_lookup["Combined"] = result.combined.ols

    annotations = []
    for i, leg in enumerate(df.index):
        ols = ols_lookup[leg]
        for j, factor in enumerate(df.columns):
            val = df.iloc[i, j]
            if pd.isna(val):
                continue
            p = ols.p_values[j + 1]  # j+1 because index 0 is intercept
            star = _significance_star(p)
            norm = (val - (-abs_max)) / (2 * abs_max) if abs_max > 0 else 0.5
            norm = max(0, min(1, norm))
            use_white = norm < 0.15 or norm > 0.85
            annotations.append(dict(
                x=factor, y=leg,
                text=f"{val:.3f}{star}",
                showarrow=False,
                font=dict(size=11, color="#ffffff" if use_white else "#1e293b"),
            ))
    fig.update_layout(annotations=annotations)

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Factor Betas by Portfolio Leg",
        height=max(220, 80 * len(df)),
        xaxis=dict(side="bottom", tickangle=-45, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def _significance_star(p: float) -> str:
    for threshold in sorted(FA_SIGNIFICANCE_LEVELS.keys()):
        if p < threshold:
            return FA_SIGNIFICANCE_LEVELS[threshold]
    return ""


def _render_return_decomposition(
    legs: list[tuple[str, LegDecomposition]], *, additive: bool, key_prefix: str = "fa",
):
    """Render return decomposition charts."""
    if len(legs) == 1:
        name, leg = legs[0]
        st.plotly_chart(
            _return_decomp_chart(name, leg, additive),
            use_container_width=True,
            key=f"{key_prefix}_retdecomp_{name}",
        )
        st.caption(_return_stats(leg))
    else:
        tabs = st.tabs([name for name, _ in legs])
        for tab, (name, leg) in zip(tabs, legs):
            with tab:
                st.plotly_chart(
                    _return_decomp_chart(name, leg, additive),
                    use_container_width=True,
                    key=f"{key_prefix}_retdecomp_{name}",
                )
                st.caption(_return_stats(leg))


def _return_stats(leg: LegDecomposition) -> str:
    """Descriptive stats: % of total return from factor vs idiosyncratic."""
    total_ret = float(leg.cumsum_total.iloc[-1])
    factor_ret = float(leg.cumsum_factor.iloc[-1])
    idio_ret = float(leg.cumsum_idio.iloc[-1])
    if abs(total_ret) > 1e-10:
        factor_pct = factor_ret / total_ret
        idio_pct = idio_ret / total_ret
        return (
            f"Total return: **{total_ret:.2%}** — "
            f"Factor: **{factor_ret:.2%}** ({factor_pct:.0%} of total) | "
            f"Idiosyncratic: **{idio_ret:.2%}** ({idio_pct:.0%} of total)"
        )
    return f"Total return: **{total_ret:.2%}** — Factor: **{factor_ret:.2%}** | Idiosyncratic: **{idio_ret:.2%}**"


def _vol_stats(leg: LegDecomposition) -> str:
    """Descriptive stats: % of total variance from factor vs idiosyncratic."""
    var_total = leg.vol_total ** 2
    var_factor = leg.vol_factor ** 2
    var_idio = leg.vol_idio ** 2
    factor_pct = var_factor / var_total if var_total > 0 else 0
    idio_pct = var_idio / var_total if var_total > 0 else 0
    return (
        f"Total vol: **{leg.vol_total:.1%}** — "
        f"Factor vol: **{leg.vol_factor:.1%}** ({factor_pct:.0%} of variance) | "
        f"Idiosyncratic vol: **{leg.vol_idio:.1%}** ({idio_pct:.0%} of variance)"
    )


def _return_decomp_chart(label: str, leg: LegDecomposition, additive: bool) -> go.Figure:
    if additive:
        factor_y = leg.cumsum_factor
        idio_y = leg.cumsum_idio
        total_y = leg.cumsum_total
        title = f"{label} — Cumulative Return Decomposition (Additive)"
    else:
        factor_y = leg.cum_factor - 1
        idio_y = leg.cum_idio - 1
        total_y = leg.cum_total - 1
        title = f"{label} — Cumulative Return Decomposition (Compounded)"
    y_label = "Cumulative Return"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=factor_y.index, y=factor_y.values,
        mode="lines", name="Factor",
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.2)",
        line=dict(width=1.5, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>Factor: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=idio_y.index, y=idio_y.values,
        mode="lines", name="Idiosyncratic",
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.2)",
        line=dict(width=1.5, color="#7c3aed"),
        hovertemplate="%{x|%b %d, %Y}<br>Idio: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=total_y.index, y=total_y.values,
        mode="lines", name="Total",
        line=dict(width=2.5, color="#0f172a", dash="dot"),
        hovertemplate="%{x|%b %d, %Y}<br>Total: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_label,
        height=400,
    )
    return fig


def _render_vol_decomposition(
    legs: list[tuple[str, LegDecomposition]], window: int, key_prefix: str = "fa",
):
    """Render rolling vol decomposition as area charts over time."""
    if len(legs) == 1:
        name, leg = legs[0]
        st.plotly_chart(
            _rolling_vol_chart(name, leg, window),
            use_container_width=True,
            key=f"{key_prefix}_voldecomp_{name}",
        )
        st.caption(_vol_stats(leg))
    else:
        tabs = st.tabs([name for name, _ in legs])
        for tab, (name, leg) in zip(tabs, legs):
            with tab:
                st.plotly_chart(
                    _rolling_vol_chart(name, leg, window),
                    use_container_width=True,
                    key=f"{key_prefix}_voldecomp_{name}",
                )
                st.caption(_vol_stats(leg))


def _rolling_vol_chart(label: str, leg: LegDecomposition, window: int) -> go.Figure:
    """Rolling vol decomposition: factor and idio vol as filled areas, total as line."""
    sqrt_ann = np.sqrt(ANNUALIZATION_FACTOR)
    roll_factor = leg.daily_factor.rolling(window).std() * sqrt_ann
    roll_idio = leg.daily_idio.rolling(window).std() * sqrt_ann
    roll_total = leg.daily_total.rolling(window).std() * sqrt_ann

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_factor.index, y=roll_factor.values,
        mode="lines", name="Factor Vol",
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.2)",
        line=dict(width=1.5, color="#2563eb"),
        hovertemplate="%{x|%b %d, %Y}<br>Factor Vol: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll_idio.index, y=roll_idio.values,
        mode="lines", name="Idiosyncratic Vol",
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.2)",
        line=dict(width=1.5, color="#7c3aed"),
        hovertemplate="%{x|%b %d, %Y}<br>Idio Vol: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll_total.index, y=roll_total.values,
        mode="lines", name="Total Vol",
        line=dict(width=2.5, color="#0f172a", dash="dot"),
        hovertemplate="%{x|%b %d, %Y}<br>Total Vol: %{y:.1%}<extra></extra>",
    ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{label} — {window}-Day Rolling Volatility Decomposition (Annualized)",
        xaxis_title="",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        height=400,
    )
    return fig
