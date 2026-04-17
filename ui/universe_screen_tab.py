"""Universe Screener — applies hedge-side analytics across the SP500 universe.

Three modes (radio):
  1. Deep Drawdown Basing — stocks deep below peak whose composite signal is
     improving. Reversion-with-catalyst setups.
  2. Regime-Conditional — per-ticker stats in the current SPX volatility regime.
  3. Industry Pair Spreads — within-industry pairs at extreme z-scores.

The pair scan is the heaviest computation and is gated behind a button.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.reversion import compute_reversion_signals
from analytics.signal_history import add_ticker
from analytics.universe_drawdown import rank_underwater_basing
from analytics.universe_pairs import scan_industry_pairs
from analytics.universe_regime import latest_regime_label, regime_conditional_stats
from data.market_monitor.constituents import (
    get_industry_map,
    get_name_map,
    get_sector_map,
)
from ui.style import PLOTLY_LAYOUT


# ---------------------------------------------------------------------------
# Cached wrappers — keyed on the latest snapshot date so a refresh invalidates.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_drawdown_scan(price_hash: str, min_dd: float, min_days: int):
    """price_hash is a string derived from prices.index.max() so the cache
    invalidates on each market data refresh."""
    prices = _shared_prices()
    return rank_underwater_basing(
        prices, min_dd=min_dd, min_days_underwater=min_days,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_regime_stats(price_hash: str):
    prices = _shared_prices()
    if "SPX" not in prices.columns:
        return pd.DataFrame()
    constituents = [c for c in prices.columns if c != "SPX"]
    spx_returns = prices["SPX"].pct_change(fill_method=None).dropna()
    return regime_conditional_stats(prices[constituents], spx_returns)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_pair_scan(price_hash: str, top_k: int, window: int):
    prices = _shared_prices()
    industry_map = get_industry_map()
    constituents = [c for c in prices.columns if c in industry_map]
    sliced = prices[constituents]
    returns = sliced.pct_change(fill_method=None).dropna(how="all")
    return scan_industry_pairs(
        sliced, returns, industry_map,
        window=window, top_k_per_industry=top_k,
    )


def _shared_prices() -> pd.DataFrame:
    """Internal: fetched from the existing market-monitor cache. Imported lazily
    so this module can be tested without spinning up the full app."""
    from data.market_monitor.cache_manager import load_cached_prices
    return load_cached_prices()


def _price_hash(prices: pd.DataFrame) -> str:
    if prices is None or prices.empty:
        return "empty"
    return f"{prices.index.max().isoformat()}|{len(prices.columns)}"


# ---------------------------------------------------------------------------
# Mode 1: Deep Drawdown Basing
# ---------------------------------------------------------------------------

def _drawdown_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["days_underwater"],
        y=df["current_dd"] * 100,
        mode="markers+text",
        marker=dict(
            size=10,
            color=df["composite_60d_improvement"].fillna(0),
            colorscale=[
                [0, "#dc2626"],   # red = composite worsening
                [0.5, "#f8fafc"],
                [1, "#16a34a"],   # green = composite improving (basing)
            ],
            cmid=0,
            showscale=True,
            colorbar=dict(title="Composite<br>60d Δ"),
            line=dict(color="#475569", width=0.5),
        ),
        text=df["ticker"],
        textposition="top center",
        textfont=dict(size=10, color="#334155"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Days underwater: %{x}<br>"
            "Current DD: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Deep Drawdown Basing — green = composite improving",
        xaxis_title="Days underwater",
        yaxis_title="Current drawdown (%)",
        height=480,
        showlegend=False,
    )
    return fig


def _render_drawdown_mode(prices: pd.DataFrame, name_map, sector_map) -> None:
    col1, col2 = st.columns(2)
    with col1:
        min_dd = st.slider(
            "Minimum drawdown depth",
            -0.50, -0.05, -0.15, 0.05,
            format="%.2f",
            help="Only show stocks at least this far below their peak.",
        )
    with col2:
        min_days = st.slider(
            "Minimum days underwater", 10, 252, 40, 5,
        )

    df = _cached_drawdown_scan(_price_hash(prices), float(min_dd), int(min_days))
    if df.empty:
        st.info("No tickers match the current filters.")
        return

    df = df.head(50).copy()
    df["name"] = df["ticker"].map(lambda t: name_map.get(t, t))
    df["sector"] = df["ticker"].map(lambda t: sector_map.get(t, "—"))

    st.plotly_chart(_drawdown_scatter(df), use_container_width=True)

    display = df.copy()
    display["Current DD"] = display["current_dd"].map(lambda v: f"{v:.1%}")
    display["Days Under"] = display["days_underwater"]
    display["Recovery from Trough"] = display["recovery_from_trough_pct"].map(lambda v: f"{v:+.1%}")
    display["Composite Today"] = display["composite_today"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.0f}"
    )
    display["Composite 60d Δ"] = display["composite_60d_improvement"].map(
        lambda v: "—" if pd.isna(v) else f"{v:+.1f}"
    )
    st.dataframe(
        display[[
            "ticker", "name", "sector", "Current DD", "Days Under",
            "Recovery from Trough", "Composite Today", "Composite 60d Δ",
        ]].rename(columns={
            "ticker": "Ticker", "name": "Name", "sector": "Sector",
        }),
        use_container_width=True, hide_index=True, height=420,
    )

    _capture_widget(df["ticker"].tolist(), "Deep Drawdown Basing", "us_dd")


# ---------------------------------------------------------------------------
# Mode 2: Regime-Conditional
# ---------------------------------------------------------------------------

def _render_regime_mode(prices: pd.DataFrame, name_map, sector_map) -> None:
    if "SPX" not in prices.columns:
        st.warning("SPX not available in the market cache — cannot compute regimes.")
        return

    spx_returns = prices["SPX"].pct_change(fill_method=None).dropna()
    today_label = latest_regime_label(spx_returns)

    df = _cached_regime_stats(_price_hash(prices))
    if df.empty:
        st.info("No regime stats available.")
        return

    available_labels = sorted(df["regime_label"].unique())
    default_idx = (
        available_labels.index(today_label) if today_label in available_labels else 0
    )
    chosen = st.selectbox(
        "Regime",
        available_labels,
        index=default_idx,
        help=f"Today's SPX regime: **{today_label}**",
    )
    st.caption(f"Today's SPX regime: **{today_label}**.")

    sliced = df[df["regime_label"] == chosen].copy()
    sliced["name"] = sliced["ticker"].map(lambda t: name_map.get(t, t))
    sliced["sector"] = sliced["ticker"].map(lambda t: sector_map.get(t, "—"))
    sliced = sliced.sort_values("sharpe", ascending=False).head(50)

    display = sliced[[
        "ticker", "name", "sector", "avg_return_ann", "vol_ann",
        "sharpe", "vs_spx_alpha", "days",
    ]].rename(columns={
        "ticker": "Ticker", "name": "Name", "sector": "Sector",
        "avg_return_ann": "Ann. Return", "vol_ann": "Ann. Vol",
        "sharpe": "Sharpe", "vs_spx_alpha": "Alpha vs SPX", "days": "Days",
    })
    styled = display.style.format({
        "Ann. Return": "{:+.1%}", "Ann. Vol": "{:.1%}",
        "Sharpe": "{:+.2f}", "Alpha vs SPX": "{:+.1%}",
    }, na_rep="—")
    st.dataframe(styled, use_container_width=True, hide_index=True, height=480)

    _capture_widget(sliced["ticker"].tolist(), f"Regime: {chosen}", "us_regime")


# ---------------------------------------------------------------------------
# Mode 3: Industry Pair Spreads
# ---------------------------------------------------------------------------

def _render_pairs_mode(prices: pd.DataFrame) -> None:
    st.caption(
        "Within-industry pair scan. Heavy operation — runs once per refresh "
        "and is then cached. Use the **Run scan** button to start."
    )
    cols = st.columns([1, 1, 1])
    with cols[0]:
        top_k = st.slider("Pairs per industry", 1, 5, 3)
    with cols[1]:
        window = st.slider("Z-score window (days)", 30, 120, 60, 10)
    with cols[2]:
        min_zscore = st.slider("Minimum |z-score|", 0.0, 4.0, 2.0, 0.25)

    if st.button("Run scan", type="primary"):
        with st.spinner("Scanning industry pairs (~30-60s)…"):
            df = _cached_pair_scan(
                _price_hash(prices), int(top_k), int(window),
            )
        st.session_state["us_pairs_result"] = df

    df = st.session_state.get("us_pairs_result")
    if df is None or df.empty:
        st.info("Click **Run scan** to populate the pair leaderboard.")
        return

    filtered = df[df["current_zscore"].abs() >= min_zscore].copy()
    if filtered.empty:
        st.info(f"No pairs above |z| ≥ {min_zscore}. Lower the threshold or rerun the scan.")
        return

    display = filtered.copy()
    display["Z-Score"] = display["current_zscore"].map(lambda v: f"{v:+.2f}")
    display["Half-Life (d)"] = display["half_life_days"].map(
        lambda v: "—" if not np.isfinite(v) else f"{v:.0f}"
    )
    display["ADF p"] = display["adf_pvalue"].map(lambda v: f"{v:.3f}")
    display["Corr (60d)"] = display["rolling_corr_60d"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.2f}"
    )
    display = display[[
        "ticker_a", "ticker_b", "industry",
        "Z-Score", "Half-Life (d)", "ADF p", "Corr (60d)",
    ]].rename(columns={
        "ticker_a": "Long Leg", "ticker_b": "Short Leg", "industry": "Industry",
    })
    st.dataframe(display, use_container_width=True, hide_index=True, height=520)


# ---------------------------------------------------------------------------
# Watchlist capture (shared)
# ---------------------------------------------------------------------------

def _capture_widget(tickers: list[str], note: str, key_prefix: str) -> None:
    if not tickers:
        return
    cols = st.columns([1, 2, 1])
    with cols[0]:
        choice = st.selectbox(
            "Capture idea", tickers, key=f"{key_prefix}_pick",
        )
    with cols[1]:
        custom_note = st.text_input(
            "Note", value=note, key=f"{key_prefix}_note",
        )
    with cols[2]:
        st.write("")
        if st.button("+ Watchlist", key=f"{key_prefix}_btn"):
            add_ticker(choice, note=custom_note)
            st.success(f"Added {choice} to watchlist.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_universe_screen_tab(prices, mm_data_available: bool) -> None:
    st.caption(
        "Universe-wide screens applying the hedge-side analytics (drawdown, "
        "regime, pairs) across all SP500 names. Results cache for 1 hour or "
        "until the market data is refreshed."
    )

    if not mm_data_available or prices is None or prices.empty:
        st.info("No market data loaded. Click **Refresh Market Data** in the sidebar.")
        return

    sector_map = get_sector_map()
    name_map = get_name_map()

    mode = st.radio(
        "Screen",
        ["Deep Drawdown Basing", "Regime-Conditional", "Industry Pair Spreads"],
        horizontal=True,
        key="us_mode",
    )
    st.divider()

    if mode == "Deep Drawdown Basing":
        _render_drawdown_mode(prices, name_map, sector_map)
    elif mode == "Regime-Conditional":
        _render_regime_mode(prices, name_map, sector_map)
    else:
        _render_pairs_mode(prices)
