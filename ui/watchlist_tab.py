"""Watchlist & Changes tab — persistent idea capture and signal-change tracking.

Three sections:
  1. "What Changed Since [date]" — categorized signal transitions between today
     and a user-selected prior date.
  2. Watchlist table — user-curated tickers with live composite/RSI overlay
     and inline notes.
  3. Export — CSV download + Bloomberg-style ticker list.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from analytics.reversion import compute_reversion_signals
from analytics.signal_history import (
    SignalDelta,
    add_ticker,
    available_dates,
    build_snapshot,
    compute_deltas,
    load_history,
    load_watchlist,
    remove_ticker,
)
from data.market_monitor.constituents import get_name_map, get_sector_map

# Single source of truth for category ordering — keeps the UI deterministic.
_CATEGORY_ORDER = [
    "Entered Oversold",
    "Exited Oversold",
    "RSI Flip <30",
    "RSI Flip >70",
    "Broke 50d MA",
    "Reclaimed 50d MA",
    "New 52w High",
    "New 52w Low",
    "Factor Trend Flip",
]


def _delta_label(delta: SignalDelta) -> str:
    """Human-readable one-liner for a delta — used inside the pill."""
    if delta.metric == "factor_trend":
        return f"{delta.prior} → {delta.current}"
    if delta.metric == "rsi":
        return f"RSI {delta.prior:.0f} → {delta.current:.0f}"
    if delta.metric == "composite":
        return f"composite {delta.prior:.0f} → {delta.current:.0f}"
    if delta.metric == "ma_distance_50d":
        return f"MA dist {delta.prior:+.1%} → {delta.current:+.1%}"
    if delta.metric == "last_price":
        return f"price {delta.current:.2f}"
    return ""


def _render_changes_section(
    history: pd.DataFrame,
    name_map: dict[str, str],
    sector_map: dict[str, str],
) -> None:
    st.subheader("What Changed Since…")
    dates = sorted(history["date"].unique()) if not history.empty else []
    if len(dates) < 2:
        st.info(
            "Need at least two snapshots to compute deltas. Click "
            "**Refresh Market Data** in the sidebar — a snapshot is appended "
            "every refresh."
        )
        return

    current = pd.Timestamp(dates[-1])
    prior_options = [pd.Timestamp(d) for d in dates[:-1]][::-1]

    col_pick, col_filter = st.columns([1, 2])
    with col_pick:
        prior = st.selectbox(
            "Compare to:",
            prior_options,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            key="wl_prior_date",
        )
    with col_filter:
        category_filter = st.multiselect(
            "Category filter",
            _CATEGORY_ORDER,
            default=[],
            key="wl_category_filter",
        )

    deltas = compute_deltas(history, current, prior)
    if not deltas:
        st.info(f"No signal changes between {prior.date()} and {current.date()}.")
        return

    if category_filter:
        deltas = [d for d in deltas if d.category in category_filter]

    # Group by category in deterministic order
    by_cat: dict[str, list[SignalDelta]] = {cat: [] for cat in _CATEGORY_ORDER}
    for d in deltas:
        by_cat.setdefault(d.category, []).append(d)

    st.caption(
        f"{len(deltas)} change(s) between **{prior.date()}** and **{current.date()}**. "
        "Click **+ watchlist** on any row to capture the idea."
    )

    for cat in _CATEGORY_ORDER:
        bucket = by_cat.get(cat, [])
        if not bucket:
            continue
        with st.expander(f"{cat} ({len(bucket)})", expanded=False):
            for d in bucket:
                cols = st.columns([1, 1.2, 3, 1])
                with cols[0]:
                    st.markdown(f"**{d.ticker}**")
                with cols[1]:
                    st.caption(sector_map.get(d.ticker, name_map.get(d.ticker, "—")))
                with cols[2]:
                    st.caption(_delta_label(d))
                with cols[3]:
                    if st.button("+ watchlist", key=f"wl_add_{cat}_{d.ticker}"):
                        add_ticker(d.ticker, note=cat)
                        st.success(f"Added {d.ticker} to watchlist.")
                        st.rerun()


def _render_watchlist_section(
    prices: pd.DataFrame,
    name_map: dict[str, str],
    sector_map: dict[str, str],
) -> None:
    st.subheader("Watchlist")
    entries = load_watchlist()
    if not entries:
        st.info(
            "Watchlist is empty. Use the **+ watchlist** buttons above (or in "
            "the Trade Ideas tab) to capture ideas."
        )
        return

    # Live overlay: today's composite + RSI for each watched ticker.
    rev = compute_reversion_signals(prices)
    rows = []
    for e in entries:
        rows.append({
            "Ticker": e.ticker,
            "Name": name_map.get(e.ticker, e.ticker),
            "Sector": sector_map.get(e.ticker, "—"),
            "Composite": rev.composite_score.get(e.ticker, float("nan")),
            "RSI": rev.rsi_14.get(e.ticker, float("nan")),
            "Added On": e.added_on[:10],
            "Note": e.note,
            "Tags": ", ".join(e.tags),
        })
    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.format({"Composite": "{:.0f}", "RSI": "{:.0f}"}, na_rep="—"),
        use_container_width=True,
        hide_index=True,
        height=min(420, 60 + 35 * len(df)),
    )

    # Per-row remove buttons in a compact grid below the table.
    st.caption("Remove from watchlist:")
    cols_per_row = 6
    for chunk_start in range(0, len(entries), cols_per_row):
        chunk = entries[chunk_start:chunk_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, e in enumerate(chunk):
            with cols[i]:
                if st.button(f"× {e.ticker}", key=f"wl_remove_{e.ticker}"):
                    remove_ticker(e.ticker)
                    st.rerun()


def _render_export_section() -> None:
    st.subheader("Export")
    entries = load_watchlist()
    if not entries:
        st.caption("Add at least one ticker to enable exports.")
        return

    # CSV export
    df = pd.DataFrame([
        {"ticker": e.ticker, "added_on": e.added_on, "note": e.note,
         "tags": ", ".join(e.tags)}
        for e in entries
    ])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    fname = f"watchlist_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    st.download_button(
        "Download watchlist (CSV)",
        data=buf.getvalue(),
        file_name=fname,
        mime="text/csv",
    )

    # Bloomberg-style ticker list
    bbg_str = " ".join(f"{e.ticker} US Equity" for e in entries)
    st.code(bbg_str, language=None)
    st.caption("Paste the line above into Bloomberg to load the watchlist.")


def render_watchlist_tab(prices, mm_data_available: bool) -> None:
    """Entry point invoked from app.py."""
    st.caption(
        "Persistent watchlist + cross-day signal-change tracker. Snapshots are "
        "appended each time you refresh market data; this tab compares today "
        "to any prior snapshot."
    )

    if not mm_data_available or prices is None or prices.empty:
        st.info("No market data loaded. Click **Refresh Market Data** in the sidebar.")
        return

    sector_map = get_sector_map()
    name_map = get_name_map()

    history = load_history(lookback_days=60)

    _render_changes_section(history, name_map, sector_map)
    st.divider()
    _render_watchlist_section(prices, name_map, sector_map)
    st.divider()
    _render_export_section()
