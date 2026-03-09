import json
import os

import streamlit as st

from config import (
    DEFAULT_END_DATE,
    DEFAULT_FACTOR_TICKERS,
    DEFAULT_RETURN_METHOD,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_START_DATE,
    DEFAULT_STOCK_TICKERS,
    INTERVAL_OPTIONS,
    RETURN_METHODS,
    ROLLING_WINDOW_OPTIONS,
)
from data.fetcher import clear_cache, validate_interval_date_range
from utils.validation import parse_tickers, validate_date_range

_LOCAL_PEER_GROUPS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "peer_groups.json")


def _peer_groups_path() -> str:
    """Return the peer groups JSON path — persistent disk on Render, local file otherwise."""
    peer_dir = os.environ.get("PEER_GROUPS_DIR")
    if peer_dir and os.path.isdir(peer_dir):
        return os.path.join(peer_dir, "peer_groups.json")
    return _LOCAL_PEER_GROUPS_PATH


def _load_peer_groups() -> dict:
    """Load saved peer groups from JSON file."""
    path = _peer_groups_path()
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    # Fall back to local defaults (e.g. first run on Render disk)
    try:
        with open(_LOCAL_PEER_GROUPS_PATH, "r") as f:
            groups = json.load(f)
        # Seed the persistent path if it's different
        if path != _LOCAL_PEER_GROUPS_PATH:
            _save_peer_groups(groups)
        return groups
    except (FileNotFoundError, json.JSONDecodeError):
        return {"stocks": {}, "factors": {}}


def _save_peer_groups(groups: dict):
    """Save peer groups to JSON file."""
    path = _peer_groups_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(groups, f, indent=4)


def _render_peer_group_controls(section: str, current_tickers: str, key_prefix: str) -> None:
    """Render load/save/delete controls for a peer group section.

    When the user selects a group, stores the tickers in a pending key
    (consumed before the widget renders on the next rerun) and resets
    the selectbox to avoid infinite rerun loops.
    """
    groups = _load_peer_groups()
    section_groups = groups.get(section, {})
    group_names = list(section_groups.keys())

    select_key = f"{key_prefix}_load"

    # Reset selectbox before it renders (deferred from previous rerun)
    if st.session_state.pop(f"{key_prefix}_reset_select", False):
        st.session_state[select_key] = ""

    if group_names:
        col_load, col_del = st.columns([3, 1])
        with col_load:
            selected = st.selectbox(
                "Load saved group",
                options=[""] + group_names,
                index=0,
                key=select_key,
                help="Select a saved group to load its tickers.",
                label_visibility="collapsed",
                placeholder="Load saved group...",
            )
        with col_del:
            if st.button("X", key=f"{key_prefix}_del", help="Delete selected group"):
                if selected and selected in section_groups:
                    del section_groups[selected]
                    groups[section] = section_groups
                    _save_peer_groups(groups)
                    st.rerun()

        if selected and selected in section_groups:
            # Stage the load and selectbox reset for next rerun
            # (can't modify widget keys after instantiation)
            st.session_state[f"{key_prefix}_pending"] = section_groups[selected]
            st.session_state[f"{key_prefix}_reset_select"] = True
            st.rerun()

    # Save controls
    col_name, col_save = st.columns([3, 1])
    with col_name:
        save_name = st.text_input(
            "Group name",
            key=f"{key_prefix}_save_name",
            label_visibility="collapsed",
            placeholder="Save as...",
        )
    with col_save:
        if st.button("Save", key=f"{key_prefix}_save", help="Save current tickers as a named group"):
            if save_name and save_name.strip():
                section_groups[save_name.strip()] = current_tickers
                groups[section] = section_groups
                _save_peer_groups(groups)
                st.success(f"Saved '{save_name.strip()}'")
                st.rerun()


def render_sidebar() -> dict | None:
    """Render all sidebar inputs and return a params dict, or None if inputs are invalid."""
    with st.sidebar:
        st.header("Settings")

        # Seed defaults on first run only
        if "stock_tickers" not in st.session_state:
            st.session_state["stock_tickers"] = DEFAULT_STOCK_TICKERS
        if "factor_tickers" not in st.session_state:
            st.session_state["factor_tickers"] = DEFAULT_FACTOR_TICKERS

        # Apply pending peer group loads BEFORE widgets render
        if "pg_stocks_pending" in st.session_state:
            st.session_state["stock_tickers"] = st.session_state.pop("pg_stocks_pending")
        if "pg_factors_pending" in st.session_state:
            st.session_state["factor_tickers"] = st.session_state.pop("pg_factors_pending")

        # --- Stock tickers ---
        st.subheader("Stocks / ETFs")
        raw_stocks = st.text_area(
            "Stock tickers (comma-separated)",
            help="Enter US stock or ETF tickers separated by commas.",
            key="stock_tickers",
        )
        stock_tickers = parse_tickers(raw_stocks)

        _render_peer_group_controls("stocks", raw_stocks, "pg_stocks")

        # --- Factor / Index tickers ---
        st.subheader("Factors / Indices")
        raw_factors = st.text_area(
            "Factor & index tickers (comma-separated)",
            help="Enter factor ETFs, sector ETFs, or index tickers (e.g. SPY, QQQ, IWM, XLF).",
            key="factor_tickers",
        )
        factor_tickers = parse_tickers(raw_factors)

        _render_peer_group_controls("factors", raw_factors, "pg_factors")

        # Deduplicate across both lists (factor list wins if overlap)
        stock_tickers = [t for t in stock_tickers if t not in factor_tickers]

        all_tickers = stock_tickers + factor_tickers
        if len(all_tickers) < 2:
            st.error("Enter at least 2 valid tickers total across stocks and factors.")
            return None

        if not factor_tickers:
            st.warning("Add at least one factor/index ticker for beta analysis.")

        st.divider()

        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=DEFAULT_START_DATE)
        with col2:
            end_date = st.date_input("End", value=DEFAULT_END_DATE)

        date_err = validate_date_range(start_date, end_date)
        if date_err:
            st.error(date_err)
            return None

        st.divider()

        # Rolling window
        window = st.select_slider(
            "Rolling window (days)",
            options=ROLLING_WINDOW_OPTIONS,
            value=DEFAULT_ROLLING_WINDOW,
        )

        # Return method
        return_method = st.radio(
            "Return method",
            options=RETURN_METHODS,
            index=RETURN_METHODS.index(DEFAULT_RETURN_METHOD),
            horizontal=True,
        )

        # Data interval
        interval = st.selectbox(
            "Data interval",
            options=INTERVAL_OPTIONS,
            index=0,
            key="data_interval",
            help="Data frequency. Intraday intervals have limited date range support.",
        )

        # Validate interval vs date range
        interval_err = validate_interval_date_range(interval, start_date, end_date)
        if interval_err:
            st.warning(interval_err)

        st.divider()

        # Cache controls
        if st.button("Clear data cache", key="clear_cache"):
            clear_cache()
            st.success("Cache cleared. Data will be re-fetched on next run.")

    return {
        "stock_tickers": stock_tickers,
        "factor_tickers": factor_tickers,
        "all_tickers": all_tickers,
        "benchmarks": factor_tickers,  # factors/indices serve as benchmarks
        "start_date": start_date,
        "end_date": end_date,
        "window": window,
        "return_method": return_method,
        "interval": interval,
    }
