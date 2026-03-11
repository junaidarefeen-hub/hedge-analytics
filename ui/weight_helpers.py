"""Shared weight management helpers for multi-ticker portfolios."""

from __future__ import annotations

import numpy as np
import streamlit as st


def equal_weight(n: int) -> float:
    """Return equal weight percentage for *n* constituents."""
    return round(100.0 / n, 2) if n > 0 else 0.0


def sync_weights(prefix: str, tickers: list[str]) -> None:
    """Reset weight keys to equal weight when the ticker list changes.

    Runs BEFORE widgets are rendered so session_state is clean.
    """
    state_key = f"{prefix}_prev_tickers"
    prev = st.session_state.get(state_key, None)
    if prev != tickers:
        eq = equal_weight(len(tickers))
        for tk in tickers:
            st.session_state[f"{prefix}_{tk}"] = eq
        if prev is not None:
            for old_tk in prev:
                if old_tk not in tickers:
                    st.session_state.pop(f"{prefix}_{old_tk}", None)
        st.session_state[state_key] = list(tickers)


def handle_normalize(prefix: str, tickers: list[str]) -> None:
    """If a normalize was requested last run, apply it before widgets render."""
    flag = f"{prefix}_do_normalize"
    if st.session_state.pop(flag, False):
        vals = {tk: st.session_state.get(f"{prefix}_{tk}", 0.0) for tk in tickers}
        total = sum(vals.values())
        if total > 0:
            for tk in tickers:
                st.session_state[f"{prefix}_{tk}"] = round(vals[tk] / total * 100, 2)


def render_weight_inputs(
    prefix: str,
    tickers: list[str],
    normalize_key: str,
) -> dict[str, float]:
    """Render per-ticker weight number_input widgets + normalize button.

    Returns dict of {ticker: weight_pct}.
    """
    weights_raw = {}
    eq = equal_weight(len(tickers))
    for tk in tickers:
        weights_raw[tk] = st.number_input(
            f"{tk} weight (%)",
            min_value=0.0,
            max_value=100.0,
            value=eq,
            step=1.0,
            key=f"{prefix}_{tk}",
        )
    weight_sum = sum(weights_raw.values())
    st.caption(f"Weight sum: **{weight_sum:.1f}%**")
    if st.button("Normalize to 100%", key=normalize_key):
        st.session_state[f"{prefix}_do_normalize"] = True
        st.rerun()
    return weights_raw


def weights_array(weights_raw: dict[str, float], tickers: list[str]) -> np.ndarray:
    """Convert percentage weight dict to decimal numpy array (sums to ~1.0)."""
    return np.array([weights_raw[tk] for tk in tickers]) / 100.0
