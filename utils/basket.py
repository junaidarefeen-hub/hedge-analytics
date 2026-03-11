"""Basket construction utilities — synthetic column for multi-ticker long positions."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASKET_COLUMN_NAME = "__LONG_BASKET__"


def inject_basket_column(
    returns: pd.DataFrame,
    tickers: list[str],
    weights: np.ndarray,
) -> tuple[pd.DataFrame, str]:
    """Add a weighted basket column to returns if multi-ticker; passthrough if single.

    Returns (augmented_df, column_name) — the column name to use as the target.
    Single-ticker case returns the original DataFrame unmodified (zero copy).
    """
    if len(tickers) == 1:
        return returns, tickers[0]
    basket = (returns[tickers] * weights).sum(axis=1)
    augmented = returns.copy()
    augmented[BASKET_COLUMN_NAME] = basket
    return augmented, BASKET_COLUMN_NAME


def exclude_basket_constituents(
    candidates: list[str],
    basket_tickers: list[str],
) -> list[str]:
    """Remove basket constituent tickers from hedge candidate list."""
    basket_set = set(basket_tickers)
    return [c for c in candidates if c not in basket_set]


def basket_display_name(tickers: list[str], weights: np.ndarray) -> str:
    """Human-readable label like 'AAPL (50%) + MSFT (50%)'."""
    if len(tickers) == 1:
        return tickers[0]
    parts = []
    for tk, w in zip(tickers, weights):
        parts.append(f"{tk} ({w:.0%})")
    return " + ".join(parts)


def is_basket(target: str) -> bool:
    """Check if the target column is the synthetic basket."""
    return target == BASKET_COLUMN_NAME
