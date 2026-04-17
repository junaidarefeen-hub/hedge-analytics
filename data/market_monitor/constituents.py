"""S&P 500 constituent list and GICS sector mapping."""

from __future__ import annotations

import json
from pathlib import Path

_CONSTITUENTS_FILE = Path(__file__).parent / "sp500_constituents.json"

GICS_SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

_cache: dict[str, dict] | None = None


def load_constituents() -> dict[str, dict]:
    """Load S&P 500 constituents from JSON.

    Returns:
        Dict mapping ticker -> {"name", "sector", "industry"}
    """
    global _cache
    if _cache is None:
        _cache = json.loads(_CONSTITUENTS_FILE.read_text(encoding="utf-8"))
    return _cache


def get_sector_map() -> dict[str, str]:
    """Return ticker -> GICS sector mapping."""
    return {ticker: info["sector"] for ticker, info in load_constituents().items()}


def get_name_map() -> dict[str, str]:
    """Return ticker -> company name mapping."""
    return {ticker: info["name"] for ticker, info in load_constituents().items()}


def get_industry_map() -> dict[str, str]:
    """Return ticker -> GICS industry mapping (more granular than sector)."""
    return {ticker: info["industry"] for ticker, info in load_constituents().items()}


def get_tickers_by_sector(sector: str) -> list[str]:
    """Return sorted list of tickers in a given GICS sector."""
    return sorted(
        ticker
        for ticker, info in load_constituents().items()
        if info["sector"] == sector
    )


def get_all_tickers() -> list[str]:
    """Return sorted list of all S&P 500 tickers."""
    return sorted(load_constituents().keys())
