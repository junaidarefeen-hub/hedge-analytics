import re
from datetime import date


def parse_tickers(raw: str) -> list[str]:
    """Parse comma/space/newline-separated ticker string into cleaned uppercase list."""
    tickers = re.split(r"[,\s\n]+", raw.strip())
    seen = set()
    result = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def validate_date_range(start: date, end: date) -> str | None:
    """Return error message if date range is invalid, else None."""
    if start >= end:
        return "Start date must be before end date."
    if (end - start).days < 30:
        return "Date range must be at least 30 days."
    return None


def check_data_sufficiency(n_rows: int, window: int) -> str | None:
    """Return warning if data is short relative to rolling window, else None."""
    if n_rows < window:
        return f"Not enough data ({n_rows} rows) for rolling window of {window}. No rolling results will be shown."
    if n_rows < 2 * window:
        return f"Data length ({n_rows}) is less than 2× the rolling window ({window}). Rolling results may be thin."
    return None
