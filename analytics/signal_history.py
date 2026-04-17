"""Persistent signal snapshots + cross-day delta classification + watchlist storage.

Phase 2 of the market-dashboard improvements: turns the dashboard from a
point-in-time view into a workflow. Each refresh appends a row-per-metric
snapshot to ``signal_history.parquet``; the watchlist is stored in
``watchlist.json``. Both files live in ``MM_CACHE_DIR`` and are git-tracked
exceptions so they ship with Render deploys.

Storage format (signal_history.parquet, long format):
    date     - pd.Timestamp (date-only, normalized)
    key      - str (ticker symbol OR factor name)
    metric   - str (one of: composite, rsi, ma_distance_50d,
                    rolling_max_252d, rolling_min_252d, factor_trend)
    value    - float (NaN for categorical rows like factor_trend)
    label    - str ("" for numeric rows; trend label for factor_trend)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.factor_monitor import classify_factor_trends
from analytics.reversion import compute_reversion_signals
from config import MM_CACHE_DIR

_DEFAULT_STORE_DIR = Path(MM_CACHE_DIR)
_HISTORY_FILE = "signal_history.parquet"
_WATCHLIST_FILE = "watchlist.json"

# Categorizer thresholds — kept in-module rather than in config.py because they
# are only meaningful in the context of the delta categorizer below.
_COMPOSITE_OVERSOLD_CUT = 25.0
_RSI_OVERBOUGHT = 70.0
_RSI_OVERSOLD = 30.0


@dataclass
class SignalSnapshot:
    """One day's signal universe — what gets persisted on each refresh."""

    date: pd.Timestamp
    composite: pd.Series  # ticker -> 0-100, lower = oversold
    rsi: pd.Series  # ticker -> 0-100
    ma_distance_50d: pd.Series  # ticker -> % distance from 50d MA
    rolling_max_252d: pd.Series  # ticker -> 252d rolling-max price
    rolling_min_252d: pd.Series
    last_price: pd.Series
    factor_trends: pd.Series | None = None  # factor name -> "Trending Up"/Neutral/Mean Reverting


@dataclass
class SignalDelta:
    """A single classified change between two snapshot dates."""

    ticker: str  # the affected ticker (or factor name for factor_trend)
    metric: str
    prior: float | str | None
    current: float | str | None
    delta: float | None
    category: str  # human-readable signal-change label


@dataclass
class WatchlistEntry:
    ticker: str
    added_on: str  # ISO timestamp
    note: str = ""
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------

def build_snapshot(
    prices: pd.DataFrame,
    factor_returns: pd.DataFrame | None = None,
) -> SignalSnapshot:
    """Compute a fresh ``SignalSnapshot`` from current prices.

    Args:
        prices: Wide-format prices (DatetimeIndex x ticker columns).
        factor_returns: Optional factor returns (used for factor-trend deltas).
    """
    rev = compute_reversion_signals(prices)

    last = prices.iloc[-1]
    rolling_max = prices.rolling(252, min_periods=20).max().iloc[-1]
    rolling_min = prices.rolling(252, min_periods=20).min().iloc[-1]

    factor_trends = None
    if factor_returns is not None and not factor_returns.empty:
        trends_df = classify_factor_trends(factor_returns)
        if not trends_df.empty and "Trend" in trends_df.columns:
            factor_trends = trends_df["Trend"]

    return SignalSnapshot(
        date=pd.Timestamp(prices.index[-1]).normalize(),
        composite=rev.composite_score,
        rsi=rev.rsi_14,
        ma_distance_50d=rev.ma_distance_50d,
        rolling_max_252d=rolling_max,
        rolling_min_252d=rolling_min,
        last_price=last,
        factor_trends=factor_trends,
    )


# ---------------------------------------------------------------------------
# On-disk persistence
# ---------------------------------------------------------------------------

def _store_path(filename: str, store_dir: Path | None) -> Path:
    return (store_dir or _DEFAULT_STORE_DIR) / filename


def _snapshot_to_long(snapshot: SignalSnapshot) -> pd.DataFrame:
    """Flatten a SignalSnapshot to long format for Parquet storage."""
    frames: list[pd.DataFrame] = []

    def _numeric(series: pd.Series, metric: str) -> pd.DataFrame:
        s = series.dropna()
        if s.empty:
            return pd.DataFrame(columns=["date", "key", "metric", "value", "label"])
        return pd.DataFrame({
            "date": snapshot.date,
            "key": s.index.astype(str),
            "metric": metric,
            "value": s.astype(float).values,
            "label": "",
        })

    frames.append(_numeric(snapshot.composite, "composite"))
    frames.append(_numeric(snapshot.rsi, "rsi"))
    frames.append(_numeric(snapshot.ma_distance_50d, "ma_distance_50d"))
    frames.append(_numeric(snapshot.rolling_max_252d, "rolling_max_252d"))
    frames.append(_numeric(snapshot.rolling_min_252d, "rolling_min_252d"))
    frames.append(_numeric(snapshot.last_price, "last_price"))

    if snapshot.factor_trends is not None:
        s = snapshot.factor_trends.dropna()
        if not s.empty:
            frames.append(pd.DataFrame({
                "date": snapshot.date,
                "key": s.index.astype(str),
                "metric": "factor_trend",
                "value": np.nan,
                "label": s.astype(str).values,
            }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "key", "metric", "value", "label"]
    )


def append_snapshot(
    snapshot: SignalSnapshot,
    *,
    store_dir: Path | None = None,
) -> None:
    """Persist a snapshot. Idempotent: re-writing the same date overwrites."""
    path = _store_path(_HISTORY_FILE, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = _snapshot_to_long(snapshot)
    if new_rows.empty:
        return

    if path.exists():
        existing = pd.read_parquet(path)
        # Drop any prior rows for the same date so re-runs are idempotent.
        existing = existing[existing["date"] != snapshot.date]
        combined = (
            new_rows if existing.empty
            else pd.concat([existing, new_rows], ignore_index=True)
        )
    else:
        combined = new_rows

    combined.sort_values(["date", "metric", "key"], inplace=True)
    combined.to_parquet(path, index=False)


def load_history(
    *,
    lookback_days: int = 30,
    store_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the long-format history, optionally limited to the most recent N
    distinct snapshot dates."""
    path = _store_path(_HISTORY_FILE, store_dir)
    if not path.exists():
        return pd.DataFrame(columns=["date", "key", "metric", "value", "label"])

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    if lookback_days:
        recent_dates = sorted(df["date"].unique())[-lookback_days:]
        df = df[df["date"].isin(recent_dates)]
    return df.reset_index(drop=True)


def available_dates(*, store_dir: Path | None = None) -> list[pd.Timestamp]:
    """Distinct snapshot dates available, ascending."""
    path = _store_path(_HISTORY_FILE, store_dir)
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=["date"])
    return sorted(pd.to_datetime(df["date"]).unique())


# ---------------------------------------------------------------------------
# Delta classification
# ---------------------------------------------------------------------------

def _wide_for_metric(history: pd.DataFrame, metric: str, value_col: str = "value") -> pd.DataFrame:
    """Pivot a long history slice to date x key wide form for one metric."""
    sliced = history[history["metric"] == metric]
    if sliced.empty:
        return pd.DataFrame()
    return sliced.pivot_table(index="date", columns="key", values=value_col, aggfunc="last")


def compute_deltas(
    history: pd.DataFrame,
    current_date: pd.Timestamp,
    prior_date: pd.Timestamp,
) -> list[SignalDelta]:
    """Classify signal changes between two dates.

    Categories produced:
        - "Entered Oversold" / "Exited Oversold" (composite crosses 25)
        - "RSI Flip <30" / "RSI Flip >70" (RSI crosses 30 / 70)
        - "Broke 50d MA" / "Reclaimed 50d MA" (ma_distance_50d sign flip)
        - "New 52w High" / "New 52w Low" (last_price hits rolling extreme)
        - "Factor Trend Flip" (factor_trend label change)
    """
    if history.empty:
        return []

    current_date = pd.Timestamp(current_date).normalize()
    prior_date = pd.Timestamp(prior_date).normalize()

    deltas: list[SignalDelta] = []

    def _numeric_pair(metric: str) -> pd.DataFrame:
        wide = _wide_for_metric(history, metric)
        if wide.empty or current_date not in wide.index or prior_date not in wide.index:
            return pd.DataFrame()
        return pd.DataFrame({
            "prior": wide.loc[prior_date],
            "current": wide.loc[current_date],
        }).dropna(how="any")

    # Composite crosses 25 (oversold band)
    comp = _numeric_pair("composite")
    for tk, row in comp.iterrows():
        prior, cur = row["prior"], row["current"]
        if prior >= _COMPOSITE_OVERSOLD_CUT and cur < _COMPOSITE_OVERSOLD_CUT:
            deltas.append(SignalDelta(tk, "composite", prior, cur, cur - prior, "Entered Oversold"))
        elif prior < _COMPOSITE_OVERSOLD_CUT and cur >= _COMPOSITE_OVERSOLD_CUT:
            deltas.append(SignalDelta(tk, "composite", prior, cur, cur - prior, "Exited Oversold"))

    # RSI crosses 30 / 70
    rsi = _numeric_pair("rsi")
    for tk, row in rsi.iterrows():
        prior, cur = row["prior"], row["current"]
        if prior >= _RSI_OVERSOLD and cur < _RSI_OVERSOLD:
            deltas.append(SignalDelta(tk, "rsi", prior, cur, cur - prior, "RSI Flip <30"))
        elif prior <= _RSI_OVERBOUGHT and cur > _RSI_OVERBOUGHT:
            deltas.append(SignalDelta(tk, "rsi", prior, cur, cur - prior, "RSI Flip >70"))

    # 50d MA sign flip
    ma = _numeric_pair("ma_distance_50d")
    for tk, row in ma.iterrows():
        prior, cur = row["prior"], row["current"]
        if prior >= 0 and cur < 0:
            deltas.append(SignalDelta(tk, "ma_distance_50d", prior, cur, cur - prior, "Broke 50d MA"))
        elif prior < 0 and cur >= 0:
            deltas.append(SignalDelta(tk, "ma_distance_50d", prior, cur, cur - prior, "Reclaimed 50d MA"))

    # New 52w highs / lows: today's last_price exceeds the prior-day rolling extreme
    last = _numeric_pair("last_price")
    hi = _numeric_pair("rolling_max_252d")
    lo = _numeric_pair("rolling_min_252d")
    if not last.empty:
        for tk, row in last.iterrows():
            if tk in hi.index and row["current"] >= hi.loc[tk, "prior"] * 0.999:
                deltas.append(SignalDelta(
                    tk, "last_price", row["prior"], row["current"], None, "New 52w High",
                ))
            elif tk in lo.index and row["current"] <= lo.loc[tk, "prior"] * 1.001:
                deltas.append(SignalDelta(
                    tk, "last_price", row["prior"], row["current"], None, "New 52w Low",
                ))

    # Factor trend label flips
    trend = history[history["metric"] == "factor_trend"]
    if not trend.empty:
        wide = trend.pivot_table(
            index="date", columns="key", values="label", aggfunc="last",
        )
        if current_date in wide.index and prior_date in wide.index:
            for factor in wide.columns:
                p, c = wide.loc[prior_date, factor], wide.loc[current_date, factor]
                if isinstance(p, str) and isinstance(c, str) and p != c:
                    deltas.append(SignalDelta(
                        factor, "factor_trend", p, c, None, "Factor Trend Flip",
                    ))

    return deltas


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def load_watchlist(*, store_dir: Path | None = None) -> list[WatchlistEntry]:
    path = _store_path(_WATCHLIST_FILE, store_dir)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [WatchlistEntry(**e) for e in raw]


def save_watchlist(
    entries: list[WatchlistEntry],
    *,
    store_dir: Path | None = None,
) -> None:
    path = _store_path(_WATCHLIST_FILE, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(e) for e in entries]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def add_ticker(
    ticker: str,
    *,
    note: str = "",
    tags: list[str] | None = None,
    store_dir: Path | None = None,
) -> None:
    """Idempotent: re-adding an existing ticker updates its note/tags only."""
    entries = load_watchlist(store_dir=store_dir)
    existing = next((e for e in entries if e.ticker == ticker), None)
    if existing is not None:
        if note:
            existing.note = note
        if tags is not None:
            existing.tags = list(tags)
    else:
        entries.append(WatchlistEntry(
            ticker=ticker,
            added_on=datetime.now(timezone.utc).isoformat(),
            note=note,
            tags=list(tags) if tags else [],
        ))
    save_watchlist(entries, store_dir=store_dir)


def remove_ticker(ticker: str, *, store_dir: Path | None = None) -> None:
    entries = [e for e in load_watchlist(store_dir=store_dir) if e.ticker != ticker]
    save_watchlist(entries, store_dir=store_dir)
