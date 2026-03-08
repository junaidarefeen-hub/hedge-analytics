import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram

from ui.style import PLOTLY_LAYOUT

# Correlation: teal-white diverging, light enough that values stay readable
_CORR_COLORSCALE = [
    [0.0, "#0d9488"],   # -1.0  teal (strong negative)
    [0.3, "#5eead4"],   # -0.4  light teal
    [0.5, "#f0fdfa"],   #  0.0  near-white
    [0.7, "#c4b5fd"],   # +0.4  light violet
    [1.0, "#6d28d9"],   # +1.0  deep violet (strong positive)
]

# Beta: soft green → white → indigo
_BETA_COLORSCALE = [
    [0.0, "#059669"],   # low beta
    [0.35, "#6ee7b7"],
    [0.5, "#f0fdf4"],   # ~1.0 center
    [0.7, "#a5b4fc"],
    [1.0, "#4338ca"],   # high beta
]


def _add_annotations(fig, df: pd.DataFrame, colorscale_dark_ranges: list[tuple[float, float]], zmin: float, zmax: float):
    """Add per-cell text annotations with adaptive black/white font color."""
    annotations = []
    for i, row_label in enumerate(df.index):
        for j, col_label in enumerate(df.columns):
            val = df.iloc[i, j]
            if pd.isna(val):
                continue
            # Normalize value to 0-1 scale position
            norm = (val - zmin) / (zmax - zmin) if zmax != zmin else 0.5
            norm = max(0, min(1, norm))
            # Use white text if the cell falls in a dark range
            use_white = any(lo <= norm <= hi for lo, hi in colorscale_dark_ranges)
            annotations.append(dict(
                x=col_label, y=row_label,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(size=12, color="#ffffff" if use_white else "#1e293b"),
            ))
    fig.update_layout(annotations=annotations)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Create a Plotly heatmap for a correlation matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=_CORR_COLORSCALE,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Corr", thickness=14, len=0.8),
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Correlation Matrix",
        height=max(420, 56 * len(corr)),
        xaxis=dict(side="bottom", tickangle=-45, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
    )
    # Dark ranges: strong negative (norm 0–0.15) and strong positive (norm 0.85–1.0)
    _add_annotations(fig, corr, [(0, 0.15), (0.85, 1.0)], -1, 1)
    return fig


def beta_heatmap(betas: pd.DataFrame) -> go.Figure:
    """Create a Plotly heatmap for a beta matrix (tickers x benchmarks)."""
    vals = betas.values.astype(float)
    zmin = float(np.nanmin(vals)) if not np.all(np.isnan(vals)) else 0
    zmax = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else 2

    fig = go.Figure(
        data=go.Heatmap(
            z=vals,
            x=betas.columns.tolist(),
            y=betas.index.tolist(),
            colorscale=_BETA_COLORSCALE,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            colorbar=dict(title="Beta", thickness=14, len=0.8),
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Beta Matrix",
        height=max(420, 56 * len(betas)),
        xaxis=dict(title="Benchmark", side="bottom", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Ticker", autorange="reversed", gridcolor="rgba(0,0,0,0)"),
    )
    # Dark ranges: low end (norm 0–0.15) and high end (norm 0.85–1.0)
    _add_annotations(fig, betas, [(0, 0.15), (0.85, 1.0)], zmin, zmax)
    return fig


def correlation_dendrogram(clustering_result: dict) -> go.Figure:
    """Render a dendrogram from correlation clustering using Plotly."""
    Z = clustering_result["linkage_matrix"]
    labels = clustering_result["labels"]

    # Use scipy dendrogram to get coordinates (no_plot=True)
    ddata = dendrogram(Z, labels=labels, no_plot=True)

    fig = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#2563eb", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Correlation Dendrogram (distance = 1 - |corr|)",
        xaxis=dict(
            tickvals=list(range(5, 10 * len(labels), 10)),
            ticktext=ddata["ivl"],
            tickangle=-45,
        ),
        yaxis_title="Distance",
        height=380,
        showlegend=False,
    )
    return fig
