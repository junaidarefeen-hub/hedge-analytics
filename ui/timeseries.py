import pandas as pd
import plotly.graph_objects as go

from ui.style import PLOTLY_LAYOUT


def rolling_correlation_chart(
    series: pd.Series, ticker_a: str, ticker_b: str, window: int
) -> go.Figure:
    """Line chart for rolling pairwise correlation."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=f"{ticker_a} / {ticker_b}",
            line=dict(width=2, color="#2563eb"),
            hovertemplate="%{x|%b %d, %Y}<br>Corr: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{window}-Day Rolling Correlation: {ticker_a} vs {ticker_b}",
        xaxis_title="",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#e2e8f0"),
        height=400,
        showlegend=False,
    )
    return fig


def rolling_beta_chart(
    series: pd.Series, ticker: str, benchmark: str, window: int
) -> go.Figure:
    """Line chart for rolling beta."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=f"{ticker} vs {benchmark}",
            line=dict(width=2, color="#7c3aed"),
            hovertemplate="%{x|%b %d, %Y}<br>Beta: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title=f"{window}-Day Rolling Beta: {ticker} vs {benchmark}",
        xaxis_title="",
        yaxis_title="Beta",
        height=400,
        showlegend=False,
    )
    return fig
