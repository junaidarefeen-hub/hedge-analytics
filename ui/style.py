import streamlit as st

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=13, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    title_font=dict(size=16, color="#0f172a"),
    margin=dict(l=60, r=24, t=52, b=48),
    xaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        zerolinecolor="#e2e8f0",
    ),
    yaxis=dict(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        zerolinecolor="#e2e8f0",
    ),
    colorway=[
        "#2563eb", "#0891b2", "#7c3aed", "#db2777",
        "#ea580c", "#16a34a", "#ca8a04", "#64748b",
    ],
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#e2e8f0",
        font_size=12,
        font_color="#334155",
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font_size=12,
    ),
)


METRIC_DESCRIPTIONS = {
    # Backtest / compare metrics
    "Total Return": "Cumulative return over the full period: (final value / initial value) - 1.",
    "Ann. Return": "Annualized compound return, scaled to a 252-day trading year.",
    "Ann. Volatility": "Annualized standard deviation of daily returns (daily std x sqrt(252)).",
    "Sharpe Ratio": "Risk-adjusted return: (mean excess return over risk-free) / volatility, annualized.",
    "Sortino Ratio": "Like Sharpe but only penalizes downside volatility (negative returns).",
    "Max Drawdown": "Largest peak-to-trough decline as a percentage of the peak value.",
    "Calmar Ratio": "Annualized return divided by the absolute value of max drawdown.",
    "Omega Ratio": "Ratio of cumulative gains to cumulative losses relative to a zero threshold. >1 means more gains than losses.",
    "Max DD Duration (days)": "Longest period (in trading days) spent below a previous high-water mark.",
    "Tracking Error": "Annualized std of return differences between hedged and unhedged portfolios. Measures hedge deviation.",
    "Information Ratio": "Annualized mean return difference / tracking error. Measures excess return per unit of deviation from the unhedged portfolio.",
    "Vol Reduction": "Percentage reduction in annualized volatility from hedging: 1 - (hedged vol / unhedged vol).",
    # Monte Carlo metrics
    "Mean Return": "Average simulated return across all paths at the end of the horizon.",
    "Median Return": "Middle simulated return (50th percentile) — less affected by outliers than the mean.",
    "Std Dev": "Standard deviation of simulated final returns across all paths.",
    "Best Case": "Highest simulated return across all paths.",
    "Worst Case": "Lowest simulated return across all paths.",
    "VaR 90%": "Value at Risk at 90% confidence: loss exceeded in only 10% of simulations.",
    "VaR 95%": "Value at Risk at 95% confidence: loss exceeded in only 5% of simulations.",
    "VaR 99%": "Value at Risk at 99% confidence: loss exceeded in only 1% of simulations.",
    "CVaR 90%": "Conditional VaR (Expected Shortfall) at 90%: average loss in the worst 10% of simulations.",
    "CVaR 95%": "Conditional VaR (Expected Shortfall) at 95%: average loss in the worst 5% of simulations.",
    "CVaR 99%": "Conditional VaR (Expected Shortfall) at 99%: average loss in the worst 1% of simulations.",
    "P(loss > 5%)": "Probability of losing more than 5% of portfolio value.",
    "P(loss > 10%)": "Probability of losing more than 10% of portfolio value.",
    "P(loss > 20%)": "Probability of losing more than 20% of portfolio value.",
    # Price performance metrics
    "Excess Return (vs Index)": "Cumulative return minus benchmark index return over the period.",
    "Ann. Excess Return (vs Index)": "Annualized excess return over the benchmark index.",
    "Tracking Error (vs Index)": "Annualized std of daily return differences vs the benchmark index.",
    "Information Ratio (vs Index)": "Annualized excess return / tracking error vs the benchmark index.",
    "Excess Return (vs Peers)": "Cumulative return minus peer group basket return over the period.",
    "Ann. Excess Return (vs Peers)": "Annualized excess return over the peer group basket.",
    "Tracking Error (vs Peers)": "Annualized std of daily return differences vs the peer group.",
    "Information Ratio (vs Peers)": "Annualized excess return / tracking error vs the peer group.",
    "Beta": "Sensitivity to benchmark: Cov(ticker, benchmark) / Var(benchmark).",
    "Beta-Adj Return": "Cumulative return after removing beta exposure: ticker - beta x benchmark.",
    "Ann. Alpha": "Annualized beta-adjusted return — the return not explained by market exposure.",
    "Residual Volatility": "Annualized volatility of beta-adjusted (residual) returns.",
}


def add_metric_descriptions(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add a 'Description' column to a metrics DataFrame based on its index.

    DEPRECATED: use render_metrics_table() for hover-based tooltips instead.
    """
    import pandas as pd
    out = df.copy()
    out.insert(0, "Description", [METRIC_DESCRIPTIONS.get(idx, "") for idx in out.index])
    return out


def render_metrics_table(df: "pd.DataFrame") -> None:
    """Render a metrics DataFrame as an HTML table with hover tooltips on metric names.

    Uses st.html() to avoid Streamlit's HTML sanitization which strips title attributes.
    Each metric name cell shows a styled tooltip on hover via CSS.
    """
    import html as html_mod

    rows_html = []
    for idx in df.index:
        desc = METRIC_DESCRIPTIONS.get(idx, "")
        escaped_idx = html_mod.escape(str(idx))
        escaped_desc = html_mod.escape(desc)
        cells = "".join(
            f'<td class="mt-val">{html_mod.escape(str(df.loc[idx, col]))}</td>'
            for col in df.columns
        )
        if desc:
            metric_cell = (
                f'<td class="mt-metric" title="{escaped_desc}">'
                f'<span class="mt-tip" data-tip="{escaped_desc}">{escaped_idx}</span></td>'
            )
        else:
            metric_cell = f'<td class="mt-metric">{escaped_idx}</td>'
        rows_html.append(f'<tr>{metric_cell}{cells}</tr>')

    header_cells = "".join(
        f'<th class="mt-hdr mt-hdr-val">{html_mod.escape(str(col))}</th>'
        for col in df.columns
    )

    table_html = f"""
    <style>
    .mt-wrap {{ overflow-x:auto; border:1px solid #e2e8f0; border-radius:8px; }}
    .mt-wrap table {{ width:100%; border-collapse:collapse; font-family:Inter,system-ui,sans-serif; }}
    .mt-hdr {{ padding:8px 12px; border-bottom:2px solid #cbd5e1; font-size:13px;
               font-weight:600; color:#475569; text-align:left; }}
    .mt-hdr-val {{ text-align:right; }}
    .mt-metric {{ padding:6px 12px; border-bottom:1px solid #e2e8f0; font-weight:500;
                  font-size:13px; position:relative; }}
    .mt-val {{ padding:6px 12px; border-bottom:1px solid #e2e8f0; text-align:right; font-size:13px; }}
    .mt-tip {{ cursor:help; text-decoration:underline dotted #94a3b8;
               text-underline-offset:3px; position:relative; display:inline-block; }}
    .mt-tip::after {{
        content: attr(data-tip);
        position: absolute; left: 0; bottom: 100%; margin-bottom: 6px;
        background: #1e293b; color: #f8fafc; padding: 6px 10px;
        border-radius: 6px; font-size: 12px; font-weight: 400;
        line-height: 1.4; white-space: normal; width: max-content; max-width: 320px;
        opacity: 0; pointer-events: none; transition: opacity 0.15s; z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .mt-tip:hover::after {{ opacity: 1; }}
    .mt-hint {{ font-size:11px; color:#94a3b8; margin-top:4px; padding-left:4px; }}
    </style>
    <div class="mt-wrap">
    <table>
        <thead><tr><th class="mt-hdr">Metric</th>{header_cells}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
    </table>
    </div>
    <p class="mt-hint">Hover over metric names for descriptions.</p>
    """
    st.html(table_html)


def inject_css():
    st.markdown(
        """
        <style>
        /* Clean up main area spacing */
        .block-container { padding-top: 2rem; padding-bottom: 1rem; }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 2px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.6rem 1.4rem;
            font-weight: 500;
            color: #64748b;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }
        .stTabs [aria-selected="true"] {
            color: #2563eb;
            border-bottom: 2px solid #2563eb;
        }

        /* Sidebar polish */
        section[data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

        /* Dataframe styling */
        .stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }

        /* Metric / header spacing */
        h1 { font-weight: 700; letter-spacing: -0.02em; color: #0f172a; }
        h2, h3 { font-weight: 600; color: #1e293b; }

        /* Remove default Streamlit branding footer */
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )
