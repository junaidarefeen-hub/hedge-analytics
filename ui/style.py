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
