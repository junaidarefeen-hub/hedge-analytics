from datetime import date, timedelta

DEFAULT_STOCK_TICKERS = "AAPL, MSFT, GOOGL, AMZN, META"
DEFAULT_FACTOR_TICKERS = "SPY"
DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_START_DATE = date.today() - timedelta(days=365 * DEFAULT_LOOKBACK_YEARS)
DEFAULT_END_DATE = date.today()
ROLLING_WINDOW_OPTIONS = [30, 60, 90, 120, 252]
DEFAULT_ROLLING_WINDOW = 60
RETURN_METHODS = ["log", "simple"]
DEFAULT_RETURN_METHOD = "simple"
CACHE_TTL_SECONDS = 3600

# Hedge optimizer
STRATEGY_OPTIONS = [
    "Minimum Variance",
    "Beta-Neutral",
    "Tail Risk (CVaR)",
    "Risk Parity",
]
DEFAULT_STRATEGY = "Minimum Variance"
DEFAULT_NOTIONAL = 10_000_000.0
DEFAULT_WEIGHT_BOUNDS = (-1.0, 0.0)  # short-only hedging
DEFAULT_MAX_GROSS_RATIO = 1.0  # max hedge notional as fraction of target notional
DEFAULT_CVAR_CONFIDENCE = 0.95
DEFAULT_MIN_HEDGE_NAMES = 3
ANNUALIZATION_FACTOR = 252

# Backtest
DEFAULT_ROLLING_VOL_WINDOW = 60
DEFAULT_RISK_FREE_RATE = 0.0

# Monte Carlo simulation
MC_DEFAULT_HORIZON = 60
MC_HORIZON_OPTIONS = [30, 60, 90, 120, 252]
MC_DEFAULT_NUM_SIMS = 5000
MC_NUM_SIMS_OPTIONS = [1000, 5000, 10000]
MC_PERCENTILE_BANDS = [5, 25, 50, 75, 95]
MC_SPAGHETTI_PATHS = 50
MC_LOSS_THRESHOLDS = [0.05, 0.10]
MC_DEFAULT_SEED = 42

# Stress Testing / Scenario Analysis
HISTORICAL_SCENARIOS = [
    {
        "name": "2008 GFC (Lehman Collapse)",
        "start": "2008-09-01",
        "end": "2009-03-09",
        "description": "Global financial crisis: Lehman Brothers bankruptcy through market bottom",
    },
    {
        "name": "2011 EU Debt Crisis",
        "start": "2011-07-22",
        "end": "2011-10-03",
        "description": "European sovereign debt contagion fears; US downgrade by S&P",
    },
    {
        "name": "2015 China Devaluation",
        "start": "2015-08-10",
        "end": "2015-09-30",
        "description": "PBoC yuan devaluation triggers global equity selloff",
    },
    {
        "name": "2018 Q4 Selloff",
        "start": "2018-10-01",
        "end": "2018-12-24",
        "description": "Fed tightening + trade war fears; S&P 500 fell ~20% from peak",
    },
    {
        "name": "2020 COVID Crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "description": "COVID-19 pandemic onset; fastest 30% drawdown in history",
    },
    {
        "name": "2022 Rate Hike Selloff",
        "start": "2022-01-03",
        "end": "2022-06-16",
        "description": "Aggressive Fed rate hikes; tech-heavy selloff, S&P down ~24%",
    },
    {
        "name": "2023 Banking Crisis",
        "start": "2023-03-08",
        "end": "2023-03-24",
        "description": "SVB + Signature Bank failures; regional banking contagion",
    },
]
STRESS_CUSTOM_SHOCK_RANGE = (-50.0, 50.0)

# Data interval / cache controls
INTERVAL_OPTIONS = ["1d", "1h", "15m", "5m", "1m"]
INTRADAY_MAX_DAYS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "1h": 730,
    "1d": 100000,  # effectively unlimited
}

# Regime detection
REGIME_VOL_WINDOW = 60
REGIME_N_REGIMES = 3
REGIME_LABELS = {0: "Low Vol", 1: "Normal", 2: "High Vol"}
REGIME_METHODS = ["quantile", "kmeans"]

# Correlation regime detection
CORR_REGIME_N_REGIMES = 3
CORR_REGIME_LABELS = {0: "Low Corr", 1: "Normal Corr", 2: "High Corr"}

# Rolling optimization
ROLLING_OPT_WINDOW = 120
ROLLING_OPT_STEP = 20

# Custom Hedge Analyzer
CHA_DEFAULT_LONG_NOTIONAL = 10_000_000.0
CHA_DEFAULT_HEDGE_NOTIONAL = 10_000_000.0

# Dynamic rebalancing backtest
REBALANCE_FREQUENCIES = ["weekly", "monthly", "quarterly"]
DEFAULT_REBALANCE_FREQUENCY = "monthly"
DEFAULT_LOOKBACK_WINDOW = 120

# ECM MCP Server
ECM_MCP_URL = "https://mcp.elementcapital.corp/mcp"
ECM_CREDENTIALS_KEY = "ecm|72c3fb7b2b25e5da"
ECM_TOKEN_BUFFER_SECONDS = 10
ECM_REQUEST_TIMEOUT = 60

# Prismatic Factor Gadgets (Quant Equity Dashboard, layout 10052)
FACTOR_GADGET_IDS = {
    "10685": "Equal Weight",
    "10686": "Momentum",
    "10687": "Size",
    "10688": "Quality",
    "10689": "Volatility",
    "10690": "Value",
    "10691": "Growth",
    "10692": "Beta",
    "10693": "Leverage",
}
FACTOR_MODEL_PREFIX = "GS"  # Use GS variant by default (vs MS)
FACTOR_CACHE_MAX_AGE_HOURS = 24  # local Parquet cache refresh interval

# Factor Analytics
FA_DEFAULT_LONG_NOTIONAL = 10_000_000.0
FA_DEFAULT_SHORT_NOTIONAL = 10_000_000.0
FA_DEFAULT_P_THRESHOLD = 0.05
FA_SIGNIFICANCE_LEVELS = {0.01: "***", 0.05: "**", 0.10: "*"}

# Default GS factor selection for the per-ticker exposure tab. Excludes
# Equal Weight (overlaps with the market regressor) and Beta (double-counts
# market exposure when paired with the market regressor).
FA_DEFAULT_PER_TICKER_FACTORS = [
    "Momentum", "Size", "Quality", "Volatility", "Value", "Growth", "Leverage",
]
# ---------------------------------------------------------------------------
# Market Monitor
# ---------------------------------------------------------------------------
MM_CACHE_DIR = "data/cache/market_monitor"
MM_CACHE_MAX_AGE_HOURS = 12  # warn if data older than this
MM_DEFAULT_LOOKBACK_DAYS = 3650  # 10 years of history for full rebuild
MM_RVX_MAX_WORKERS = 15  # parallel RVX fetch threads

# Multi-period return windows (trading days)
MM_PERIODS = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "YTD": None,  # computed dynamically
    "1Y": 252,
}

# Reversion signal defaults
MM_RSI_WINDOW = 14
MM_ZSCORE_WINDOWS = [20, 60]
MM_MA_WINDOWS = [50, 200]
MM_BOLLINGER_WINDOW = 20
MM_BOLLINGER_STD = 2.0
MM_COMPOSITE_WEIGHTS = {
    "rsi_14": 0.30,
    "zscore_20d": 0.25,
    "zscore_60d": 0.15,
    "ma_distance_50d": 0.15,
    "bollinger_pctb": 0.15,
}

# Factor monitor
MM_FACTOR_BETA_WINDOW = 120  # trailing-window fallback (used by callers that don't pass explicit dates, e.g. trade_ideas_tab)
MM_FACTOR_TREND_SHORT = 20  # short MA for trend detection
MM_FACTOR_TREND_LONG = 60  # long MA for trend detection

# Trade screener
MM_REVERSION_THRESHOLD = 25  # composite score <= this = oversold candidate
MM_FACTOR_BETA_THRESHOLD = 1.0  # min absolute beta to qualify as factor play
