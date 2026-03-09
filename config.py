from datetime import date, timedelta

DEFAULT_STOCK_TICKERS = "AAPL, MSFT, GOOGL, AMZN, META"
DEFAULT_FACTOR_TICKERS = "SPY, QQQ, IWM, XLF, XLE"
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

# Rolling optimization
ROLLING_OPT_WINDOW = 120
ROLLING_OPT_STEP = 20

# Dynamic rebalancing backtest
REBALANCE_FREQUENCIES = ["weekly", "monthly", "quarterly"]
DEFAULT_REBALANCE_FREQUENCY = "monthly"
DEFAULT_LOOKBACK_WINDOW = 120
