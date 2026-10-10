"""Parameters of the study, each declared here once."""

# Sample ----------------------------------------------------------------------
END = "2026-08-31"
BACKTEST_START = "2021-09-01"
REFITS = ("2021-09-01", "2022-09-01", "2023-09-01", "2024-09-01", "2025-09-01")
HORIZON = 21  # trading days in a month: equations (3), (5) and (12)

# Data cleaning ---------------------------------------------------------------
DROPPED = {
    "0011.HK": "no Yahoo data, delisted 2026",
    "0203.HK": "no Yahoo data, delisted",
    "0494.HK": "no Yahoo data, delisted 2020",
    "0906.HK": "no Yahoo data, delisted 2008",
    "0013.HK": "code reused, Yahoo returns HUTCHMED from 2021",
    "0020.HK": "code reused, Yahoo returns SenseTime from 2021",
    "1880.HK": "code reused, Yahoo returns China Tourism Group Duty Free from 2022",
}
YAHOO_ERRORS = (("0004.HK", "2017-11-16"),)  # spin-off adjustment leaves a false +15.7%
USDHKD_BAND = (7.7, 7.9)                     # quotes outside are data errors

# Covariance and beta ---------------------------------------------------------
WINDOW = 252  # daily returns before each rebalance
K_MAX = 4     # most factors f_k of equation (10)

# LSTM-VAE --------------------------------------------------------------------
SEQ_LEN = 240            # L
HIDDEN = 32
LAYERS = 2
DROPOUT = 0.25
LATENT = 2               # d, also the number of PCA components
LATENT_M3 = 3            # d of the VAE of M3
LR = 1e-3
BATCH = 256
MAX_EPOCHS = 300
PATIENCE = 10
CLIP = 1.0               # gradient norm
VALID_DAYS = 252         # final twelve months before each refit
ACTIVE_THRESHOLD = 0.01  # active unit: posterior-mean variance above this, Table A.2
SEED = 20260920

# Stock portfolios ------------------------------------------------------------
WEIGHT_CAP = 0.03
COST_BPS = 15.0  # one way
BL_DELTA = 2.5   # delta of equation (5)
BL_ZETA = 0.05   # zeta of equation (5); cancels from (6)

# Contracts and costs ---------------------------------------------------------
FUND_USD = 100e6
MULTIPLIER = 50            # HK$ per HSI point
IM_HKD = 117_705           # HKCC initial margin per HSI futures contract from 2 Mar 2026
IM_DATE = "2026-03-02"     # S_M of equation (13) is this close
STRIKE_SWITCH = 20_000     # STRIKE_STEP_LOW below, STRIKE_STEP_HIGH at or above
STRIKE_STEP_LOW = 100
STRIKE_STEP_HIGH = 200
SPREAD_MIN = 30            # psi of equation (14), index points
SPREAD_PCT = 0.10
SPREAD_LIMIT = 750
SPREAD_ABOVE = 75
OPTION_FEES = 10.54        # HK$ per contract per side
FUTURES_FEES = 10.54       # HK$ per contract per side
FUTURES_HALF_SPREAD = 1.0  # index points, one tick
DIV_YIELD = 0.03           # q

# Derivatives strategies ------------------------------------------------------
CALL_MONEYNESS = 1.04       # P1 strike over S, rounded up to a listed strike
PUT_LONG = 0.95             # K1 over S, rounded down to a listed strike
PUT_SHORT = 0.90            # K2 over S, rounded down to a listed strike
CALL_BAND = (1.05, 1.10)    # K3 over S: zero net premium strike held in the band, rounded up
VHSI_ON = 30.0              # P2 on above
VHSI_OFF = 22.0             # P2 off below
DELTA_BUDGET = 0.15         # cap on P2 |delta| at entry, share of NAV
DD_ON, DD_OFF = 0.05, 0.02  # drawdown on and off, "P2, full trigger" only
COVERAGE = 0.5              # h, base case
COVERAGE_FULL = 1.0         # h of Appendix E
SCENARIOS = ("None", "P1", "P2", "P1 and P2")  # P1 yield, P2 protection
VARIANTS = ("P1 unhedged", "P2, full trigger")
SHIFT_MAX = 5.0             # vol points, upper end of the break-even search
