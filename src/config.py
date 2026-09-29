"""Every parameter of the study, declared once."""

# sample
END = "2026-08-31"
BACKTEST_START = "2021-09-01"
REFITS = ("2021-09-01", "2022-09-01", "2023-09-01", "2024-09-01", "2025-09-01")   # first trading day on or after each

# data cleaning
DROPPED = {
    "0011.HK": "no Yahoo data, delisted 2026",
    "0203.HK": "no Yahoo data, delisted",
    "0494.HK": "no Yahoo data, delisted 2020",
    "0906.HK": "no Yahoo data, delisted 2008",
    "0013.HK": "code reused, Yahoo returns HUTCHMED from 2021",
    "0020.HK": "code reused, Yahoo returns SenseTime from 2021",
    "1880.HK": "code reused, Yahoo returns China Tourism Group Duty Free from 2022",
}
YAHOO_ERRORS = (("0004.HK", "2017-11-16"),)   # spin-off adjustment leaves a false +15.7 per cent
USDHKD_BAND = (7.7, 7.9)                      # quotes outside are data errors

# factors
WINDOW = 252
K_MAX = 4

# LSTM-VAE
SEQ_LEN = 240
HORIZON = 21
HIDDEN = 32
LAYERS = 2
DROPOUT = 0.25
LATENT = 2                    # latent size d, and the number of path components in the PCA benchmark
LATENT_M3 = 3                 # latent size of the second VAE, book M3
LR = 1e-3
BATCH = 256
MAX_EPOCHS = 300              # a ceiling only, early stopping ends training
PATIENCE = 10
CLIP = 1.0
VALID_DAYS = 252              # final twelve months of each training window
ACTIVE_THRESHOLD = 0.01       # reported check only, Burda et al. (2016)

# books
WEIGHT_CAP = 0.03             # Proposal 1 single-name limit
COST_BPS = 15.0               # one way, books and benchmark alike
BL_DELTA = 2.5                # Black-Litterman risk aversion, He and Litterman (1999)
BL_TAU = 0.05                 # prior scale; cancels from the posterior when Omega = diag(tau Sigma)

# fund and options
FUND_USD = 100e6
MULTIPLIER = 50               # HK$ per HSI point
IM_HKD = 117_705              # HKCC initial margin per HSI futures contract from 2 Mar 2026
IM_DATE = "2026-03-02"        # the margin ratio IM / (50 S) is taken at this date's close
CALL_MONEYNESS = 1.04         # P1 short call, first listed strike at or above; middle of Proposal 1's 3 to 5 per cent OTM
PUT_LONG = 0.95               # P2 long put, first listed strike at or below; Proposal 1 puts 5 to 10 per cent below spot
PUT_SHORT = 0.90              # P2 short put, first listed strike at or below; the put spread of the original notebook
CALL_BAND = (1.05, 1.10)      # P2 short call at zero net premium, held within Proposal 1's 5 to 10 per cent above spot
VHSI_ON = 30.0                # P2 on above, Proposal 1 rainy day; its drawdown and PC1 triggers are not used
VHSI_OFF = 22.0               # P2 off below, Proposal 1 unwind
DELTA_BUDGET = 0.15           # Proposal 1 delta budget, applied to P2 at entry as a share of NAV
DD_ON, DD_OFF = 0.05, 0.02    # Proposal 1 drawdown trigger of the book, in the diagnostic arm only
STRIKE_SWITCH = 20_000        # HKEX: 100-point strikes below, 200-point at or above
STRIKE_STEP_LOW = 100
STRIKE_STEP_HIGH = 200
SPREAD_MIN = 30               # HKEX market maker maximum spread max(30, 10% of bid) up to 750 points
SPREAD_PCT = 0.10
SPREAD_LIMIT = 750
SPREAD_ABOVE = 75
OPTION_FEES = 10.54           # exchange fee 10.00 plus commission levy 0.54, per contract per side
FUTURES_FEES = 10.54          # HSI futures, the same HKEX exchange fee and levy per contract per side
FUTURES_HALF_SPREAD = 1.0     # index points, one HSI futures tick (HKEX); the quoted spread itself has no source
DIV_YIELD = 0.03              # HSI dividend yield in Black-Scholes, 2800.HK 2014 to Aug 2021 average 3.27% rounded
COVERAGE = 0.5                # h, main case
COVERAGE_REF = 1.0            # h, reported for reference
ARMS = ("none", "P1", "P2", "P1+P2")    # P1 yield, P2 protection
DIAGNOSTIC_ARMS = ("P1 unhedged", "P2 Proposal 1")    # the call without its hedge, P2 on Proposal 1's full trigger
SHIFT_MAX = 5.0               # vol points, upper end of the break-even search

SEED = 20260920
