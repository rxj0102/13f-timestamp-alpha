"""
config.py — Central configuration for 13F Timestamp Arbitrage strategy.

All hyperparameters, endpoints, file paths, and thresholds live here.
Nothing in src/ should contain hard-coded values; import from this module.

Usage
-----
    from config import EDGAR_SUBMISSIONS_URL, MIN_AUM_BILLION, DATA_DIR
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Repository root — all paths derived from here
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw" / "filings"
PROCESSED_DIR  = DATA_DIR / "processed"
REFERENCE_DIR  = DATA_DIR / "reference"
OUTPUTS_DIR    = ROOT_DIR / "outputs"
NOTEBOOKS_DIR  = ROOT_DIR / "notebooks"
SRC_DIR        = ROOT_DIR / "src"

# Ensure runtime directories exist (reference/ must be pre-populated)
for _d in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Reference file paths
# ─────────────────────────────────────────────────────────────────────────────
FILING_DEADLINES_CSV  = REFERENCE_DIR / "filing_deadlines.csv"
MANAGER_UNIVERSE_CSV  = REFERENCE_DIR / "manager_universe.csv"
CTR_LOG_CSV           = REFERENCE_DIR / "ctr_log.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Processed output paths
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE_FILING_HISTORY_CSV     = PROCESSED_DIR / "universe_filing_history.csv"
UNIVERSE_HOLDINGS_PANEL_PARQUET = PROCESSED_DIR / "universe_holdings_panel.parquet"
CUSIP_TICKER_MAP_JSON           = PROCESSED_DIR / "cusip_ticker_map.json"

# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR API endpoints
# ─────────────────────────────────────────────────────────────────────────────

# Full-text search index (form type filter + date range)
EDGAR_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q={query}&forms=13F-HR&dateRange=custom&startdt={start}&enddt={end}"
)

# JSON submissions feed — filing history for a single CIK
# CIK must be zero-padded to 10 digits, e.g. CIK0001649339
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# HTML filing browser (used to extract precise timestamps from filing index)
EDGAR_FILING_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type=13F-HR&dateb=&owner=include&count=40"
)

# Base URL for accession folder contents
# accession_number must have dashes removed, e.g. 0001649339-21-000010 → 000164933921000010
EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"

# Filing header document (contains precise filing datetime to the second)
EDGAR_HEADER_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession}.hdr.sgml"
)

# ─────────────────────────────────────────────────────────────────────────────
# Manager universe filters
# (see docs/strategy_spec.md for rationale)
# ─────────────────────────────────────────────────────────────────────────────

# AUM band: below $5B → sparse coverage; above $50B → compliance-driven filing
MIN_AUM_BILLION = 5.0
MAX_AUM_BILLION = 50.0

# Position count band: <10 → not a proper equity book; >200 → likely quant
MIN_POSITIONS = 10
MAX_POSITIONS = 200

# Manager types eligible for the signal
MANAGER_TYPE_FILTER = ["hedge_fund", "family_office"]

# ─────────────────────────────────────────────────────────────────────────────
# Filing timing parameters
# ─────────────────────────────────────────────────────────────────────────────
QUARTER_END_DATES   = ["03-31", "06-30", "09-30", "12-31"]
FILING_DEADLINE_DAYS = 45  # calendar days after quarter-end (SEC rule)

# Urgency thresholds (filing_urgency = days_remaining / 45)
# Higher urgency → filed later → manager still holding → higher conviction
URGENCY_EARLY_THRESHOLD = 0.33   # < 0.33  → "EARLY" filer (filed in first 15 days)
URGENCY_LATE_THRESHOLD  = 0.75   # > 0.75  → "LATE"  filer (filed in last 11 days)

# Minimum fractional position change to register as a real trade (not rounding noise)
MIN_POSITION_DELTA_PCT = 0.05   # 5% change in share count

# ─────────────────────────────────────────────────────────────────────────────
# Conviction / signal parameters
# ─────────────────────────────────────────────────────────────────────────────

# Minimum number of independent managers with non-zero conviction for a
# ticker-quarter combination to be included in the cross-manager signal
CROSS_MANAGER_MIN_COUNT = 3

# Minimum quarters of filing history required before a manager's urgency
# deviation score is considered reliable
CONVICTION_SCORE_WINDOW = 2   # quarters

# Expected signal holding horizon (one quarter forward)
SIGNAL_HOLDING_DAYS = 60

# ─────────────────────────────────────────────────────────────────────────────
# Backtest parameters
# ─────────────────────────────────────────────────────────────────────────────
BACKTEST_START       = "2014-01-01"
BACKTEST_END         = "2023-12-31"

# One-way transaction cost assumption (basis points)
# 10 bps is mid-liquidity; tighten to 5 for mega-caps, widen to 20 for small-caps
TRANSACTION_COST_BPS = 10

# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR rate limiting — Terms of Service: max 10 requests/second
# Hard minimum: 0.10 s between requests → 10 req/s ceiling
# ─────────────────────────────────────────────────────────────────────────────
REQUEST_DELAY_SECONDS = 0.15   # conservative: ~6.7 req/s
MAX_RETRIES           = 3      # exponential back-off: 2^n seconds (2, 4, 8)

# User-Agent header required by SEC EDGAR ToS
# Replace with your name/contact before running live scraping
EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "13F-Timestamp-Research/1.0 research@example.com"
)

# ─────────────────────────────────────────────────────────────────────────────
# External API endpoints
# ─────────────────────────────────────────────────────────────────────────────

# OpenFIGI — free CUSIP-to-ticker mapping (no auth required for basic lookups)
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

# SEC company ticker JSON (fallback CUSIP reference)
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
