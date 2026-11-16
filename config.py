"""Settings for the 13F filing timing study. Fixed before the study was run.

The universe filters reproduce the strategy's original manager screen, so the study can say
whether the managers the strategy would trade behave differently from the rest.
"""
import os

# Universe screen (scripts/build_manager_universe.py)
MIN_AUM_BILLION = 5.0
MAX_AUM_BILLION = 50.0
MIN_POSITIONS = 10
MAX_POSITIONS = 200
MANAGER_TYPE_FILTER = ["hedge_fund", "family_office"]

# Filing calendar
FILING_DEADLINE_DAYS = 45          # calendar days after quarter end (SEC rule 13f-1)
URGENCY_EARLY_THRESHOLD = 0.33     # the strategy's EARLY bucket: filed in the first third of the window

# SEC EDGAR access. SEC asks for a User-Agent that identifies you; set it before fetching:
#   export EDGAR_USER_AGENT="Your Name your.address@example.com"
REQUEST_DELAY_SECONDS = 0.15
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "")
