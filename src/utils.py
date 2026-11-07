"""
utils.py — Shared utilities for the 13F Timestamp Arbitrage strategy.

Contents
--------
RateLimiter            : Rolling-window rate limiter for EDGAR API calls.
lag_dataframe          : Look-ahead bias guard; shifts signal columns back N periods.
get_quarter_label      : pd.Timestamp → "Q1_2023" string.
get_quarter_end        : "Q1_2023" string → pd.Timestamp of last day of quarter.
next_business_day      : Rolls a date forward to the next trading day (US market).
setup_logging          : Configures the root logger with the project format.
"""

import logging
import time
from collections import deque
from typing import Optional

import pandas as pd
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import USFederalHolidayCalendar

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: Optional[str] = None) -> None:
    """Configure the root logger with the project-standard format.

    Parameters
    ----------
    level : str, optional
        Override log level string (e.g. "DEBUG"). Defaults to ``config.LOG_LEVEL``.

    Notes
    -----
    Call once at the top of each script / notebook kernel. Subsequent calls
    are no-ops because ``basicConfig`` is idempotent when handlers already exist.
    """
    effective_level = level or config.LOG_LEVEL
    logging.basicConfig(
        level=getattr(logging, effective_level.upper(), logging.INFO),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket style rate limiter for SEC EDGAR API compliance.

    SEC EDGAR Terms of Service mandate a maximum of **10 requests per second**.
    This class enforces a configurable ceiling using a rolling deque of request
    timestamps and blocks (``time.sleep``) when the ceiling would be breached.

    Parameters
    ----------
    max_calls : int
        Maximum number of requests allowed within ``period`` seconds.
        Default is 8 (conservative; 80 % of the 10 req/s ceiling).
    period : float
        Rolling window in seconds. Default is 1.0.
    min_delay : float
        Minimum sleep between any two consecutive calls, regardless of the
        rolling window. Default is ``config.REQUEST_DELAY_SECONDS``.

    Usage
    -----
    Instantiate once per process (or per scraper object) and call
    ``limiter.wait()`` immediately before every outbound EDGAR HTTP request::

        limiter = RateLimiter()
        limiter.wait()
        response = requests.get(url, headers=headers)
    """

    def __init__(
        self,
        max_calls: int = 8,
        period: float = 1.0,
        min_delay: float = config.REQUEST_DELAY_SECONDS,
    ) -> None:
        self.max_calls = max_calls
        self.period = period
        self.min_delay = min_delay
        self._calls: deque[float] = deque()
        self._last_call: float = 0.0

    def wait(self) -> None:
        """Block until a new request can safely be issued.

        Enforces both the rolling-window ceiling and the per-call minimum delay.
        """
        now = time.monotonic()

        # Enforce minimum per-call delay
        elapsed_since_last = now - self._last_call
        if elapsed_since_last < self.min_delay:
            time.sleep(self.min_delay - elapsed_since_last)
            now = time.monotonic()

        # Enforce rolling window ceiling
        # Remove timestamps outside the current window
        cutoff = now - self.period
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()

        if len(self._calls) >= self.max_calls:
            oldest = self._calls[0]
            sleep_for = self.period - (now - oldest)
            if sleep_for > 0:
                logger.debug("Rate limit: sleeping %.3fs", sleep_for)
                time.sleep(sleep_for)
                now = time.monotonic()

        self._calls.append(now)
        self._last_call = now

    @property
    def call_count(self) -> int:
        """Number of calls recorded in the current rolling window."""
        cutoff = time.monotonic() - self.period
        return sum(1 for t in self._calls if t >= cutoff)


# ─────────────────────────────────────────────────────────────────────────────
# Look-ahead bias guard
# ─────────────────────────────────────────────────────────────────────────────

def lag_dataframe(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Shift all numeric signal columns back by ``periods`` to prevent look-ahead bias.

    This must be applied to any signal DataFrame before it is joined with
    forward return data. Shifting ensures that signal values at time *t* are
    only informed by information available at time *t - periods*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex or a ``quarter_label`` / ``quarter_end_date``
        column that can be used as the sort key. Signal columns are shifted;
        identifier columns (strings, booleans) are not.
    periods : int, optional
        Number of periods to lag. Default is 1 (one quarter).

    Returns
    -------
    pd.DataFrame
        Lagged copy of ``df``. The first ``periods`` rows will contain NaN for
        all numeric columns — drop them before backtesting.

    Notes
    -----
    This function is the primary guard against the most common mistake in
    event-driven strategy research: using filing information from quarter *t*
    to trade at the *beginning* of quarter *t* rather than after the filing
    date. The filing date itself is preserved as an event timestamp; this
    lag applies only to cross-sectional aggregation.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    result = df.copy()
    result[numeric_cols] = result[numeric_cols].shift(periods)
    logger.debug(
        "lag_dataframe: shifted %d numeric columns by %d period(s); "
        "first %d rows will be NaN",
        len(numeric_cols), periods, periods,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Quarter label utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_quarter_label(date: pd.Timestamp) -> str:
    """Convert a date to a quarter label string.

    Parameters
    ----------
    date : pd.Timestamp
        Any date within the target quarter.

    Returns
    -------
    str
        Label in "Q{n}_{yyyy}" format, e.g. "Q1_2023" for 2023-02-15.

    Examples
    --------
    >>> get_quarter_label(pd.Timestamp("2023-03-31"))
    'Q1_2023'
    >>> get_quarter_label(pd.Timestamp("2022-10-01"))
    'Q4_2022'
    """
    quarter_num = (date.month - 1) // 3 + 1
    return f"Q{quarter_num}_{date.year}"


def get_quarter_end(quarter_label: str) -> pd.Timestamp:
    """Convert a quarter label string to the last day of that quarter.

    Parameters
    ----------
    quarter_label : str
        Label in "Q{n}_{yyyy}" format, e.g. "Q3_2022".

    Returns
    -------
    pd.Timestamp
        Last calendar day of the specified quarter.

    Raises
    ------
    ValueError
        If ``quarter_label`` is not in the expected format.

    Examples
    --------
    >>> get_quarter_end("Q1_2023")
    Timestamp('2023-03-31 00:00:00')
    >>> get_quarter_end("Q4_2022")
    Timestamp('2022-12-31 00:00:00')
    """
    try:
        parts = quarter_label.split("_")
        q_num = int(parts[0][1:])  # strip leading "Q"
        year  = int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"quarter_label must be 'Q{{n}}_{{yyyy}}', got '{quarter_label}'"
        ) from exc

    quarter_end_month = q_num * 3
    # Last day of month — use pd.Period for robustness
    period = pd.Period(year=year, month=quarter_end_month, freq="M")
    return period.to_timestamp(how="end").normalize()


# ─────────────────────────────────────────────────────────────────────────────
# Business day utilities
# ─────────────────────────────────────────────────────────────────────────────

# Cache the holiday calendar so we don't rebuild it on every call
_CAL = USFederalHolidayCalendar()
_HOLIDAYS: Optional[pd.DatetimeIndex] = None


def _get_holidays() -> pd.DatetimeIndex:
    """Lazily build and cache the US federal holiday index (2010–2030)."""
    global _HOLIDAYS
    if _HOLIDAYS is None:
        _HOLIDAYS = _CAL.holidays(start="2010-01-01", end="2030-12-31")
    return _HOLIDAYS


def next_business_day(date: pd.Timestamp) -> pd.Timestamp:
    """Roll ``date`` forward to the next NYSE business day if it falls on a
    weekend or US federal holiday.

    Used to compute the actual SEC filing deadline when the 45th calendar day
    falls on a non-business day. SEC enforcement rolls the deadline to the
    next business day; being off by one day corrupts the filing_urgency
    calculation.

    Parameters
    ----------
    date : pd.Timestamp
        Candidate deadline date (e.g. ``quarter_end + 45 days``).

    Returns
    -------
    pd.Timestamp
        The same date if it is already a business day; otherwise the next
        valid business day, accounting for US federal holidays.

    Notes
    -----
    Uses ``pandas.tseries.holiday.USFederalHolidayCalendar`` for holiday
    detection and ``pandas.tseries.offsets.BDay`` for day increments.

    Examples
    --------
    >>> next_business_day(pd.Timestamp("2023-11-11"))  # Veterans Day
    Timestamp('2023-11-13 00:00:00')
    >>> next_business_day(pd.Timestamp("2023-11-10"))  # Friday
    Timestamp('2023-11-10 00:00:00')
    """
    holidays = _get_holidays()
    d = date.normalize()
    while d.weekday() >= 5 or d in holidays:
        d += pd.Timedelta(days=1)
    return d


def days_to_deadline(filing_date: pd.Timestamp, deadline: pd.Timestamp) -> int:
    """Compute calendar days remaining between a filing date and its deadline.

    Parameters
    ----------
    filing_date : pd.Timestamp
        Date the 13F was submitted to EDGAR.
    deadline : pd.Timestamp
        Adjusted business-day deadline for the relevant quarter.

    Returns
    -------
    int
        ``(deadline - filing_date).days``. Negative if filed after deadline.
    """
    return (deadline.normalize() - filing_date.normalize()).days
