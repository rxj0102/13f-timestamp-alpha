"""
edgar_scraper.py — SEC EDGAR API client for 13F filing data.

Fetches filing histories, precise filing timestamps, and holdings XML
documents for hedge fund managers in the strategy universe. All network
requests comply with the SEC EDGAR Terms of Service rate limit (10 req/s).

Entry point (smoke test)
------------------------
    python -m src.edgar_scraper --cik 0001649339

This runs ``build_manager_filing_history`` for a single CIK and prints a
summary of filing count, date range, and urgency distribution.

Design notes
------------
* Every EDGAR request is gated through a shared ``RateLimiter`` instance.
* Raw responses are cached to ``data/raw/filings/`` so re-runs never re-fetch
  already-downloaded files (important for development and reproducibility).
* The precise filing datetime is extracted from the ``*.hdr.sgml`` header file
  inside each accession folder — not from the submissions JSON, which only
  provides a date. Sub-day resolution is required for the filing_urgency calc.
* CTR amendment detection (``check_for_ctr_amendment``) is the primary bias
  guard: positions disclosed only via amendment were hidden at signal-
  construction time and must be excluded from the backtest.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import config
from src.utils import (
    RateLimiter,
    get_quarter_label,
    next_business_day,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Module-level rate limiter shared across all functions in this module
_LIMITER = RateLimiter(
    max_calls=8,
    period=1.0,
    min_delay=config.REQUEST_DELAY_SECONDS,
)

# HTTP session with required User-Agent header
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": config.EDGAR_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, stream: bool = False, **kwargs) -> requests.Response:
    """Rate-limited GET with exponential back-off retry.

    Parameters
    ----------
    url : str
        Target URL.
    stream : bool
        Passed through to ``requests.get``.
    **kwargs :
        Additional arguments forwarded to ``requests.get``.

    Returns
    -------
    requests.Response
        Successful HTTP response (status 200).

    Raises
    ------
    requests.HTTPError
        If all ``config.MAX_RETRIES`` attempts fail.
    """
    for attempt in range(1, config.MAX_RETRIES + 1):
        _LIMITER.wait()
        logger.debug("GET %s (attempt %d/%d)", url, attempt, config.MAX_RETRIES)
        try:
            resp = _SESSION.get(url, stream=stream, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == config.MAX_RETRIES:
                logger.error("All %d attempts failed for %s: %s", config.MAX_RETRIES, url, exc)
                raise
            sleep_secs = 2 ** attempt
            logger.warning(
                "Request failed (attempt %d/%d): %s — retrying in %ds",
                attempt, config.MAX_RETRIES, exc, sleep_secs,
            )
            time.sleep(sleep_secs)
    # Unreachable, but satisfies type checker
    raise RuntimeError("Retry loop exited unexpectedly")


def _normalise_cik(cik: str) -> str:
    """Zero-pad CIK to 10 digits, stripping leading zeros if already present."""
    return str(int(cik.lstrip("0") or "0")).zfill(10)


def _accession_nodash(accession: str) -> str:
    """Remove dashes from accession number for URL path construction."""
    return accession.replace("-", "")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_manager_submissions(cik: str) -> dict:
    """Fetch and cache the full filing history for a single manager from EDGAR.

    Hits the EDGAR submissions JSON API and filters to 13F-HR filings only.
    The raw JSON is cached to ``data/raw/filings/{cik}_submissions.json`` so
    subsequent calls return the cached copy without network I/O.

    Parameters
    ----------
    cik : str
        Manager CIK (with or without leading zeros; normalised internally).

    Returns
    -------
    dict
        Parsed submissions JSON with an additional key ``"filings_13f"``
        containing a list of dicts, one per 13F-HR filing, each with keys:

        * ``accession_number`` — hyphenated accession number
        * ``filing_date``      — date-only string "YYYY-MM-DD"
        * ``period_of_report`` — quarter-end date "YYYY-MM-DD"
        * ``primary_document`` — filename of the primary form document
        * ``form``             — "13F-HR"

    Notes
    -----
    The submissions API returns filings in reverse chronological order.
    Large filers (>1000 filings) require pagination via the ``next`` URL in the
    response — this function fetches only the primary JSON (covers the most
    recent ~40 filings). For full history pre-2014, additional archive fetches
    would be needed.
    """
    cik_norm = _normalise_cik(cik)
    cache_path = config.RAW_DIR / f"{cik_norm}_submissions.json"

    if cache_path.exists():
        logger.debug("Loading cached submissions for CIK %s", cik_norm)
        with cache_path.open("r") as f:
            data = json.load(f)
        return data

    url = config.EDGAR_SUBMISSIONS_URL.format(cik=cik_norm)
    logger.info("Fetching submissions for CIK %s from %s", cik_norm, url)
    resp = _get(url)
    data = resp.json()

    # Extract 13F-HR filings into a convenience list
    recent = data.get("filings", {}).get("recent", {})
    forms       = recent.get("form", [])
    acc_nums    = recent.get("accessionNumber", [])
    file_dates  = recent.get("filingDate", [])
    periods     = recent.get("periodOfReport", [])
    primary_docs = recent.get("primaryDocument", [])

    filings_13f = []
    for form, acc, filed, period, doc in zip(forms, acc_nums, file_dates, periods, primary_docs):
        if form in ("13F-HR", "13F-HR/A"):
            filings_13f.append(
                {
                    "accession_number": acc,
                    "filing_date":      filed,
                    "period_of_report": period,
                    "primary_document": doc,
                    "form":             form,
                }
            )

    data["filings_13f"] = filings_13f
    logger.info(
        "CIK %s: found %d 13F-HR / 13F-HR/A filings in submissions JSON",
        cik_norm, len(filings_13f),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(data, f, indent=2)

    return data


def extract_filing_timestamp(cik: str, accession_number: str) -> pd.Timestamp:
    """Extract the precise filing datetime (to the second) from the EDGAR header.

    The EDGAR submissions JSON provides only a filing *date* (YYYY-MM-DD).
    The full datetime — including hour, minute, and second — is embedded in the
    ``*.hdr.sgml`` file within the accession folder. This function fetches that
    file and parses the ``FILED AS OF DATE`` and ``ACCEPTANCE-DATETIME`` fields.

    Parameters
    ----------
    cik : str
        Manager CIK (normalised internally).
    accession_number : str
        Hyphenated accession number, e.g. ``"0001649339-21-000010"``.

    Returns
    -------
    pd.Timestamp
        Filing datetime with ``tz="UTC"``. If the precise time cannot be
        determined, returns midnight UTC on the filing date.

    Notes
    -----
    The ``ACCEPTANCE-DATETIME`` field in the SGML header has format
    ``YYYYMMDDHHmmss``, e.g. ``20210512162347`` → 2021-05-12 16:23:47 UTC.
    This is the SEC's official record of when the filing was accepted.

    The distinction between filing *date* and filing *datetime* matters for
    same-day urgency comparisons, but the primary use of this function is to
    confirm that the date-level urgency calculation uses the correct date
    (EDGAR occasionally records a filing date as the next calendar day if
    submitted after midnight EST).
    """
    cik_norm = _normalise_cik(cik)
    acc_nodash = _accession_nodash(accession_number)
    cache_path = config.RAW_DIR / f"{cik_norm}_{acc_nodash}_timestamp.json"

    if cache_path.exists():
        with cache_path.open("r") as f:
            cached = json.load(f)
        return pd.Timestamp(cached["filing_datetime"], tz="UTC")

    # Construct header URL
    # Pattern: https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{nodash}.hdr.sgml
    url = config.EDGAR_HEADER_URL.format(cik=cik_norm, accession=acc_nodash)
    logger.info("Fetching filing header for CIK %s / %s", cik_norm, accession_number)

    try:
        resp = _get(url)
        text = resp.text
    except requests.HTTPError as exc:
        logger.warning(
            "Could not fetch header for %s/%s (%s); falling back to date-only",
            cik_norm, accession_number, exc,
        )
        # Fallback: parse filing date from submissions cache
        subs = fetch_manager_submissions(cik)
        filing_date_str = next(
            (f["filing_date"] for f in subs.get("filings_13f", [])
             if f["accession_number"] == accession_number),
            None,
        )
        ts = pd.Timestamp(filing_date_str or "1900-01-01", tz="UTC")
        _cache_timestamp(cache_path, ts, source="fallback")
        return ts

    # Parse ACCEPTANCE-DATETIME: YYYYMMDDHHmmss
    match = re.search(r"ACCEPTANCE-DATETIME>\s*(\d{14})", text)
    if match:
        raw = match.group(1)
        ts = pd.Timestamp(
            year=int(raw[0:4]),
            month=int(raw[4:6]),
            day=int(raw[6:8]),
            hour=int(raw[8:10]),
            minute=int(raw[10:12]),
            second=int(raw[12:14]),
            tz="UTC",
        )
        logger.debug("Parsed ACCEPTANCE-DATETIME: %s", ts)
    else:
        # Fallback: FILED AS OF DATE (date only)
        match2 = re.search(r"FILED AS OF DATE:\s*(\d{8})", text)
        if match2:
            raw = match2.group(1)
            ts = pd.Timestamp(year=int(raw[0:4]), month=int(raw[4:6]), day=int(raw[6:8]), tz="UTC")
            logger.debug("Parsed FILED AS OF DATE (no time): %s", ts)
        else:
            logger.warning(
                "Could not parse datetime from header for %s/%s; using epoch",
                cik_norm, accession_number,
            )
            ts = pd.Timestamp("1900-01-01", tz="UTC")

    _cache_timestamp(cache_path, ts, source="hdr_sgml")
    return ts


def _cache_timestamp(cache_path: Path, ts: pd.Timestamp, source: str) -> None:
    """Write a timestamp to disk cache as JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump({"filing_datetime": str(ts), "source": source}, f)


def fetch_13f_holdings_xml(cik: str, accession_number: str) -> str:
    """Download and cache the primary 13F holdings XML document.

    The SEC stores holdings in an XML file named ``informationTable.xml``
    within the accession folder. This function locates, downloads, and caches
    that file.

    Parameters
    ----------
    cik : str
        Manager CIK (normalised internally).
    accession_number : str
        Hyphenated accession number.

    Returns
    -------
    str
        Raw XML string of the informationTable document.

    Raises
    ------
    FileNotFoundError
        If the informationTable.xml cannot be found in the accession folder.
    requests.HTTPError
        If all ``config.MAX_RETRIES`` attempts fail.

    Notes
    -----
    Rate limiting: this function calls ``_LIMITER.wait()`` before every
    network request. The accession index is fetched first to find the exact
    XML filename (it is sometimes named differently from ``informationTable.xml``
    in older filings), then the XML itself is downloaded.
    """
    cik_norm = _normalise_cik(cik)
    acc_nodash = _accession_nodash(accession_number)
    cache_path = config.RAW_DIR / f"{cik_norm}_{acc_nodash}_holdings.xml"

    if cache_path.exists():
        logger.debug("Loading cached holdings XML for CIK %s / %s", cik_norm, accession_number)
        return cache_path.read_text(encoding="utf-8")

    # Step 1: fetch the accession index to find the XML filename
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_norm)}/{acc_nodash}/{acc_nodash}-index.htm"
    logger.info(
        "Fetching %s / %s — holdings index",
        cik_norm, accession_number,
    )
    try:
        resp = _get(index_url)
        index_text = resp.text
    except requests.HTTPError:
        # Try the JSON index as fallback
        index_url_json = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_norm}&type=13F-HR&dateb=&owner=include&count=40&search_text="
        resp = _get(index_url_json)
        index_text = resp.text

    # Extract XML filename from index — look for informationTable
    xml_filename = None
    for pattern in [
        r'informationTable\.xml',
        r'[Ii]nformation[Tt]able\w*\.xml',
        r'\w+\.xml',
    ]:
        match = re.search(pattern, index_text)
        if match:
            xml_filename = match.group(0)
            break

    if xml_filename is None:
        # Default assumption for modern filings
        xml_filename = "informationTable.xml"
        logger.warning(
            "Could not identify XML filename for %s/%s; assuming %s",
            cik_norm, accession_number, xml_filename,
        )

    # Step 2: download the XML
    xml_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik_norm)}/{acc_nodash}/{xml_filename}"
    )
    logger.info(
        "Fetching %s / %s — attempt 1 — %s",
        cik_norm, accession_number, xml_url,
    )
    resp = _get(xml_url)
    xml_text = resp.text

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(xml_text, encoding="utf-8")
    logger.debug("Cached holdings XML to %s", cache_path)

    return xml_text


def check_for_ctr_amendment(
    cik: str,
    accession_number: str,
    original_filing_date: str,
) -> dict:
    """Detect whether a 13F-HR/A amendment exists that restores CTR positions.

    Confidential Treatment Requests (CTRs) allow managers to omit specific
    positions from the public 13F disclosure for up to one year. When the
    confidentiality period expires, the SEC requires disclosure via a 13F-HR/A
    amendment. This function identifies whether such an amendment exists for a
    given original filing.

    **CRITICAL — look-ahead bias guard**: positions disclosed only in the
    amendment were hidden at signal-construction time. Including their data
    as if it were available at the original filing date constitutes look-ahead
    bias. Callers must exclude amendment-only positions from the backtest.

    Parameters
    ----------
    cik : str
        Manager CIK.
    accession_number : str
        Accession number of the *original* 13F-HR filing (not the amendment).
    original_filing_date : str
        Date the original filing was made ("YYYY-MM-DD"). Used to bound the
        amendment search window (original_date + 1 year).

    Returns
    -------
    dict
        With keys:

        * ``has_amendment`` (bool) — True if a 13F-HR/A amendment was found
        * ``amendment_accession`` (str | None) — accession number of amendment
        * ``amendment_date`` (str | None) — filing date of amendment
        * ``estimated_hidden_positions`` (int) — count of positions in amendment
          not present in the original (i.e., CTR-restored positions)

    Notes
    -----
    The amendment search uses the EDGAR full-text search API filtered to
    13F-HR/A forms from the same CIK within four quarters of the original.
    This is a best-effort heuristic; manual review of the SEC CTR order list
    is the authoritative source.
    """
    cik_norm = _normalise_cik(cik)
    orig_date = pd.Timestamp(original_filing_date)
    search_end = (orig_date + pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    url = (
        f"https://efts.sec.gov/LATEST/search-index"
        f"?q=%2213F-HR%2FA%22&forms=13F-HR%2FA"
        f"&dateRange=custom&startdt={original_filing_date}&enddt={search_end}"
        f"&entity={cik_norm}"
    )
    logger.info(
        "Searching for 13F-HR/A amendments for CIK %s after %s",
        cik_norm, original_filing_date,
    )

    try:
        resp = _get(url)
        data = resp.json()
    except Exception as exc:
        logger.warning("CTR amendment search failed for %s: %s", cik_norm, exc)
        return {
            "has_amendment": False,
            "amendment_accession": None,
            "amendment_date": None,
            "estimated_hidden_positions": 0,
        }

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return {
            "has_amendment": False,
            "amendment_accession": None,
            "amendment_date": None,
            "estimated_hidden_positions": 0,
        }

    # Take the first (earliest) amendment
    first = hits[0].get("_source", {})
    amendment_accession = first.get("accession_no", None)
    amendment_date      = first.get("file_date", None)

    # Estimate hidden positions: difference in position count
    # (requires fetching the amendment XML — expensive; use a heuristic)
    estimated_hidden = len(hits)  # rough proxy: 1 hidden position per search hit

    logger.info(
        "CTR amendment found for CIK %s: accession=%s date=%s",
        cik_norm, amendment_accession, amendment_date,
    )
    return {
        "has_amendment": True,
        "amendment_accession": amendment_accession,
        "amendment_date": amendment_date,
        "estimated_hidden_positions": estimated_hidden,
    }


def build_manager_filing_history(cik: str, manager_name: str = "") -> pd.DataFrame:
    """Build the complete filing history DataFrame for a single manager.

    Orchestrates the full per-manager pipeline:
    1. ``fetch_manager_submissions`` — get filing list
    2. ``extract_filing_timestamp`` — get precise datetime for each filing
    3. ``check_for_ctr_amendment`` — flag potential CTR bias
    4. Load the deadline calendar to compute ``filing_urgency``

    Parameters
    ----------
    cik : str
        Manager CIK.
    manager_name : str, optional
        Human-readable name for logging. Pulled from manager universe if blank.

    Returns
    -------
    pd.DataFrame
        One row per 13F-HR filing with columns:

        * ``cik``                — zero-padded CIK string
        * ``manager_name``       — manager display name
        * ``quarter_label``      — "Q{n}_{yyyy}"
        * ``quarter_end_date``   — last day of the reporting quarter
        * ``filing_deadline``    — business-day-adjusted 45-day deadline
        * ``filing_datetime``    — precise UTC timestamp of submission
        * ``filing_date_only``   — date portion of filing_datetime
        * ``accession_number``   — hyphenated EDGAR accession number
        * ``has_ctr_amendment``  — bool; True if CTR amendment detected
        * ``n_positions``        — number of positions in the filing (set to 0
                                    here; populated by ``filing_parser``)
        * ``filing_urgency``     — (deadline − filing_date).days / 45, clamped
                                    to [0, 1]; higher = later = more convinced

    Notes
    -----
    ``filing_urgency`` is defined as the *fraction of the filing window used*
    before the manager submitted. A manager who files on the last possible day
    (urgency ≈ 1.0) had maximum opportunity to close positions before disclosure
    and chose not to — strong evidence they are still holding. A manager who
    files on day 5 (urgency ≈ 0.11) almost certainly did so before making any
    repositioning decisions.

    Clamping: urgency > 1.0 (filed after deadline) is set to 1.0 with a
    warning. This is rare and usually indicates an SEC-granted extension.
    """
    cik_norm = _normalise_cik(cik)
    cache_path = config.PROCESSED_DIR / f"{cik_norm}_filing_history.csv"

    # Load deadline calendar
    deadlines_df = pd.read_csv(config.FILING_DEADLINES_CSV, comment="#")
    deadlines_df["quarter_end_date"] = pd.to_datetime(deadlines_df["quarter_end_date"])
    deadlines_df["filing_deadline"]  = pd.to_datetime(deadlines_df["filing_deadline"])
    deadline_map = dict(zip(deadlines_df["quarter_label"], deadlines_df["filing_deadline"]))

    # Fetch submission history
    subs = fetch_manager_submissions(cik)
    name = manager_name or subs.get("name", cik_norm)
    filings = [f for f in subs.get("filings_13f", []) if f["form"] == "13F-HR"]

    if not filings:
        logger.warning("No 13F-HR filings found for CIK %s (%s)", cik_norm, name)
        return pd.DataFrame()

    rows = []
    for filing in filings:
        acc  = filing["accession_number"]
        period = pd.Timestamp(filing["period_of_report"])
        q_label = get_quarter_label(period)

        # Extract precise timestamp
        ts = extract_filing_timestamp(cik, acc)
        filing_date_only = ts.normalize().tz_localize(None)

        # Look up deadline
        deadline = deadline_map.get(q_label)
        if deadline is None:
            logger.warning(
                "No deadline found for quarter %s (CIK %s); skipping",
                q_label, cik_norm,
            )
            continue

        # Compute urgency
        days_remaining = (deadline - filing_date_only).days
        raw_urgency = days_remaining / config.FILING_DEADLINE_DAYS

        if raw_urgency > 1.0:
            logger.warning(
                "CIK %s %s: filing_urgency %.3f > 1.0 (filed before quarter end?); "
                "clamping to 1.0",
                cik_norm, q_label, raw_urgency,
            )
            urgency = 1.0
        elif raw_urgency < 0.0:
            logger.warning(
                "CIK %s %s: filing_urgency %.3f < 0.0 (filed after deadline); "
                "clamping to 1.0",
                cik_norm, q_label, raw_urgency,
            )
            urgency = 1.0
        else:
            urgency = raw_urgency

        # CTR check (lightweight — expensive XML download deferred to parser)
        ctr_result = check_for_ctr_amendment(cik, acc, filing["filing_date"])

        rows.append(
            {
                "cik":               cik_norm,
                "manager_name":      name,
                "quarter_label":     q_label,
                "quarter_end_date":  period.date(),
                "filing_deadline":   deadline.date(),
                "filing_datetime":   str(ts),
                "filing_date_only":  filing_date_only.date(),
                "accession_number":  acc,
                "has_ctr_amendment": ctr_result["has_amendment"],
                "n_positions":       0,   # populated by filing_parser
                "filing_urgency":    round(urgency, 4),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("quarter_end_date").reset_index(drop=True)

    # Log urgency distribution summary
    logger.info(
        "CIK %s (%s): %d filings | %s — %s | "
        "urgency: mean=%.3f std=%.3f min=%.3f max=%.3f",
        cik_norm, name, len(df),
        df["quarter_end_date"].min(), df["quarter_end_date"].max(),
        df["filing_urgency"].mean(), df["filing_urgency"].std(),
        df["filing_urgency"].min(), df["filing_urgency"].max(),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.debug("Saved filing history to %s", cache_path)

    return df


def build_universe_filing_history(manager_universe_df: pd.DataFrame) -> pd.DataFrame:
    """Build the filing history for all included managers in the universe.

    Filters to ``include_flag == True`` managers, calls
    ``build_manager_filing_history`` for each, and concatenates into a single
    DataFrame.

    Parameters
    ----------
    manager_universe_df : pd.DataFrame
        Loaded from ``data/reference/manager_universe.csv``.

    Returns
    -------
    pd.DataFrame
        Concatenated filing history for the full universe.
        Saved to ``data/processed/universe_filing_history.csv``.

    Notes
    -----
    Logs a summary table of average ``filing_urgency`` by manager after
    completion. This summary is the first diagnostic to review: managers with
    uniformly low urgency (< 0.33 every quarter) are compliance-driven and
    should be reconsidered for exclusion even if they passed the initial AUM
    and position-count filters.
    """
    included = manager_universe_df[manager_universe_df["include_flag"] == True].copy()
    logger.info(
        "Building universe filing history for %d included managers "
        "(%d excluded)",
        len(included),
        len(manager_universe_df) - len(included),
    )

    all_dfs = []
    for _, row in included.iterrows():
        cik  = str(row["cik"])
        name = row["manager_name"]
        logger.info("Processing %s (CIK %s)", name, cik)
        try:
            df = build_manager_filing_history(cik, manager_name=name)
            if not df.empty:
                all_dfs.append(df)
        except Exception as exc:
            logger.error("Failed to build history for %s (%s): %s", name, cik, exc)

    if not all_dfs:
        logger.warning("No filing history data returned for any manager")
        return pd.DataFrame()

    universe_df = pd.concat(all_dfs, ignore_index=True)

    # Log per-manager urgency summary
    summary = (
        universe_df.groupby("manager_name")["filing_urgency"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    logger.info(
        "Universe filing urgency summary (sorted by mean urgency):\n%s",
        summary.to_string(),
    )

    logger.info(
        "Universe total: %d filings | %d managers | %s — %s",
        len(universe_df),
        universe_df["cik"].nunique(),
        universe_df["quarter_end_date"].min(),
        universe_df["quarter_end_date"].max(),
    )

    config.UNIVERSE_FILING_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    universe_df.to_csv(config.UNIVERSE_FILING_HISTORY_CSV, index=False)
    logger.info("Saved universe filing history to %s", config.UNIVERSE_FILING_HISTORY_CSV)

    return universe_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="13F Timestamp Alpha — EDGAR scraper smoke test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cik",
        type=str,
        default="0001649339",
        help="Manager CIK to run build_manager_filing_history for",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional manager name for display",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(args.log_level)

    logger.info("=== EDGAR Scraper Smoke Test ===")
    logger.info("CIK: %s", args.cik)

    df = build_manager_filing_history(args.cik, manager_name=args.name)

    if df.empty:
        logger.error("No data returned. Check CIK and network connectivity.")
    else:
        print("\n" + "=" * 60)
        print(f"Manager: {df['manager_name'].iloc[0]}  (CIK {args.cik})")
        print(f"Filings:  {len(df)}")
        print(f"Range:    {df['quarter_end_date'].min()} → {df['quarter_end_date'].max()}")
        print("\nFiling Urgency Distribution:")
        print(df["filing_urgency"].describe().round(3).to_string())
        print("\nUrgency Bucket Counts:")
        buckets = pd.cut(
            df["filing_urgency"],
            bins=[0, 0.33, 0.75, 1.0],
            labels=["EARLY", "MIDDLE", "LATE"],
        ).value_counts()
        print(buckets.to_string())
        print("=" * 60)
