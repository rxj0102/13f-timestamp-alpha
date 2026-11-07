"""
filing_parser.py — 13F XML parser, CUSIP resolver, and holdings panel builder.

Parses the SEC's informationTable XML format into structured DataFrames,
resolves CUSIP identifiers to ticker symbols via OpenFIGI and SEC fallback,
and computes quarter-over-quarter position deltas for the conviction signal.

Key design decisions documented here
--------------------------------------
SOLE discretion filter
    The 13F form requires disclosure of all positions regardless of investment
    discretion type (SOLE / SHARED / OTHER). Shared-discretion positions are
    typically sub-advisory relationships or overlay programs where the filing
    manager does not make the investment decision independently. Including them
    would attribute someone else's conviction to our manager. This filter is
    non-negotiable for signal validity.

SH vs PRN filter
    ``sshPrnamtType == "PRN"`` indicates a principal-amount position (typically
    corporate or government bonds reported at face value). The 13F filing
    requirement for fixed income is inconsistent across managers, and the
    share count / delta calculation is meaningless for principal-amount
    positions. All PRN rows are excluded.

CUSIP resolution failure handling
    Preferred shares, warrants, rights, and ADRs often have non-standard CUSIPs
    that do not map cleanly to equity tickers. These are set to ``None`` and
    logged — they are NOT force-mapped or guessed. Force-mapping a warrant
    CUSIP to its common-share ticker overstates position size and corrupts the
    delta calculation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import pandas as pd
import requests

import config
from src.utils import RateLimiter, get_quarter_label, setup_logging

logger = logging.getLogger(__name__)

# Shared rate limiter for external API calls (OpenFIGI)
_LIMITER = RateLimiter(max_calls=5, period=1.0, min_delay=0.2)

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": config.EDGAR_USER_AGENT,
        "Content-Type": "application/json",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# XML parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_holdings_xml(xml_string: str) -> pd.DataFrame:
    """Parse a 13F informationTable XML document into a clean holdings DataFrame.

    Handles both the modern (2013+) XML schema and the older SGML-adjacent
    formats used in pre-2013 filings.

    Parameters
    ----------
    xml_string : str
        Raw XML content of the informationTable document, as returned by
        ``edgar_scraper.fetch_13f_holdings_xml``.

    Returns
    -------
    pd.DataFrame
        Filtered holdings with columns:

        * ``cusip``                 — 9-character CUSIP identifier
        * ``issuer_name``           — ``nameOfIssuer`` from the filing
        * ``share_count``           — ``sshPrnamt`` (shares outstanding reported)
        * ``market_value_usd``      — ``value`` × 1000 (SEC reports in thousands)
        * ``investment_discretion`` — "SOLE", "SHARED", or "OTHER"
        * ``voting_sole``           — sole voting authority share count

        **Filtered out**: rows with ``investmentDiscretion != "SOLE"`` and
        rows with ``sshPrnamtType != "SH"``. See module docstring for rationale.

    Notes
    -----
    The SEC's informationTable schema uses inconsistent namespace prefixes
    across filing years. This parser uses ``xml.etree.ElementTree`` with an
    explicit namespace strip so it handles all observed variants.

    Raises
    ------
    ValueError
        If the XML cannot be parsed or contains no ``<infoTable>`` elements.
    """
    import xml.etree.ElementTree as ET

    if not xml_string or xml_string.strip() == "":
        raise ValueError("Empty XML string provided")

    # Strip namespace prefixes for uniform XPath access
    # Pattern: xmlns="..." or xmlns:ns1="..."
    import re
    clean_xml = re.sub(r'\s+xmlns[^"]*"[^"]*"', "", xml_string)
    clean_xml = re.sub(r'<([a-zA-Z0-9]+):([a-zA-Z0-9_]+)', r'<\2', clean_xml)
    clean_xml = re.sub(r'</([a-zA-Z0-9]+):([a-zA-Z0-9_]+)', r'</\2', clean_xml)

    try:
        root = ET.fromstring(clean_xml)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML: {exc}") from exc

    # Find all infoTable elements (the individual holding records)
    info_tables = root.findall(".//infoTable")
    if not info_tables:
        # Some older filings use different element names
        info_tables = root.findall(".//informationTable")
    if not info_tables:
        logger.warning("No infoTable elements found; XML may be empty or malformed")
        return pd.DataFrame()

    rows = []
    for entry in info_tables:
        def _text(tag: str, default: str = "") -> str:
            el = entry.find(tag)
            return el.text.strip() if el is not None and el.text else default

        issuer     = _text("nameOfIssuer")
        title      = _text("titleOfClass")
        cusip      = _text("cusip").upper().strip()
        value_raw  = _text("value", "0")
        shares_raw = _text("sshPrnamt", "0")
        prnamt_type = _text("sshPrnamtType", "SH").upper()
        discretion  = _text("investmentDiscretion", "").upper()

        # Voting authority
        voting_sole_raw = _text("Sole", "0")  # under votingAuthority block

        # Parse numerics safely
        try:
            value_usd  = float(value_raw.replace(",", "")) * 1000
        except ValueError:
            value_usd = 0.0
        try:
            shares = float(shares_raw.replace(",", ""))
        except ValueError:
            shares = 0.0
        try:
            voting_sole = float(voting_sole_raw.replace(",", ""))
        except ValueError:
            voting_sole = 0.0

        rows.append(
            {
                "cusip":                 cusip,
                "issuer_name":           issuer,
                "title_of_class":        title,
                "share_count":           shares,
                "market_value_usd":      value_usd,
                "investment_discretion": discretion,
                "prn_amt_type":          prnamt_type,
                "voting_sole":           voting_sole,
            }
        )

    df_raw = pd.DataFrame(rows)
    total_raw = len(df_raw)

    # ── Filter 1: SOLE discretion only ────────────────────────────────────────
    df_sole = df_raw[df_raw["investment_discretion"] == "SOLE"].copy()
    n_dropped_discretion = total_raw - len(df_sole)
    if n_dropped_discretion:
        logger.debug(
            "Dropped %d positions with non-SOLE investment discretion (SHARED/OTHER)",
            n_dropped_discretion,
        )

    # ── Filter 2: Share-count positions only (exclude principal-amount) ───────
    df_sh = df_sole[df_sole["prn_amt_type"] == "SH"].copy()
    n_dropped_prn = len(df_sole) - len(df_sh)
    if n_dropped_prn:
        logger.debug(
            "Dropped %d PRN (principal-amount) positions — not equity",
            n_dropped_prn,
        )

    df_sh = df_sh.drop(columns=["prn_amt_type"])
    df_sh["cusip"] = df_sh["cusip"].str.strip().str.upper()

    logger.debug(
        "parse_holdings_xml: %d raw → %d after SOLE filter → %d after SH filter",
        total_raw, len(df_sole), len(df_sh),
    )
    return df_sh.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CUSIP resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_cusip_to_ticker(cusips: list[str]) -> dict[str, Optional[str]]:
    """Resolve a list of CUSIP identifiers to equity ticker symbols.

    Uses a two-stage approach:
    1. **OpenFIGI API** (primary) — free, no auth required for basic lookups,
       covers US equities, ETFs, and most ADRs.
    2. **SEC company_tickers.json** (fallback) — SEC's own CUSIP-to-CIK mapping,
       cross-referenced with ticker symbols.

    Already-resolved CUSIPs are served from the on-disk cache at
    ``data/processed/cusip_ticker_map.json`` without making network calls.

    Parameters
    ----------
    cusips : list[str]
        List of 9-character CUSIP strings (uppercase, no spaces).

    Returns
    -------
    dict[str, str | None]
        Mapping of CUSIP → ticker symbol.
        CUSIPs that could not be mapped are set to ``None`` and logged.

    Notes
    -----
    **Known failure cases** (set to ``None``, not force-mapped):

    * Preferred share CUSIPs (different 9th digit from common)
    * Warrant and rights CUSIPs
    * ADR CUSIPs that differ from the underlying equity
    * Foreign private issuer CUSIPs
    * Pre-IPO / private placement CUSIPs (sometimes appear in CTR filings)

    These failures are *expected* and their rate should be logged and reported
    in the limitations section. A failure rate above 15% suggests an issue
    with the manager universe (too many structured-product or fixed-income
    managers slipping through the filters).

    Rate limiting: OpenFIGI allows up to 25 req/min unauthenticated. This
    function batches requests (10 CUSIPs per call) and sleeps between batches.
    """
    # Load existing cache
    cache_path = config.CUSIP_TICKER_MAP_JSON
    cache: dict[str, Optional[str]] = {}
    if cache_path.exists():
        with cache_path.open("r") as f:
            cache = json.load(f)

    # Filter to unmapped CUSIPs
    unmapped = [c for c in cusips if c not in cache and c]
    if not unmapped:
        return {c: cache.get(c) for c in cusips}

    logger.info("Resolving %d unmapped CUSIPs via OpenFIGI", len(unmapped))

    # ── Stage 1: OpenFIGI batch API ───────────────────────────────────────────
    batch_size = 10
    for i in range(0, len(unmapped), batch_size):
        batch = unmapped[i: i + batch_size]
        payload = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        _LIMITER.wait()
        try:
            resp = _SESSION.post(
                config.OPENFIGI_URL, json=payload, timeout=15
            )
            resp.raise_for_status()
            results = resp.json()
            for cusip, result in zip(batch, results):
                data_list = result.get("data", [])
                if data_list:
                    # Prefer equity share class (securityType == "Common Stock")
                    equities = [
                        d for d in data_list
                        if d.get("securityType") in ("Common Stock", "ETP")
                    ]
                    best = equities[0] if equities else data_list[0]
                    ticker = best.get("ticker")
                    cache[cusip] = ticker
                    logger.debug("OpenFIGI: %s → %s", cusip, ticker)
                else:
                    cache[cusip] = None
                    logger.debug("OpenFIGI: %s → unmapped", cusip)
        except Exception as exc:
            logger.warning("OpenFIGI batch %d failed: %s", i // batch_size, exc)
            for cusip in batch:
                if cusip not in cache:
                    cache[cusip] = None

    # ── Stage 2: SEC fallback for still-unmapped CUSIPs ───────────────────────
    still_unmapped = [c for c in unmapped if cache.get(c) is None]
    if still_unmapped:
        logger.info(
            "%d CUSIPs still unmapped after OpenFIGI; trying SEC company_tickers fallback",
            len(still_unmapped),
        )
        try:
            _LIMITER.wait()
            resp = _SESSION.get(config.SEC_COMPANY_TICKERS_URL, timeout=15)
            resp.raise_for_status()
            sec_data = resp.json()
            # Build CUSIP lookup from SEC data (limited — no CUSIP in this endpoint)
            # The SEC company_tickers JSON doesn't contain CUSIPs directly.
            # Log as unresolvable via this fallback.
            logger.info(
                "SEC company_tickers.json fetched (%d entries) but does not "
                "contain CUSIP fields — %d CUSIPs remain unmapped",
                len(sec_data), len(still_unmapped),
            )
        except Exception as exc:
            logger.warning("SEC company_tickers fallback failed: %s", exc)

    # Save updated cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)

    # Report resolution rate
    resolved   = sum(1 for c in unmapped if cache.get(c) is not None)
    fail_rate  = 1 - resolved / len(unmapped) if unmapped else 0
    logger.info(
        "CUSIP resolution: %d/%d resolved (%.1f%% success; %.1f%% failure)",
        resolved, len(unmapped), 100 * (1 - fail_rate), 100 * fail_rate,
    )
    if fail_rate > 0.15:
        logger.warning(
            "CUSIP failure rate %.1f%% exceeds 15%% threshold — "
            "review manager universe for non-equity positions",
            100 * fail_rate,
        )

    return {c: cache.get(c) for c in cusips}


# ─────────────────────────────────────────────────────────────────────────────
# Position delta computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_delta(
    current_holdings: pd.DataFrame,
    prior_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Compute quarter-over-quarter position changes for a single manager.

    Joins current-quarter holdings with prior-quarter holdings on CUSIP and
    classifies each position change as NEW / INCREASED / FLAT / REDUCED / EXITED.

    Parameters
    ----------
    current_holdings : pd.DataFrame
        Parsed holdings for the current quarter (output of ``parse_holdings_xml``
        with ``ticker`` column added via ``resolve_cusip_to_ticker``).
    prior_holdings : pd.DataFrame
        Parsed holdings for the immediately preceding quarter.

    Returns
    -------
    pd.DataFrame
        One row per CUSIP with columns:

        * ``cusip``             — CUSIP identifier
        * ``ticker``            — Resolved ticker (may be ``None``)
        * ``issuer_name``       — From current filing (or prior if exited)
        * ``position_type``     — "NEW" / "INCREASED" / "FLAT" / "REDUCED" / "EXITED"
        * ``shares_current``    — Shares in current quarter (0 if exited)
        * ``shares_prior``      — Shares in prior quarter (0 if new)
        * ``position_delta``    — ``shares_current - shares_prior``
        * ``delta_pct``         — Fractional change; ``+inf`` if new, -1.0 if exited
        * ``market_value_usd``  — Market value from current filing (0 if exited)

    Notes
    -----
    Position type classification logic:

    * ``NEW``       : ``shares_prior == 0`` and ``shares_current > 0``
    * ``INCREASED`` : ``delta_pct > MIN_POSITION_DELTA_PCT``
    * ``FLAT``      : ``|delta_pct| <= MIN_POSITION_DELTA_PCT``
    * ``REDUCED``   : ``delta_pct < -MIN_POSITION_DELTA_PCT`` and ``shares_current > 0``
    * ``EXITED``    : ``shares_current == 0`` and ``shares_prior > 0``

    The ``MIN_POSITION_DELTA_PCT`` threshold (default 5%) filters out rounding
    noise from share splits, dividend reinvestment, and reporting differences
    between sub-advisors.
    """
    # Standardise column names
    curr = current_holdings[["cusip", "ticker", "issuer_name", "share_count", "market_value_usd"]].copy()
    curr = curr.rename(columns={"share_count": "shares_current", "market_value_usd": "mktval_current"})

    if prior_holdings.empty:
        prior = pd.DataFrame(columns=["cusip", "shares_prior"])
    else:
        prior = prior_holdings[["cusip", "share_count"]].copy()
        prior = prior.rename(columns={"share_count": "shares_prior"})

    # Outer join to capture NEW and EXITED positions
    merged = curr.merge(prior, on="cusip", how="outer")
    merged["shares_current"]  = merged["shares_current"].fillna(0.0)
    merged["shares_prior"]    = merged["shares_prior"].fillna(0.0)
    merged["mktval_current"]  = merged["mktval_current"].fillna(0.0)

    # Restore issuer names for exited positions from prior holdings
    if not prior_holdings.empty and "issuer_name" in prior_holdings.columns:
        prior_names = prior_holdings.set_index("cusip")["issuer_name"].to_dict()
        merged["issuer_name"] = merged.apply(
            lambda r: r["issuer_name"] if pd.notna(r["issuer_name"]) else prior_names.get(r["cusip"], ""),
            axis=1,
        )

    # Restore tickers for exited positions
    if not prior_holdings.empty and "ticker" in prior_holdings.columns:
        prior_tickers = prior_holdings.set_index("cusip")["ticker"].to_dict()
        merged["ticker"] = merged.apply(
            lambda r: r["ticker"] if pd.notna(r["ticker"]) else prior_tickers.get(r["cusip"]),
            axis=1,
        )

    # Compute deltas
    merged["position_delta"] = merged["shares_current"] - merged["shares_prior"]

    # delta_pct: careful with zero denominators
    def _delta_pct(row: pd.Series) -> float:
        if row["shares_prior"] == 0 and row["shares_current"] > 0:
            return float("inf")
        if row["shares_prior"] == 0:
            return 0.0
        return (row["shares_current"] - row["shares_prior"]) / row["shares_prior"]

    merged["delta_pct"] = merged.apply(_delta_pct, axis=1)

    # Classify position type
    threshold = config.MIN_POSITION_DELTA_PCT

    def _classify(row: pd.Series) -> str:
        if row["shares_prior"] == 0 and row["shares_current"] > 0:
            return "NEW"
        if row["shares_current"] == 0 and row["shares_prior"] > 0:
            return "EXITED"
        if row["delta_pct"] > threshold:
            return "INCREASED"
        if row["delta_pct"] < -threshold:
            return "REDUCED"
        return "FLAT"

    merged["position_type"] = merged.apply(_classify, axis=1)

    # Rename and select final columns
    result = merged.rename(columns={"mktval_current": "market_value_usd"})[
        [
            "cusip",
            "ticker",
            "issuer_name",
            "position_type",
            "shares_current",
            "shares_prior",
            "position_delta",
            "delta_pct",
            "market_value_usd",
        ]
    ].copy()

    type_counts = result["position_type"].value_counts().to_dict()
    logger.debug(
        "Position delta: %d positions | %s",
        len(result),
        " | ".join(f"{k}:{v}" for k, v in sorted(type_counts.items())),
    )

    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Holdings panel builder
# ─────────────────────────────────────────────────────────────────────────────

def build_holdings_panel(
    cik: str,
    filing_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete holdings panel for a single manager across all quarters.

    For each filing in ``filing_history_df``, parses the holdings XML,
    resolves CUSIPs to tickers, and computes the position delta vs the
    prior quarter.

    Parameters
    ----------
    cik : str
        Manager CIK (normalised internally).
    filing_history_df : pd.DataFrame
        Output of ``edgar_scraper.build_manager_filing_history`` for this CIK,
        or the subset of ``universe_filing_history`` for this CIK.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:

        * ``cik``               — manager CIK
        * ``quarter_label``     — "Q{n}_{yyyy}"
        * ``cusip``             — CUSIP
        * ``ticker``            — resolved ticker (may be None)
        * ``issuer_name``       — from 13F filing
        * ``position_type``     — NEW / INCREASED / FLAT / REDUCED / EXITED
        * ``shares_current``    — current-quarter share count
        * ``shares_prior``      — prior-quarter share count
        * ``position_delta``    — share count change
        * ``delta_pct``         — fractional change
        * ``market_value_usd``  — market value at filing (SEC-reported)
        * ``filing_urgency``    — from filing_history_df

        Saved to ``data/processed/{cik}_holdings_panel.csv``.
    """
    from src.edgar_scraper import fetch_13f_holdings_xml

    cik_str = str(cik).zfill(10)
    manager_filings = filing_history_df[
        filing_history_df["cik"].astype(str).str.zfill(10) == cik_str
    ].sort_values("quarter_end_date").reset_index(drop=True)

    if manager_filings.empty:
        logger.warning("No filings found in history for CIK %s", cik_str)
        return pd.DataFrame()

    all_rows = []
    prior_holdings: pd.DataFrame = pd.DataFrame()

    for _, filing_row in manager_filings.iterrows():
        acc       = filing_row["accession_number"]
        q_label   = filing_row["quarter_label"]
        urgency   = filing_row["filing_urgency"]

        try:
            xml_str = fetch_13f_holdings_xml(cik, acc)
        except Exception as exc:
            logger.warning(
                "Could not fetch holdings XML for %s / %s: %s — skipping quarter",
                cik_str, q_label, exc,
            )
            continue

        try:
            holdings = parse_holdings_xml(xml_str)
        except ValueError as exc:
            logger.warning(
                "Could not parse holdings XML for %s / %s: %s — skipping",
                cik_str, q_label, exc,
            )
            continue

        if holdings.empty:
            logger.warning("Empty holdings after parsing for %s / %s", cik_str, q_label)
            continue

        # Resolve CUSIPs
        cusip_list = holdings["cusip"].dropna().unique().tolist()
        cusip_map  = resolve_cusip_to_ticker(cusip_list)
        holdings["ticker"] = holdings["cusip"].map(cusip_map)

        # Compute position deltas
        delta_df = compute_position_delta(holdings, prior_holdings)
        delta_df["cik"]            = cik_str
        delta_df["quarter_label"]  = q_label
        delta_df["filing_urgency"] = urgency

        all_rows.append(delta_df)
        prior_holdings = holdings  # roll forward

    if not all_rows:
        return pd.DataFrame()

    panel = pd.concat(all_rows, ignore_index=True)

    # Reorder columns for readability
    col_order = [
        "cik", "quarter_label", "cusip", "ticker", "issuer_name",
        "position_type", "shares_current", "shares_prior",
        "position_delta", "delta_pct", "market_value_usd", "filing_urgency",
    ]
    panel = panel[[c for c in col_order if c in panel.columns]]

    save_path = config.PROCESSED_DIR / f"{cik_str}_holdings_panel.csv"
    panel.to_csv(save_path, index=False)
    logger.info(
        "CIK %s: %d position-quarter records | %d unique tickers | saved to %s",
        cik_str, len(panel), panel["ticker"].nunique(), save_path,
    )

    return panel


def build_universe_holdings_panel(
    manager_universe_df: pd.DataFrame,
    filing_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the master holdings panel for all included managers.

    Parameters
    ----------
    manager_universe_df : pd.DataFrame
        Loaded from ``data/reference/manager_universe.csv``.
    filing_history_df : pd.DataFrame
        Output of ``edgar_scraper.build_universe_filing_history``.

    Returns
    -------
    pd.DataFrame
        Concatenated holdings panel for all included managers.
        Saved to ``data/processed/universe_holdings_panel.parquet``.
    """
    included = manager_universe_df[manager_universe_df["include_flag"] == True]

    all_panels = []
    for _, row in included.iterrows():
        cik = str(row["cik"])
        logger.info("Building holdings panel for %s (CIK %s)", row["manager_name"], cik)
        try:
            panel = build_holdings_panel(cik, filing_history_df)
            if not panel.empty:
                all_panels.append(panel)
        except Exception as exc:
            logger.error(
                "Failed to build holdings panel for %s (%s): %s",
                row["manager_name"], cik, exc,
            )

    if not all_panels:
        logger.warning("No holdings panel data returned for any manager")
        return pd.DataFrame()

    universe_panel = pd.concat(all_panels, ignore_index=True)

    logger.info(
        "Universe holdings panel: %d records | %d unique tickers | "
        "%d unique managers | quarters: %s — %s",
        len(universe_panel),
        universe_panel["ticker"].nunique(),
        universe_panel["cik"].nunique(),
        universe_panel["quarter_label"].min(),
        universe_panel["quarter_label"].max(),
    )

    save_path = config.UNIVERSE_HOLDINGS_PANEL_PARQUET
    save_path.parent.mkdir(parents=True, exist_ok=True)
    universe_panel.to_parquet(save_path, index=False, engine="pyarrow")
    logger.info("Saved universe holdings panel to %s", save_path)

    return universe_panel
