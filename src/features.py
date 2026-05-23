"""
features.py — Conviction score construction and cross-manager signal aggregation.

This module is the mathematical core of the strategy. It implements:

1. ``compute_filing_urgency``          — normalise filing timing to [0, 1]
2. ``compute_conviction_score``        — the core formula: urgency × sign × log(1 + |Δ|)
3. ``compute_cross_manager_signal``    — aggregate across managers per ticker-quarter
4. ``apply_ctr_filter``               — flag CTR-contaminated positions
5. ``compute_manager_filing_pattern_baseline`` — manager-demeaned urgency signal

Canonical formula (implemented ONCE here; imported everywhere else)
-------------------------------------------------------------------
For manager *i*, quarter *t*, ticker *k*:

    conviction_score(i, t, k) =
        filing_urgency(i, t)
        × sign(position_delta(i, t, k))
        × log(1 + |delta_pct(i, t, k)|)

Where:
    filing_urgency = (filing_deadline − filing_date).days / 45   ∈ [0, 1]
    sign(delta)    = +1 for NEW/INCREASED, −1 for REDUCED/EXITED, 0 for FLAT
    log            = natural logarithm

The log transformation compresses the delta_pct distribution (which is
right-skewed due to position initiations) and ensures that a 100% increase
contributes log(2) ≈ 0.693 to the score, not an unbounded raw number.

Manager-demeaned variant (preferred in production):
    urgency_deviation(i, t) = filing_urgency(i, t) − mean(filing_urgency(i, 1..t-1))
    conviction_score_adj(i, t, k) = urgency_deviation(i, t) × sign × log(1 + |Δ|)

The demeaned variant controls for managers who are structurally early or late
filers regardless of conviction. Example: a compliance-conservative manager
who always files on day 10 shows urgency ≈ 0.78 but is not making a conviction
statement. Demeaning removes this structural bias.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Filing urgency
# ─────────────────────────────────────────────────────────────────────────────

def compute_filing_urgency(
    filing_history_df: pd.DataFrame,
    deadlines_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute and add filing_urgency and urgency_bucket to the filing history.

    ``filing_urgency`` = (filing_deadline − filing_date_only).days / FILING_DEADLINE_DAYS

    Clamped to [0, 1]. Values > 1 are set to 1.0 with a warning (late filers).

    Parameters
    ----------
    filing_history_df : pd.DataFrame
        Output of ``edgar_scraper.build_universe_filing_history``.
        Must contain columns: ``quarter_label``, ``filing_date_only``.
    deadlines_df : pd.DataFrame
        Loaded from ``data/reference/filing_deadlines.csv``.
        Must contain columns: ``quarter_label``, ``filing_deadline``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:

        * ``filing_urgency``  — float in [0, 1]; higher = filed later = higher conviction
        * ``urgency_bucket``  — "EARLY" (< 0.33) / "MIDDLE" (0.33–0.75) / "LATE" (> 0.75)

    Notes
    -----
    If ``filing_urgency`` already exists in the input (populated by
    ``edgar_scraper``), this function validates the values rather than
    recomputing from scratch. Recomputation happens only for rows with NaN urgency.
    """
    df = filing_history_df.copy()

    # Merge deadline calendar if filing_urgency is missing or needs validation
    deadlines_df = deadlines_df.copy()
    deadlines_df["filing_deadline"] = pd.to_datetime(deadlines_df["filing_deadline"])

    if "filing_urgency" not in df.columns or df["filing_urgency"].isna().any():
        logger.info("Computing filing_urgency from deadline calendar merge")
        df = df.merge(
            deadlines_df[["quarter_label", "filing_deadline"]],
            on="quarter_label",
            how="left",
            suffixes=("", "_cal"),
        )
        df["filing_date_only"] = pd.to_datetime(df["filing_date_only"])
        if "filing_deadline_cal" in df.columns:
            df["filing_deadline"] = df["filing_deadline_cal"].combine_first(
                df.get("filing_deadline")
            )
            df = df.drop(columns=["filing_deadline_cal"])

        df["filing_urgency"] = (
            (df["filing_deadline"] - df["filing_date_only"]).dt.days
            / config.FILING_DEADLINE_DAYS
        )

    # Clamp and warn
    over_one = df["filing_urgency"] > 1.0
    under_zero = df["filing_urgency"] < 0.0
    if over_one.any():
        logger.warning(
            "%d filings with urgency > 1.0 (filed before quarter end?); clamping to 1.0",
            over_one.sum(),
        )
        df.loc[over_one, "filing_urgency"] = 1.0
    if under_zero.any():
        logger.warning(
            "%d filings with urgency < 0.0 (filed after deadline); clamping to 1.0",
            under_zero.sum(),
        )
        df.loc[under_zero, "filing_urgency"] = 1.0

    # Bucket assignment
    df["urgency_bucket"] = pd.cut(
        df["filing_urgency"],
        bins=[0.0, config.URGENCY_EARLY_THRESHOLD, config.URGENCY_LATE_THRESHOLD, 1.0],
        labels=["EARLY", "MIDDLE", "LATE"],
        include_lowest=True,
        right=True,
    ).astype(str)

    logger.info(
        "Urgency distribution — EARLY: %d | MIDDLE: %d | LATE: %d",
        (df["urgency_bucket"] == "EARLY").sum(),
        (df["urgency_bucket"] == "MIDDLE").sum(),
        (df["urgency_bucket"] == "LATE").sum(),
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Conviction score (canonical implementation)
# ─────────────────────────────────────────────────────────────────────────────

def _sign_from_position_type(position_type: str) -> int:
    """Return +1 for bullish moves, -1 for bearish, 0 for flat."""
    if position_type in ("NEW", "INCREASED"):
        return +1
    if position_type in ("REDUCED", "EXITED"):
        return -1
    return 0  # FLAT


def compute_conviction_score(holdings_panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-position conviction score.

    Implements the canonical formula (defined once here, imported everywhere):

        conviction_score = filing_urgency × sign(position_delta) × log(1 + |delta_pct|)

    Where ``sign`` is +1 for NEW/INCREASED, −1 for REDUCED/EXITED, 0 for FLAT.

    Parameters
    ----------
    holdings_panel_df : pd.DataFrame
        Output of ``filing_parser.build_universe_holdings_panel``.
        Required columns: ``filing_urgency``, ``position_type``, ``delta_pct``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added column ``conviction_score``.

    Notes
    -----
    **Worked numerical examples** (matching the strategy specification):

    *Example 1 — Late filer, moderate increase:*
        Manager files on day 40 of 45 → urgency = (45−40)/45 = 0.11... wait:
        filing_urgency = (deadline − filing_date).days / 45
        If deadline is day 45 and manager files on day 40:
            days_remaining = 45 − 40 = 5 → urgency = 5/45 = 0.11
        That's an EARLY filer. Correction:
        filing_urgency = days_remaining / 45
        Day 5 filing: 40 days remain → urgency = 40/45 = 0.89 (LATE)
        Day 40 filing: 5 days remain → urgency = 5/45 = 0.11 (EARLY)

    *Example 2 — Late filer (high urgency), 25% position increase:*
        Manager files on the 5th day of the 45-day window (40 days remain):
            filing_urgency = 40/45 = 0.889
        Position increased by 25% (delta_pct = 0.25):
            conviction = 0.889 × (+1) × log(1.25) = 0.889 × 0.223 = +0.198

    *Example 3 — Early filer (low urgency), new position opened:*
        Manager files on day 40 of 45 (5 days remain):
            filing_urgency = 5/45 = 0.111
        New position (delta_pct = +inf → use log(2.0) as practical proxy for doubling):
            conviction = 0.111 × (+1) × log(2.0) = 0.111 × 0.693 = +0.076

        Same bullish action, dramatically different signal strength.
        The late filer's conviction score is 2.6× higher.

    *Example 4 — FLAT position:*
        Regardless of filing_urgency, conviction_score = 0.0 by construction.

    The log transform is critical: without it, a 1000% position increase
    would dominate the signal even for an early filer. The log ensures that
    the filing timing dimension (urgency) remains the primary driver of
    signal strength, not the raw position change magnitude.
    """
    df = holdings_panel_df.copy()

    # Sign vector
    df["_sign"] = df["position_type"].apply(_sign_from_position_type)

    # |delta_pct| with inf capped at a large value for NEW positions
    # log(1 + inf) → inf; cap at log(1 + 10) = log(11) ≈ 2.40 for new positions
    # This prevents any single "new" position from dominating with an infinite score
    MAX_DELTA_PCT_CAP = 10.0

    def _safe_delta_pct(dp: float) -> float:
        if not np.isfinite(dp):
            return MAX_DELTA_PCT_CAP
        return min(abs(dp), MAX_DELTA_PCT_CAP)

    df["_abs_delta_capped"] = df["delta_pct"].apply(_safe_delta_pct)

    # Compute conviction score
    df["conviction_score"] = (
        df["filing_urgency"]
        * df["_sign"]
        * np.log1p(df["_abs_delta_capped"])
    )

    # FLAT positions always get zero
    flat_mask = df["position_type"] == "FLAT"
    df.loc[flat_mask, "conviction_score"] = 0.0

    # Drop helper columns
    df = df.drop(columns=["_sign", "_abs_delta_capped"])

    # Summary stats
    logger.info(
        "Conviction scores: mean=%.4f std=%.4f "
        "positive=%d negative=%d zero=%d",
        df["conviction_score"].mean(),
        df["conviction_score"].std(),
        (df["conviction_score"] > 0).sum(),
        (df["conviction_score"] < 0).sum(),
        (df["conviction_score"] == 0).sum(),
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cross-manager signal aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_manager_signal(holdings_panel_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual conviction scores into a cross-manager signal.

    For each unique (quarter_label, ticker) pair, sums conviction scores
    across all managers and applies the minimum-manager filter.

    Parameters
    ----------
    holdings_panel_df : pd.DataFrame
        Output of ``compute_conviction_score``.
        Required columns: ``quarter_label``, ``ticker``, ``conviction_score``.

    Returns
    -------
    pd.DataFrame
        Signal DataFrame with columns:

        * ``quarter_label``         — quarter identifier
        * ``ticker``                — equity ticker
        * ``aggregated_conviction`` — Σ conviction_score across managers
        * ``manager_count``         — managers with non-zero conviction
        * ``bull_count``            — managers with positive conviction
        * ``bear_count``            — managers with negative conviction
        * ``net_direction``         — +1 (bull), -1 (bear), 0 (balanced)
        * ``signal_quality``        — "HIGH", "MIXED", or "THIN"

    Notes
    -----
    Ticker-quarters with ``manager_count < CROSS_MANAGER_MIN_COUNT`` are
    excluded. With only 2 managers agreeing, we cannot distinguish genuine
    consensus from correlated positions in a popular stock (which would be
    noise, not signal).

    Signal quality classification:

    * ``HIGH``  — all managers agree on direction (all bull or all bear)
    * ``THIN``  — exactly ``CROSS_MANAGER_MIN_COUNT`` managers; statistically weak
    * ``MIXED`` — both bull and bear managers present; conflicted signal
    """
    df = holdings_panel_df.copy()

    # Drop rows with no ticker (unresolved CUSIP)
    df = df.dropna(subset=["ticker"])
    df = df[df["ticker"].str.strip() != ""]

    # Only rows with non-zero conviction
    df_nonzero = df[df["conviction_score"] != 0.0].copy()

    agg = (
        df_nonzero.groupby(["quarter_label", "ticker"])
        .agg(
            aggregated_conviction=("conviction_score", "sum"),
            manager_count=("cik", "nunique"),
            bull_count=("conviction_score", lambda x: (x > 0).sum()),
            bear_count=("conviction_score", lambda x: (x < 0).sum()),
        )
        .reset_index()
    )

    # Apply minimum manager count filter
    n_before = len(agg)
    agg = agg[agg["manager_count"] >= config.CROSS_MANAGER_MIN_COUNT]
    n_filtered = n_before - len(agg)
    logger.info(
        "Cross-manager aggregation: %d ticker-quarters before filter, "
        "%d after (removed %d with < %d managers)",
        n_before, len(agg), n_filtered, config.CROSS_MANAGER_MIN_COUNT,
    )

    # Net direction
    agg["net_direction"] = np.sign(agg["aggregated_conviction"]).astype(int)

    # Signal quality
    def _quality(row: pd.Series) -> str:
        if row["manager_count"] == config.CROSS_MANAGER_MIN_COUNT:
            return "THIN"
        if row["bull_count"] == 0 or row["bear_count"] == 0:
            return "HIGH"
        return "MIXED"

    agg["signal_quality"] = agg.apply(_quality, axis=1)

    quality_counts = agg["signal_quality"].value_counts().to_dict()
    logger.info(
        "Signal quality distribution: %s",
        " | ".join(f"{k}:{v}" for k, v in sorted(quality_counts.items())),
    )

    return agg.sort_values(
        ["quarter_label", "aggregated_conviction"], ascending=[True, False]
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CTR filter
# ─────────────────────────────────────────────────────────────────────────────

def apply_ctr_filter(
    signal_df: pd.DataFrame,
    ctr_log_df: pd.DataFrame,
) -> pd.DataFrame:
    """Flag ticker-quarters where a confidential treatment request was active.

    **CTR filtering is the strategy's primary guard against selection bias.**

    Positions that managers chose to conceal are systematically the
    highest-conviction ideas. An early filing with a CTR has an entirely
    different interpretation than an early filing without one: the manager
    may be filing early precisely *because* they are still accumulating a
    hidden position and want to minimise market impact.

    Including CTR-affected disclosures as if they were ordinary positions
    would corrupt the filing_urgency signal and likely overstate strategy
    performance (because the eventual disclosure of hidden positions adds
    information that was not available at signal-construction time).

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of ``compute_cross_manager_signal``.
        Required columns: ``quarter_label``, ``ticker``.
    ctr_log_df : pd.DataFrame
        Loaded from ``data/reference/ctr_log.csv``.
        Required columns: ``quarter_label``, ``ticker_cusip``.

    Returns
    -------
    pd.DataFrame
        ``signal_df`` enriched with:

        * ``ctr_affected`` — bool; True if any CTR was active for this
          ticker-quarter across any manager in the universe
        * ``ctr_note`` — human-readable description of the CTR event

    Notes
    -----
    ``ctr_log_df["ticker_cusip"]`` may contain either a ticker symbol or a
    CUSIP (some CTR records are only identified by CUSIP at disclosure time).
    This function attempts ticker matching first; CUSIP matching would require
    joining through the CUSIP map (done in the notebook analysis).
    """
    df = signal_df.copy()
    df["ctr_affected"] = False
    df["ctr_note"] = ""

    # Build set of (quarter_label, ticker) pairs from CTR log
    ctr_df = ctr_log_df.copy()
    ctr_pairs = set(
        zip(ctr_df["quarter_label"].astype(str), ctr_df["ticker_cusip"].astype(str))
    )

    if not ctr_pairs:
        logger.info("CTR log is empty — no positions flagged")
        return df

    def _is_ctr_affected(row: pd.Series) -> tuple[bool, str]:
        key = (str(row["quarter_label"]), str(row.get("ticker", "")))
        if key in ctr_pairs:
            ctr_rows = ctr_df[
                (ctr_df["quarter_label"] == row["quarter_label"])
                & (ctr_df["ticker_cusip"] == row.get("ticker", ""))
            ]
            if not ctr_rows.empty:
                r = ctr_rows.iloc[0]
                note = (
                    f"CTR by {r['manager_name']} in {r['quarter_label']}; "
                    f"disclosed: {r['ctr_subsequently_disclosed']} "
                    f"(lag {r['disclosure_lag_quarters']} qtrs)"
                )
                return True, note
        return False, ""

    results = df.apply(_is_ctr_affected, axis=1, result_type="expand")
    df["ctr_affected"] = results[0]
    df["ctr_note"]     = results[1]

    n_flagged = df["ctr_affected"].sum()
    logger.info(
        "CTR filter: %d / %d ticker-quarters flagged as CTR-affected",
        n_flagged, len(df),
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Manager-demeaned urgency signal
# ─────────────────────────────────────────────────────────────────────────────

def compute_manager_filing_pattern_baseline(
    filing_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute each manager's historical filing urgency baseline and deviation.

    Manager-demeaned urgency is the preferred signal variant because it
    controls for structural filing style differences across managers.

    **Rationale**: A manager who normally files on day 10 of the 45-day
    window (urgency ≈ 0.78) filing on day 5 (urgency ≈ 0.89) is signalling
    something different from a manager who always files on day 5. The
    deviation from the manager's own baseline — not the raw urgency — is
    the true conviction signal.

    This is analogous to earnings surprise (actual vs consensus) versus
    the absolute earnings level. The deviation is what moves prices.

    Parameters
    ----------
    filing_history_df : pd.DataFrame
        Output of ``edgar_scraper.build_universe_filing_history`` or
        ``compute_filing_urgency``. Required columns:
        ``cik``, ``quarter_label``, ``filing_urgency``, ``quarter_end_date``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:

        * ``manager_avg_urgency``    — expanding mean urgency (prior quarters only)
        * ``manager_urgency_std``    — expanding std (prior quarters only)
        * ``urgency_deviation``      — current_urgency − manager_avg_urgency
        * ``conviction_score_adj``   — see note below

        Note: ``conviction_score_adj`` replaces the raw urgency in the
        conviction formula with ``urgency_deviation``. Negative values indicate
        a manager filing *earlier* than their own typical pattern — a bearish
        signal even if absolute urgency is moderate.

    Notes
    -----
    **Minimum history requirement**: The expanding mean uses at least
    ``config.CONVICTION_SCORE_WINDOW`` (default 2) quarters of prior data.
    Managers with fewer than 2 prior filings receive ``NaN`` for
    ``manager_avg_urgency`` and ``urgency_deviation``. These rows are
    excluded from the demeaned signal but retained for the raw urgency signal.
    """
    df = filing_history_df.copy()
    df["quarter_end_date"] = pd.to_datetime(df["quarter_end_date"])
    df = df.sort_values(["cik", "quarter_end_date"]).reset_index(drop=True)

    results = []
    for cik, group in df.groupby("cik"):
        group = group.sort_values("quarter_end_date").reset_index(drop=True)
        avg_urgencies = []
        std_urgencies = []

        for i in range(len(group)):
            if i < config.CONVICTION_SCORE_WINDOW:
                avg_urgencies.append(np.nan)
                std_urgencies.append(np.nan)
            else:
                prior = group.iloc[:i]["filing_urgency"]
                avg_urgencies.append(prior.mean())
                std_urgencies.append(prior.std() if len(prior) > 1 else np.nan)

        group["manager_avg_urgency"] = avg_urgencies
        group["manager_urgency_std"] = std_urgencies
        group["urgency_deviation"] = group["filing_urgency"] - group["manager_avg_urgency"]
        results.append(group)

    df = pd.concat(results, ignore_index=True)

    n_baseline_available = df["urgency_deviation"].notna().sum()
    logger.info(
        "Manager baseline computed: %d / %d filings have sufficient history "
        "(min %d quarters)",
        n_baseline_available, len(df), config.CONVICTION_SCORE_WINDOW,
    )

    return df


def recompute_conviction_adj(
    holdings_panel_df: pd.DataFrame,
    filing_history_with_baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute conviction score using manager-demeaned urgency (urgency_deviation).

    Merges the urgency deviation back into the holdings panel and computes
    ``conviction_score_adj`` using the same canonical formula but substituting
    ``urgency_deviation`` for raw ``filing_urgency``.

    Parameters
    ----------
    holdings_panel_df : pd.DataFrame
        Holdings panel with ``conviction_score`` already computed.
    filing_history_with_baseline_df : pd.DataFrame
        Output of ``compute_manager_filing_pattern_baseline``.
        Must contain: ``cik``, ``quarter_label``, ``urgency_deviation``.

    Returns
    -------
    pd.DataFrame
        Holdings panel enriched with ``urgency_deviation`` and
        ``conviction_score_adj`` columns.

    Notes
    -----
    Rows where ``urgency_deviation`` is NaN (insufficient manager history)
    will have ``conviction_score_adj = NaN``. These rows participate in
    the raw urgency signal but not the demeaned variant.
    """
    # Merge urgency deviation
    baseline = filing_history_with_baseline_df[
        ["cik", "quarter_label", "urgency_deviation", "manager_avg_urgency"]
    ].copy()
    df = holdings_panel_df.merge(baseline, on=["cik", "quarter_label"], how="left")

    sign_series = df["position_type"].apply(_sign_from_position_type)
    MAX_DELTA_PCT_CAP = 10.0
    abs_delta = df["delta_pct"].apply(
        lambda d: min(abs(d), MAX_DELTA_PCT_CAP) if np.isfinite(d) else MAX_DELTA_PCT_CAP
    )
    log_term = np.log1p(abs_delta)

    df["conviction_score_adj"] = df["urgency_deviation"] * sign_series * log_term

    # FLAT positions → zero regardless
    flat_mask = df["position_type"] == "FLAT"
    df.loc[flat_mask, "conviction_score_adj"] = 0.0

    logger.info(
        "Demeaned conviction scores: %d valid (%.1f%% of panel)",
        df["conviction_score_adj"].notna().sum(),
        100 * df["conviction_score_adj"].notna().mean(),
    )

    return df
