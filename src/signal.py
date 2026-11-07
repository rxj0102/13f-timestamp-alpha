"""
signal.py — Signal construction, staleness decay, and hypothesis testing.

Builds the final investable signal from the cross-manager conviction scores,
applies the exponential staleness decay, and runs statistical tests of the
core hypothesis: that filing urgency predicts forward equity returns.

Staleness decay
---------------
The conviction score captures information at the moment of filing. As time
passes after the filing date, the manager's position may have changed —
particularly for short-horizon traders. The staleness decay function reduces
signal weight as a function of time elapsed since filing:

    staleness_weight = filing_urgency × exp(−days_since_filing / 45)

The half-life of 45 days corresponds to one filing cycle. A position disclosed
at the last possible moment (urgency ≈ 1.0, days_since_filing = 0) starts with
maximum weight. At 45 days post-filing, the weight decays to urgency × e⁻¹ ≈
37% of the original. This decay is appropriate because quarterly 13F data is
stale by construction — it reflects positions as of the quarter end, not today.

Information Coefficient
-----------------------
The IC (Spearman rank correlation between conviction score and forward return)
is the primary out-of-sample validation metric. The expected hierarchy is:

    IC(demeaned urgency, HIGH quality) >
    IC(demeaned urgency, all quality)  >
    IC(raw urgency, HIGH quality)      >
    IC(raw urgency, all quality)

If this hierarchy does not hold, either the manager universe filter is
insufficiently tight or the sample period is too short for reliable inference.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Quarterly signal construction
# ─────────────────────────────────────────────────────────────────────────────

def build_quarterly_signal(
    signal_df: pd.DataFrame,
    quarter_label: str,
) -> pd.DataFrame:
    """Build the investable long/short signal for a single quarter.

    Filters, ranks, and assigns deciles to the cross-manager conviction scores
    for a specific quarter. Applies quality and CTR filters.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of ``features.apply_ctr_filter`` (which enriches
        ``compute_cross_manager_signal`` output).
        Required columns: ``quarter_label``, ``ticker``,
        ``aggregated_conviction``, ``signal_quality``, ``ctr_affected``.
    quarter_label : str
        Target quarter, e.g. "Q2_2021".

    Returns
    -------
    pd.DataFrame
        Positioned signal DataFrame with columns:

        * ``ticker``                — equity ticker
        * ``signal_decile``         — 1 (lowest conviction) to 10 (highest)
        * ``aggregated_conviction`` — raw conviction aggregate
        * ``manager_count``         — number of agreeing managers
        * ``signal_quality``        — HIGH / MIXED / THIN
        * ``position_side``         — "LONG" (decile 9-10), "SHORT" (decile 1-2), or "NEUTRAL"
        * ``ctr_affected``          — bool; CTR-contaminated positions excluded from LONG/SHORT

    Notes
    -----
    Signal selection rules:

    * **LONG candidates** : decile 9 or 10, signal_quality ≠ "THIN",
      ctr_affected == False
    * **SHORT candidates**: decile 1 or 2, signal_quality ≠ "THIN",
      ctr_affected == False
    * **NEUTRAL**         : all others (decile 3-8, THIN quality, or CTR-affected)

    THIN-quality signals are excluded from the investable signal because with
    exactly ``CROSS_MANAGER_MIN_COUNT`` managers, a single manager's outlier
    position can dominate the aggregated_conviction — insufficient consensus
    for confident position-taking.
    """
    qdf = signal_df[signal_df["quarter_label"] == quarter_label].copy()

    if qdf.empty:
        logger.warning("No signal data for quarter %s", quarter_label)
        return pd.DataFrame()

    # Rank into deciles
    qdf["signal_decile"] = pd.qcut(
        qdf["aggregated_conviction"].rank(method="first"),
        q=10,
        labels=range(1, 11),
    ).astype(int)

    # Assign sides
    def _side(row: pd.Series) -> str:
        if row["ctr_affected"]:
            return "NEUTRAL"
        if row["signal_quality"] == "THIN":
            return "NEUTRAL"
        if row["signal_decile"] >= 9:
            return "LONG"
        if row["signal_decile"] <= 2:
            return "SHORT"
        return "NEUTRAL"

    qdf["position_side"] = qdf.apply(_side, axis=1)

    # Preserve date and grouping columns for backtest entry-date computation
    extra_cols = ["quarter_label", "filing_date_only", "filing_datetime",
                  "quarter_end_date", "urgency_bucket"]
    base_cols = [
        "ticker", "signal_decile", "aggregated_conviction",
        "manager_count", "signal_quality", "position_side", "ctr_affected",
    ]
    cols = base_cols + [c for c in extra_cols if c in qdf.columns and c not in base_cols]
    result = qdf[[c for c in cols if c in qdf.columns]].sort_values(
        "signal_decile", ascending=False
    )

    logger.info(
        "Quarter %s signal: LONG=%d | SHORT=%d | NEUTRAL=%d (total=%d)",
        quarter_label,
        (result["position_side"] == "LONG").sum(),
        (result["position_side"] == "SHORT").sum(),
        (result["position_side"] == "NEUTRAL").sum(),
        len(result),
    )

    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Staleness decay
# ─────────────────────────────────────────────────────────────────────────────

def compute_signal_staleness_decay(
    signal_df: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Apply exponential staleness decay to reduce signal weight over time.

    The conviction score reflects the manager's disclosed position as of the
    quarter end. As calendar time passes after the filing date, the disclosed
    position becomes increasingly stale — the manager may have repositioned.

    Staleness weight formula:
        staleness_weight = filing_urgency × exp(−days_since_filing / 45)

    Where ``days_since_filing`` is measured from the filing date to ``as_of_date``
    (default: today).

    Parameters
    ----------
    signal_df : pd.DataFrame
        Cross-manager signal DataFrame (output of ``apply_ctr_filter``).
        Required columns: ``filing_urgency`` (or ``aggregated_conviction``),
        ``filing_date_only`` (or ``filing_datetime``).
    as_of_date : pd.Timestamp, optional
        Reference date for computing ``days_since_filing``.
        Defaults to today (``pd.Timestamp.now()``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with:

        * ``days_since_filing``  — calendar days from filing date to as_of_date
        * ``staleness_weight``   — urgency × exp(−days/45), in [0, urgency]
        * ``decayed_conviction`` — aggregated_conviction × exp(−days/45)

    Notes
    -----
    **Half-life interpretation**: At exactly 45 days post-filing, the
    staleness weight equals urgency × e⁻¹ ≈ 37% of the original.
    At 90 days (two quarters), it falls to urgency × e⁻² ≈ 14%.
    This decay schedule implies that 13F signals are essentially exhausted
    by the time the next quarter's filings begin (≈ 90 days from the prior
    quarter-end filing deadline).

    **Sizing implication**: Position sizes in the live signal should scale
    with ``staleness_weight``, not ``aggregated_conviction`` alone. A high-
    conviction signal filed 80 days ago deserves less capital than a moderate-
    conviction signal filed 10 days ago.
    """
    df = signal_df.copy()

    if as_of_date is None:
        as_of_date = pd.Timestamp.now().normalize()

    # Find the filing date column
    if "filing_date_only" in df.columns:
        df["_filing_date"] = pd.to_datetime(df["filing_date_only"])
    elif "filing_datetime" in df.columns:
        df["_filing_date"] = pd.to_datetime(df["filing_datetime"]).dt.normalize()
    else:
        logger.warning(
            "No filing date column found; using quarter end date for staleness calc"
        )
        df["_filing_date"] = pd.to_datetime(df.get("quarter_end_date", as_of_date))

    df["days_since_filing"] = (as_of_date - df["_filing_date"]).dt.days.clip(lower=0)

    # Decay factor
    decay = np.exp(-df["days_since_filing"] / config.SIGNAL_HOLDING_DAYS)

    # Staleness weight: if filing_urgency available at ticker level, use it
    # Otherwise fall back to 1.0 (decay only, no urgency scaling)
    if "filing_urgency" in df.columns:
        df["staleness_weight"] = df["filing_urgency"] * decay
    else:
        df["staleness_weight"] = decay

    # Decayed conviction
    if "aggregated_conviction" in df.columns:
        df["decayed_conviction"] = df["aggregated_conviction"] * decay

    df = df.drop(columns=["_filing_date"])

    logger.info(
        "Staleness decay as of %s: mean days_since_filing=%.1f | "
        "mean staleness_weight=%.3f",
        as_of_date.date(),
        df["days_since_filing"].mean(),
        df["staleness_weight"].mean(),
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistical hypothesis testing
# ─────────────────────────────────────────────────────────────────────────────

def _compute_forward_return(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon_days: int,
    filing_date_col: str = "filing_date_only",
) -> pd.DataFrame:
    """Merge forward returns onto the signal DataFrame.

    Internal helper used by ``run_signal_ttest`` and ``compute_information_coefficient``.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal rows; must contain ``ticker`` and ``filing_date_col``.
    price_df : pd.DataFrame
        Daily adjusted close prices. Expected format: DatetimeIndex rows,
        ticker columns (wide format from ``yfinance``).
    horizon_days : int
        Trading days forward to compute the return window.
    filing_date_col : str
        Column name for the event (filing) date in ``signal_df``.

    Returns
    -------
    pd.DataFrame
        ``signal_df`` with added ``forward_return`` column.
        Rows where price data is unavailable receive NaN.
    """
    df = signal_df.copy()

    if price_df is None or price_df.empty:
        logger.warning("price_df is empty; forward returns will be NaN")
        df["forward_return"] = np.nan
        return df

    fwd_returns = []
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    for _, row in df.iterrows():
        ticker = row.get("ticker")
        event_date = pd.to_datetime(row.get(filing_date_col))

        if ticker is None or ticker not in price_df.columns or pd.isna(event_date):
            fwd_returns.append(np.nan)
            continue

        # Find the next available price on or after the event date
        future = price_df.loc[price_df.index >= event_date, ticker].dropna()
        if len(future) < horizon_days + 1:
            fwd_returns.append(np.nan)
            continue

        entry_price = future.iloc[0]
        exit_price  = future.iloc[horizon_days]

        if entry_price == 0 or np.isnan(entry_price):
            fwd_returns.append(np.nan)
        else:
            fwd_returns.append((exit_price - entry_price) / entry_price)

    df["forward_return"] = fwd_returns
    n_valid = sum(1 for r in fwd_returns if not np.isnan(r))
    logger.debug(
        "Forward returns computed: %d / %d valid (horizon=%d days)",
        n_valid, len(df), horizon_days,
    )

    return df


def run_signal_ttest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon_days: int = 60,
) -> dict:
    """Two-sample t-test: top conviction decile vs bottom conviction decile.

    Tests the null hypothesis H₀: mean forward return is equal between
    high-conviction (top decile) and low-conviction (bottom decile) positions.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Positioned signal DataFrame (output of ``build_quarterly_signal``
        concatenated across quarters), with column ``signal_decile``.
    price_df : pd.DataFrame
        Daily adjusted close prices (wide format: tickers as columns).
    horizon_days : int, optional
        Return horizon in trading days. Default 60 (approximately one quarter).

    Returns
    -------
    dict
        Keys:

        * ``n_high``          — sample size, top decile
        * ``n_low``           — sample size, bottom decile
        * ``mean_high``       — mean forward return, top decile
        * ``mean_low``        — mean forward return, bottom decile
        * ``std_high``        — std, top decile
        * ``std_low``         — std, bottom decile
        * ``t_stat``          — t-statistic
        * ``p_value``         — two-tailed p-value
        * ``significant``     — True if p < 0.05
        * ``horizon_days``    — horizon used

    Notes
    -----
    Also tests raw vs demeaned urgency signal if ``conviction_score_adj`` is
    present in ``signal_df``. Prints a formatted comparison table to stdout.

    Sample size caveat: with ~40 managers × ~40 quarters × ~20 tickers per
    decile, expect roughly 800 observations per decile. This is adequate for
    t-test power at conventional significance levels but leaves limited degrees
    of freedom for cross-sectional dependence corrections.
    """
    df_with_returns = _compute_forward_return(signal_df, price_df, horizon_days)
    df_valid = df_with_returns.dropna(subset=["forward_return"])

    if "signal_decile" not in df_valid.columns:
        logger.error("signal_decile column not found; cannot run t-test")
        return {}

    high = df_valid[df_valid["signal_decile"] >= 9]["forward_return"]
    low  = df_valid[df_valid["signal_decile"] <= 2]["forward_return"]

    if len(high) < 5 or len(low) < 5:
        logger.warning("Insufficient observations for t-test (high=%d, low=%d)", len(high), len(low))
        return {"error": "insufficient_observations"}

    t_stat, p_value = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")

    result = {
        "n_high":       len(high),
        "n_low":        len(low),
        "mean_high":    high.mean(),
        "mean_low":     low.mean(),
        "std_high":     high.std(),
        "std_low":      low.std(),
        "t_stat":       t_stat,
        "p_value":      p_value,
        "significant":  p_value < 0.05,
        "horizon_days": horizon_days,
    }

    # Print formatted table
    print("\n" + "=" * 72)
    print(f"  T-Test: High vs Low Conviction — {horizon_days}d Forward Return")
    print("=" * 72)
    print(f"  {'Group':<12} {'N':>6} {'Mean Fwd Ret':>14} {'Std':>10} {'t-stat':>10} {'p-value':>10}")
    print("-" * 72)
    print(f"  {'HIGH (D9-10)':<12} {result['n_high']:>6} {result['mean_high']:>13.4%} "
          f"{result['std_high']:>10.4f} {result['t_stat']:>10.3f} {result['p_value']:>10.4f}")
    print(f"  {'LOW (D1-2)':<12} {result['n_low']:>6} {result['mean_low']:>13.4%} "
          f"{result['std_low']:>10.4f} {'':>10} {'':>10}")
    print("-" * 72)
    sig_str = "*** SIGNIFICANT (p < 0.05)" if result["significant"] else "NOT significant"
    print(f"  {sig_str}")
    print("=" * 72 + "\n")

    return result


def compute_information_coefficient(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon_days: int = 60,
) -> dict:
    """Spearman rank IC: conviction score vs forward return.

    Computes the Information Coefficient across four signal variants to
    identify which formulation has the highest predictive power.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with columns: ``aggregated_conviction``,
        optionally ``conviction_score_adj``, ``signal_quality``.
    price_df : pd.DataFrame
        Daily adjusted close price DataFrame.
    horizon_days : int, optional
        Return horizon in trading days.

    Returns
    -------
    dict
        Keys for each IC variant:

        * ``ic_raw_all``         — raw urgency, all quality levels
        * ``ic_raw_high``        — raw urgency, HIGH quality only
        * ``ic_adj_all``         — demeaned urgency, all quality
        * ``ic_adj_high``        — demeaned urgency, HIGH quality only
        * ``ic_raw_all_pval``    — p-values for each
        * ... (pattern repeats)

    Notes
    -----
    Expected hierarchy (if strategy thesis holds):
        ic_adj_high > ic_adj_all > ic_raw_high > ic_raw_all

    If ic_raw_all > ic_adj_high, the manager universe likely contains
    structurally-late filers whose urgency is informationally redundant,
    and the demeaning adds noise rather than signal.
    """
    df_with_returns = _compute_forward_return(signal_df, price_df, horizon_days)
    df_valid = df_with_returns.dropna(subset=["forward_return"])

    results = {}

    def _ic(df_sub: pd.DataFrame, conv_col: str, label: str) -> tuple[float, float]:
        sub = df_sub.dropna(subset=[conv_col, "forward_return"])
        if len(sub) < 10:
            logger.warning("IC '%s': only %d valid observations", label, len(sub))
            return np.nan, np.nan
        r, pval = stats.spearmanr(sub[conv_col], sub["forward_return"])
        return r, pval

    # Raw conviction (aggregated_conviction)
    ic_r, pval_r = _ic(df_valid, "aggregated_conviction", "raw_all")
    results["ic_raw_all"] = ic_r
    results["ic_raw_all_pval"] = pval_r

    high_mask = df_valid.get("signal_quality", pd.Series(dtype=str)) == "HIGH"
    if high_mask.any():
        ic_rh, pval_rh = _ic(df_valid[high_mask], "aggregated_conviction", "raw_high")
        results["ic_raw_high"] = ic_rh
        results["ic_raw_high_pval"] = pval_rh
    else:
        results["ic_raw_high"] = np.nan
        results["ic_raw_high_pval"] = np.nan

    # Demeaned conviction (conviction_score_adj — if available)
    if "conviction_score_adj" in df_valid.columns:
        ic_a, pval_a = _ic(df_valid, "conviction_score_adj", "adj_all")
        results["ic_adj_all"] = ic_a
        results["ic_adj_all_pval"] = pval_a

        if high_mask.any():
            ic_ah, pval_ah = _ic(
                df_valid[high_mask], "conviction_score_adj", "adj_high"
            )
            results["ic_adj_high"] = ic_ah
            results["ic_adj_high_pval"] = pval_ah
        else:
            results["ic_adj_high"] = np.nan
            results["ic_adj_high_pval"] = np.nan
    else:
        results["ic_adj_all"] = np.nan
        results["ic_adj_all_pval"] = np.nan
        results["ic_adj_high"] = np.nan
        results["ic_adj_high_pval"] = np.nan

    # Print IC table
    print("\n" + "=" * 60)
    print(f"  Information Coefficient — {horizon_days}d Forward Return")
    print("=" * 60)
    print(f"  {'Signal Variant':<30} {'IC':>8} {'p-value':>10}")
    print("-" * 60)
    for label, ic_key, pval_key in [
        ("Raw urgency (all quality)",    "ic_raw_all",  "ic_raw_all_pval"),
        ("Raw urgency (HIGH only)",      "ic_raw_high", "ic_raw_high_pval"),
        ("Demeaned urgency (all)",       "ic_adj_all",  "ic_adj_all_pval"),
        ("Demeaned urgency (HIGH only)", "ic_adj_high", "ic_adj_high_pval"),
    ]:
        ic_val   = results.get(ic_key, np.nan)
        pval_val = results.get(pval_key, np.nan)
        ic_str   = f"{ic_val:.4f}" if not np.isnan(ic_val) else "  N/A  "
        pval_str = f"{pval_val:.4f}" if not np.isnan(pval_val) else "  N/A  "
        print(f"  {label:<30} {ic_str:>8} {pval_str:>10}")
    print("=" * 60 + "\n")

    return results


def run_activist_subgroup_analysis(
    signal_df: pd.DataFrame,
    manager_universe_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon_days: int = 60,
) -> pd.DataFrame:
    """Compare IC between activist and non-activist manager subgroups.

    Hypothesis: Activist managers exhibit the strongest filing_urgency signal
    because:
    1. Their positions are by definition high-conviction (activist campaigns
       require significant resource commitment).
    2. A single PM typically controls both the investment decision and the
       filing timing — the identification assumption is most valid here.
    3. Activist positions are less likely to be compliance-driven (activists
       do not run compliance-dominated portfolio construction processes).

    Parameters
    ----------
    signal_df : pd.DataFrame
        Full signal DataFrame (all managers).
    manager_universe_df : pd.DataFrame
        Universe reference with ``primary_strategy`` column.
    price_df : pd.DataFrame
        Daily price data.
    horizon_days : int
        Return horizon in trading days.

    Returns
    -------
    pd.DataFrame
        Subgroup comparison with columns:
        ``subgroup``, ``ic``, ``p_value``, ``n_obs``, ``horizon_days``.
    """
    # Identify activist CIKs
    activist_ciks = set(
        manager_universe_df[
            manager_universe_df["primary_strategy"] == "activist"
        ]["cik"].astype(str)
    )

    if "cik" in signal_df.columns:
        activist_mask = signal_df["cik"].astype(str).isin(activist_ciks)
    else:
        logger.warning(
            "No 'cik' column in signal_df; activist subgroup analysis requires CIK"
        )
        activist_mask = pd.Series(False, index=signal_df.index)

    activist_df     = signal_df[activist_mask].copy()
    nonactivist_df  = signal_df[~activist_mask].copy()

    results = []
    for label, subset in [("activist", activist_df), ("non_activist", nonactivist_df)]:
        if subset.empty:
            logger.warning("Empty subset for %s subgroup", label)
            results.append({"subgroup": label, "ic": np.nan, "p_value": np.nan,
                            "n_obs": 0, "horizon_days": horizon_days})
            continue

        df_wr = _compute_forward_return(subset, price_df, horizon_days)
        df_v  = df_wr.dropna(subset=["aggregated_conviction", "forward_return"])
        if len(df_v) < 5:
            results.append({"subgroup": label, "ic": np.nan, "p_value": np.nan,
                            "n_obs": len(df_v), "horizon_days": horizon_days})
            continue

        ic, pval = stats.spearmanr(df_v["aggregated_conviction"], df_v["forward_return"])
        results.append({"subgroup": label, "ic": ic, "p_value": pval,
                        "n_obs": len(df_v), "horizon_days": horizon_days})

    result_df = pd.DataFrame(results)

    # Print comparison
    print("\n" + "=" * 60)
    print(f"  Activist vs Non-Activist IC ({horizon_days}d horizon)")
    print("=" * 60)
    print(f"  {'Subgroup':<15} {'N':>6} {'IC':>8} {'p-value':>10}")
    print("-" * 60)
    for _, r in result_df.iterrows():
        ic_s = f"{r['ic']:.4f}" if not np.isnan(r["ic"]) else "  N/A  "
        pv_s = f"{r['p_value']:.4f}" if not np.isnan(r["p_value"]) else "  N/A  "
        print(f"  {r['subgroup']:<15} {int(r['n_obs']):>6} {ic_s:>8} {pv_s:>10}")
    print("=" * 60 + "\n")

    return result_df
