"""
backtest.py — Long/short backtesting engine for the 13F timestamp strategy.

Implements quarterly-rebalancing backtests that hold positions from the
filing date (when information becomes public) rather than from the quarter
end (which would be look-ahead bias — the information was not public yet).

CRITICAL design choice: trade entry on FILING DATE, not quarter-end date.
-----------------------------------------------------------------------
13F filings become public on EDGAR the moment they are accepted. The quarter
end is when the positions were held; the filing date is when an external
observer first knows about them. Any backtest that enters positions on the
quarter-end date is using information before it is available — look-ahead bias.

Stratified urgency backtests
----------------------------
The strategy's primary empirical validation is the three-way urgency split:

    EARLY filers (urgency < 0.33) → expected: weak or negative alpha
    MIDDLE filers (urgency 0.33–0.75) → expected: neutral
    LATE filers (urgency > 0.75) → expected: strongest alpha

If this monotonic pattern holds, it confirms that filing timing is
informative of conviction, not merely correlated with some other factor.

CTR contamination test
----------------------
Running the backtest both with and without CTR-affected positions is a
look-ahead bias audit. CTR-affected positions disclose information that
was hidden at signal-construction time. If including them materially
improves backtest performance, it suggests the model is inadvertently
exploiting post-hoc information — a critical red flag.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import lag_dataframe

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_position_return(
    ticker: str,
    entry_date: pd.Timestamp,
    hold_days: int,
    price_df: pd.DataFrame,
    side: str,
) -> float:
    """Compute the holding-period return for a single ticker position.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol.
    entry_date : pd.Timestamp
        Date to enter the position (typically the filing date).
    hold_days : int
        Number of trading days to hold.
    price_df : pd.DataFrame
        Wide-format daily adjusted close prices (tickers as columns).
    side : str
        "LONG" or "SHORT".

    Returns
    -------
    float
        Net return (positive = profitable). NaN if price data unavailable.
    """
    if ticker not in price_df.columns:
        return np.nan

    price_series = price_df[ticker].dropna()
    future = price_series.loc[price_series.index >= entry_date]

    if len(future) < hold_days + 1:
        return np.nan

    entry = future.iloc[0]
    exit_ = future.iloc[hold_days]

    if entry <= 0 or np.isnan(entry):
        return np.nan

    raw_return = (exit_ - entry) / entry
    return raw_return if side == "LONG" else -raw_return


def _apply_transaction_costs(gross_return: float, tc_bps: int = config.TRANSACTION_COST_BPS) -> float:
    """Deduct round-trip transaction costs from a gross return.

    Applies one-way cost on entry and one-way cost on exit.
    Total round-trip cost = 2 × tc_bps × 0.0001.
    """
    round_trip = 2 * tc_bps * 1e-4
    return gross_return - round_trip


def _compute_drawdown(cumulative_pnl: pd.Series) -> pd.Series:
    """Compute the rolling maximum drawdown from a cumulative PnL series."""
    rolling_max = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - rolling_max) / (1 + rolling_max).clip(lower=1e-10)
    return drawdown


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_long_short_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    hold_days: Optional[int] = None,
    tc_bps: Optional[int] = None,
    label: str = "",
) -> pd.DataFrame:
    """Run the quarterly-rebalancing long/short backtest.

    Rebalances once per quarter when new 13F filings drop. Entry is on the
    filing date (information first available to the market), not the quarter
    end. Position holding period = ``SIGNAL_HOLDING_DAYS`` (default 60 days).

    Parameters
    ----------
    signal_df : pd.DataFrame
        Positioned signal (output of ``signal.build_quarterly_signal``
        concatenated across multiple quarters).
        Required columns: ``quarter_label``, ``ticker``, ``position_side``,
        and a filing date column (``filing_date_only`` or ``quarter_end_date``).
    price_df : pd.DataFrame
        Wide-format daily adjusted close prices, DatetimeIndex.
    hold_days : int, optional
        Override default holding period from ``config.SIGNAL_HOLDING_DAYS``.
    tc_bps : int, optional
        Override default transaction cost from ``config.TRANSACTION_COST_BPS``.
    label : str, optional
        Label for logging and table printing.

    Returns
    -------
    pd.DataFrame
        Backtest results with columns:

        * ``quarter_label``      — quarter identifier
        * ``long_return``        — equal-weight mean return, long book
        * ``short_return``       — equal-weight mean return, short book
        * ``ls_return``          — gross long/short return
        * ``tc_drag``            — transaction cost drag
        * ``net_ls_return``      — LS return after costs
        * ``cumulative_pnl``     — compounded net return
        * ``drawdown``           — rolling drawdown from peak
        * ``n_long``             — number of long positions
        * ``n_short``            — number of short positions

    Notes
    -----
    Each position is equal-weighted within the long and short books.
    The long book and short book are each assumed to be notional $1,
    so the combined LS portfolio is $2 notional (1 long + 1 short),
    and the reported LS return is on the $1 deployed basis (i.e., the
    return is not levered by the short proceeds).

    The maximum drawdown peak and trough dates are logged separately via
    ``compute_performance_metrics``.
    """
    effective_hold  = hold_days if hold_days is not None else config.SIGNAL_HOLDING_DAYS
    effective_tc    = tc_bps    if tc_bps    is not None else config.TRANSACTION_COST_BPS

    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)

    # Determine filing date column
    date_col = None
    for c in ["filing_date_only", "filing_datetime", "quarter_end_date"]:
        if c in signal_df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("No filing date column found in signal_df")

    quarters = sorted(signal_df["quarter_label"].unique())
    rows = []

    for q in quarters:
        q_signal = signal_df[signal_df["quarter_label"] == q]
        longs    = q_signal[q_signal["position_side"] == "LONG"]
        shorts   = q_signal[q_signal["position_side"] == "SHORT"]

        if longs.empty and shorts.empty:
            logger.debug("Quarter %s: no long or short positions", q)
            continue

        # Determine entry date: median filing date within the quarter
        q_dates = pd.to_datetime(signal_df.loc[signal_df["quarter_label"] == q, date_col])
        entry_date = q_dates.median()
        if pd.isna(entry_date):
            logger.warning("Could not determine entry date for quarter %s; skipping", q)
            continue
        entry_date = pd.Timestamp(entry_date).normalize()

        # Long book
        long_returns = []
        for _, row in longs.iterrows():
            ret = _compute_position_return(
                row["ticker"], entry_date, effective_hold, price_df, "LONG"
            )
            long_returns.append(ret)
        long_rets = [r for r in long_returns if not np.isnan(r)]
        mean_long = np.mean(long_rets) if long_rets else 0.0

        # Short book
        short_returns = []
        for _, row in shorts.iterrows():
            ret = _compute_position_return(
                row["ticker"], entry_date, effective_hold, price_df, "SHORT"
            )
            short_returns.append(ret)
        short_rets = [r for r in short_returns if not np.isnan(r)]
        mean_short = np.mean(short_rets) if short_rets else 0.0

        gross_ls = mean_long + mean_short
        n_trades = (len(longs) + len(shorts)) * 2   # entry + exit per position
        tc_drag  = n_trades * effective_tc * 1e-4
        net_ls   = gross_ls - tc_drag

        rows.append(
            {
                "quarter_label":  q,
                "entry_date":     entry_date,
                "long_return":    mean_long,
                "short_return":   mean_short,
                "ls_return":      gross_ls,
                "tc_drag":        tc_drag,
                "net_ls_return":  net_ls,
                "n_long":         len(longs),
                "n_short":        len(shorts),
            }
        )

    if not rows:
        logger.warning("Backtest (%s): no valid quarterly observations", label)
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("quarter_label").reset_index(drop=True)

    # Cumulative PnL (compounded)
    result["cumulative_pnl"] = (1 + result["net_ls_return"]).cumprod() - 1

    # Drawdown
    result["drawdown"] = _compute_drawdown(result["cumulative_pnl"] + 1) * 100

    logger.info(
        "Backtest (%s): %d quarters | mean net LS = %.2f%% | "
        "cumulative = %.2f%% | max drawdown = %.2f%%",
        label or "default",
        len(result),
        result["net_ls_return"].mean() * 100,
        result["cumulative_pnl"].iloc[-1] * 100 if not result.empty else 0,
        result["drawdown"].min(),
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stratified backtests
# ─────────────────────────────────────────────────────────────────────────────

def run_urgency_stratified_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run three separate backtests stratified by urgency bucket.

    This is the strategy's primary empirical validation. The expected result
    is a monotonic increase in alpha from EARLY to LATE filers:

        alpha(EARLY) < alpha(MIDDLE) < alpha(LATE)

    If this ordering holds, it confirms the filing timing mechanism — managers
    who wait until the last moment before disclosing their positions are still
    holding those positions at disclosure time, and those positions subsequently
    outperform.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Full signal DataFrame with ``urgency_bucket`` column
        (populated by ``features.compute_filing_urgency``).
    price_df : pd.DataFrame
        Daily adjusted close prices.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "EARLY", "MIDDLE", "LATE".
        Values: backtest_df from ``run_long_short_backtest`` for each bucket.

    Notes
    -----
    The EARLY bucket backtest serves as a negative control: if EARLY filers
    (who had maximum opportunity to exit before disclosure) show neutral or
    negative alpha, this validates that the urgency signal is not simply
    capturing a general momentum effect.
    """
    results = {}
    for bucket in ["EARLY", "MIDDLE", "LATE"]:
        if "urgency_bucket" not in signal_df.columns:
            logger.warning(
                "urgency_bucket column not in signal_df; "
                "run features.compute_filing_urgency first"
            )
            break

        subset = signal_df[signal_df["urgency_bucket"] == bucket].copy()
        if subset.empty:
            logger.warning("No signal data for urgency bucket %s", bucket)
            results[bucket] = pd.DataFrame()
            continue

        logger.info(
            "Running stratified backtest for %s bucket (%d ticker-quarters)",
            bucket, len(subset),
        )
        bt = run_long_short_backtest(subset, price_df, label=bucket)
        results[bucket] = bt

    return results


def run_demeaned_vs_raw_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compare backtests using raw vs manager-demeaned conviction scores.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Must contain both ``aggregated_conviction`` (raw) and
        ``conviction_score_adj`` (demeaned, may be NaN for managers with
        insufficient history). Also needs ``position_side`` for both variants.
    price_df : pd.DataFrame
        Daily adjusted close prices.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "raw", "demeaned".
        Values: backtest_df for each conviction variant.

    Notes
    -----
    If the demeaned version does not show a higher Sharpe ratio, one likely
    explanation is that the manager universe is too homogeneous — if all
    included managers are naturally late filers, there is little variation
    in urgency deviation to exploit. In this case, the raw urgency signal
    may actually be more informative because it captures the absolute level
    of filing timing discipline.
    """
    results = {}

    # Raw urgency backtest (use existing position_side based on aggregated_conviction)
    logger.info("Running raw urgency backtest")
    results["raw"] = run_long_short_backtest(signal_df, price_df, label="raw_urgency")

    # Demeaned urgency backtest (rebuild position_side using conviction_score_adj)
    if "conviction_score_adj" in signal_df.columns:
        adj_df = signal_df.copy()
        # Re-assign position_side based on demeaned score quartile ranks
        adj_df = adj_df.dropna(subset=["conviction_score_adj"])
        adj_df["_adj_rank"] = adj_df.groupby("quarter_label")["conviction_score_adj"].rank(
            pct=True
        )
        def _adj_side(row: pd.Series) -> str:
            if row.get("ctr_affected", False):
                return "NEUTRAL"
            if row.get("signal_quality", "") == "THIN":
                return "NEUTRAL"
            if row["_adj_rank"] >= 0.80:
                return "LONG"
            if row["_adj_rank"] <= 0.20:
                return "SHORT"
            return "NEUTRAL"

        adj_df["position_side"] = adj_df.apply(_adj_side, axis=1)
        logger.info("Running demeaned urgency backtest")
        results["demeaned"] = run_long_short_backtest(adj_df, price_df, label="demeaned_urgency")
    else:
        logger.warning("conviction_score_adj not found; skipping demeaned backtest")
        results["demeaned"] = pd.DataFrame()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(
    backtest_df: pd.DataFrame,
    label: str = "",
    periods_per_year: float = 4.0,
) -> dict:
    """Compute annualised performance metrics for a backtest result.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``run_long_short_backtest``.
    label : str, optional
        Label for display.
    periods_per_year : float, optional
        Rebalancing frequency (default 4.0 for quarterly).

    Returns
    -------
    dict
        Keys:

        * ``annualised_return``   — CAGR
        * ``annualised_vol``      — annualised volatility of period returns
        * ``sharpe``              — Sharpe ratio (assumes 0 risk-free rate)
        * ``max_drawdown``        — maximum drawdown (as negative fraction)
        * ``calmar``              — CAGR / |Max Drawdown|
        * ``sortino``             — Sortino ratio
        * ``win_rate``            — fraction of quarters with positive return
        * ``avg_quarterly_ls``    — mean quarterly LS return
        * ``tc_drag_annualised``  — annualised transaction cost drag
    """
    if backtest_df.empty:
        logger.warning("compute_performance_metrics: empty backtest_df for '%s'", label)
        return {}

    returns = backtest_df["net_ls_return"].dropna()
    if len(returns) < 2:
        logger.warning("Insufficient observations for metrics (%s)", label)
        return {}

    n = len(returns)
    total_return = (1 + returns).prod() - 1
    ann_return   = (1 + total_return) ** (periods_per_year / n) - 1
    ann_vol      = returns.std() * np.sqrt(periods_per_year)
    sharpe       = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd_series   = (cum - rolling_max) / rolling_max
    max_dd      = dd_series.min()
    calmar      = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    # Sortino
    downside     = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino      = ann_return / downside_std if (downside_std is not None and downside_std > 0) else np.nan

    win_rate = (returns > 0).mean()
    avg_ls   = returns.mean()
    tc_drag  = backtest_df.get("tc_drag", pd.Series(0)).mean() * periods_per_year

    metrics = {
        "label":               label,
        "n_quarters":          n,
        "annualised_return":   ann_return,
        "annualised_vol":      ann_vol,
        "sharpe":              sharpe,
        "max_drawdown":        max_dd,
        "calmar":              calmar,
        "sortino":             sortino,
        "win_rate":            win_rate,
        "avg_quarterly_ls":    avg_ls,
        "tc_drag_annualised":  tc_drag,
    }

    # Print formatted table
    print("\n" + "=" * 60)
    print(f"  Performance Metrics — {label or 'Strategy'}")
    print("=" * 60)
    print(f"  Quarters:              {n}")
    print(f"  Annualised Return:     {ann_return:>10.2%}")
    print(f"  Annualised Vol:        {ann_vol:>10.2%}")
    print(f"  Sharpe Ratio:          {sharpe:>10.3f}" if not np.isnan(sharpe) else "  Sharpe Ratio:               N/A")
    print(f"  Max Drawdown:          {max_dd:>10.2%}")
    print(f"  Calmar Ratio:          {calmar:>10.3f}" if not np.isnan(calmar) else "  Calmar Ratio:               N/A")
    print(f"  Sortino Ratio:         {sortino:>10.3f}" if not np.isnan(sortino) else "  Sortino Ratio:              N/A")
    print(f"  Win Rate:              {win_rate:>10.1%}")
    print(f"  Avg Quarterly LS Ret:  {avg_ls:>10.2%}")
    print(f"  TC Drag (annualised):  {tc_drag:>10.2%}")
    print("=" * 60 + "\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CTR contamination test
# ─────────────────────────────────────────────────────────────────────────────

def run_ctr_contamination_test(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    ctr_log_df: pd.DataFrame,
) -> dict:
    """Bias audit: compare backtest performance with vs without CTR positions.

    **This test is a look-ahead bias detector, not a strategy variant.**

    CTR-affected positions were hidden at signal-construction time. Their
    eventual disclosure contains information that was not available to a
    real-money implementation of this strategy. If including CTR positions
    materially improves the backtest, it indicates that the strategy is
    inadvertently exploiting post-hoc disclosure — the returns are inflated
    by hidden information.

    Expected result (if strategy is correctly implemented):
        performance_with_CTR ≈ performance_without_CTR

    Red flag (look-ahead bias present):
        performance_with_CTR >> performance_without_CTR

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``ctr_affected`` column (populated by
        ``features.apply_ctr_filter``).
    price_df : pd.DataFrame
        Daily adjusted close prices.
    ctr_log_df : pd.DataFrame
        CTR log for auditing.

    Returns
    -------
    dict
        Keys: ``with_ctr``, ``without_ctr``, each mapping to a metrics dict
        from ``compute_performance_metrics``.
        Also: ``contamination_detected`` (bool) and ``contamination_magnitude``
        (difference in Sharpe ratios).
    """
    logger.info("Running CTR contamination audit")

    # Without CTR: exclude CTR-affected positions
    no_ctr_signal = signal_df.copy()
    if "ctr_affected" in no_ctr_signal.columns:
        no_ctr_signal = no_ctr_signal[~no_ctr_signal["ctr_affected"]]
    else:
        logger.warning("ctr_affected column not found; results may be identical")

    # With CTR: include everything (including CTR-affected as if they were clean)
    with_ctr_signal = signal_df.copy()
    if "ctr_affected" in with_ctr_signal.columns:
        with_ctr_signal = with_ctr_signal.copy()
        # Override the position_side for CTR positions to LONG/SHORT
        # (treating them as valid signal positions)
        with_ctr_signal.loc[
            with_ctr_signal["ctr_affected"] & (with_ctr_signal["aggregated_conviction"] > 0),
            "position_side",
        ] = "LONG"
        with_ctr_signal.loc[
            with_ctr_signal["ctr_affected"] & (with_ctr_signal["aggregated_conviction"] < 0),
            "position_side",
        ] = "SHORT"

    bt_without = run_long_short_backtest(no_ctr_signal,  price_df, label="without_CTR")
    bt_with    = run_long_short_backtest(with_ctr_signal, price_df, label="with_CTR")

    metrics_without = compute_performance_metrics(bt_without, label="Without CTR positions")
    metrics_with    = compute_performance_metrics(bt_with,    label="With CTR positions (bias audit)")

    sharpe_diff = (
        (metrics_with.get("sharpe",   0) or 0)
        - (metrics_without.get("sharpe", 0) or 0)
    )
    contaminated = sharpe_diff > 0.1   # 0.1 Sharpe threshold for material impact

    if contaminated:
        logger.warning(
            "CTR CONTAMINATION DETECTED: Sharpe improves by %.3f when CTR "
            "positions are included. This suggests look-ahead bias — CTR "
            "positions contain information not available at signal-construction "
            "time and must be excluded from any live implementation.",
            sharpe_diff,
        )
    else:
        logger.info(
            "CTR contamination test: Sharpe difference = %.3f (below 0.10 threshold). "
            "No material look-ahead bias detected from CTR inclusion.",
            sharpe_diff,
        )

    return {
        "without_ctr":              metrics_without,
        "with_ctr":                 metrics_with,
        "contamination_detected":   contaminated,
        "contamination_magnitude":  sharpe_diff,
        "bt_without_ctr":           bt_without,
        "bt_with_ctr":              bt_with,
    }
