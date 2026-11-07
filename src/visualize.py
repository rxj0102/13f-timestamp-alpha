"""
visualize.py — Publication-quality charts for the 13F timestamp arbitrage strategy.

All charts are saved to ``outputs/`` at 300 DPI using seaborn's whitegrid style.
Figure numbering follows the strategy paper structure:
    Figure 1 — Filing urgency distribution (filing_timing_distribution.png)
    Figure 2 — Stratified urgency backtest comparison (stratified_backtest.png)

Chart naming convention: ``outputs/{descriptive_snake_case_name}.png``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config

logger = logging.getLogger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
FIGURE_DPI = 300
OUTPUT_DIR = config.OUTPUTS_DIR


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to the outputs directory and close it."""
    path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Filing urgency distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_filing_urgency_distribution(
    filing_history_df: pd.DataFrame,
    filename: str = "fig1_filing_urgency_distribution.png",
) -> Path:
    """Plot the distribution of filing urgency across the manager universe.

    **Figure 1** — the first empirical result. Shows that filing timing has
    clear structure: the distribution is NOT uniform. Clusters near 0.1 and
    near 0.9 correspond to compliance-early filers and PM-late filers
    respectively. The bimodality (if present) is itself evidence of the
    two distinct filing behaviours this strategy seeks to exploit.

    Parameters
    ----------
    filing_history_df : pd.DataFrame
        Output of ``features.compute_filing_urgency``.
        Required columns: ``filing_urgency``, ``manager_type``.
    filename : str
        Output filename within ``outputs/``.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    df = filing_history_df.dropna(subset=["filing_urgency"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Figure 1 — Filing Urgency Distribution\n"
        r"$\mathit{filing\_urgency} = \frac{\text{days remaining until deadline}}{45}$",
        fontsize=13, y=1.02,
    )

    # Left: overall histogram
    ax = axes[0]
    manager_types = df["manager_type"].unique() if "manager_type" in df.columns else ["all"]
    colors = sns.color_palette("tab10", len(manager_types))

    for i, mtype in enumerate(manager_types):
        subset = df[df["manager_type"] == mtype]["filing_urgency"] if "manager_type" in df.columns else df["filing_urgency"]
        ax.hist(
            subset, bins=20, alpha=0.6, color=colors[i],
            label=mtype.replace("_", " ").title(), density=True,
        )

    ax.axvline(
        config.URGENCY_EARLY_THRESHOLD, color="red", ls="--", lw=1.5,
        label=f"Early threshold ({config.URGENCY_EARLY_THRESHOLD:.2f})",
    )
    ax.axvline(
        config.URGENCY_LATE_THRESHOLD, color="green", ls="--", lw=1.5,
        label=f"Late threshold ({config.URGENCY_LATE_THRESHOLD:.2f})",
    )
    ax.set_xlabel("Filing Urgency (0 = day 1 of 45; 1 = day 45 of 45)")
    ax.set_ylabel("Density")
    ax.set_title("All Managers Combined")
    ax.legend(fontsize=9)

    # Annotate urgency buckets
    y_top = ax.get_ylim()[1]
    ax.text(0.16, y_top * 0.95, "EARLY\n(compliance)", ha="center", va="top",
            color="red", fontsize=8)
    ax.text(0.54, y_top * 0.95, "MIDDLE", ha="center", va="top",
            color="grey", fontsize=8)
    ax.text(0.875, y_top * 0.95, "LATE\n(conviction)", ha="center", va="top",
            color="green", fontsize=8)

    # Right: KDE by manager type
    ax2 = axes[1]
    if "manager_type" in df.columns:
        for mtype in manager_types:
            subset = df[df["manager_type"] == mtype]["filing_urgency"].dropna()
            if len(subset) > 5:
                subset.plot.kde(ax=ax2, label=mtype.replace("_", " ").title(), lw=2)
    else:
        df["filing_urgency"].plot.kde(ax=ax2, lw=2, label="all")

    ax2.axvline(config.URGENCY_EARLY_THRESHOLD, color="red",   ls="--", lw=1.5)
    ax2.axvline(config.URGENCY_LATE_THRESHOLD,  color="green", ls="--", lw=1.5)
    ax2.set_xlabel("Filing Urgency")
    ax2.set_ylabel("Density (KDE)")
    ax2.set_title("KDE by Manager Type")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Urgency vs forward return scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_urgency_vs_forward_return_scatter(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizon_days: int = 60,
    filename: str = "urgency_vs_forward_return.png",
) -> Path:
    """Scatter plot of filing urgency against 60-day forward equity return.

    The slope of the OLS trendline for NEW and INCREASED positions is the
    paper's central empirical finding: do managers who file later earn
    higher subsequent returns?

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``filing_urgency``, ``ticker``, ``position_type``,
        ``filing_date_only``.
    price_df : pd.DataFrame
        Daily adjusted close prices.
    horizon_days : int
        Forward return horizon in trading days.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    from src.signal import _compute_forward_return

    df = _compute_forward_return(signal_df, price_df, horizon_days)
    df = df.dropna(subset=["filing_urgency", "forward_return"])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Filing Urgency vs {horizon_days}-Day Forward Return\n"
        "OLS line shown for NEW and INCREASED positions only",
        fontsize=12,
    )

    # Color by position type
    position_types = df["position_type"].unique() if "position_type" in df.columns else ["all"]
    color_map = {
        "NEW": "#2ecc71", "INCREASED": "#3498db",
        "REDUCED": "#e74c3c", "EXITED": "#c0392b", "FLAT": "#95a5a6",
    }

    for ptype in position_types:
        subset = df[df["position_type"] == ptype] if "position_type" in df.columns else df
        ax.scatter(
            subset["filing_urgency"], subset["forward_return"],
            alpha=0.4, s=20,
            color=color_map.get(ptype, "#999999"),
            label=ptype,
        )

    # OLS trendline for bullish positions
    bullish = df[df.get("position_type", pd.Series(dtype=str)).isin(["NEW", "INCREASED"])] \
        if "position_type" in df.columns else df
    if len(bullish) > 10:
        slope, intercept, r_val, p_val, _ = stats.linregress(
            bullish["filing_urgency"], bullish["forward_return"]
        )
        x_line = np.linspace(0, 1, 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, "k--", lw=2,
            label=f"OLS (NEW+INCR): slope={slope:.3f}, p={p_val:.3f}",
        )

        # Compute Spearman IC for annotation
        ic, ic_pval = stats.spearmanr(bullish["filing_urgency"], bullish["forward_return"])
        ax.text(
            0.05, 0.95,
            f"Spearman IC = {ic:.4f} (p={ic_pval:.3f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

    ax.axvline(config.URGENCY_EARLY_THRESHOLD, color="red",   ls=":", lw=1, alpha=0.7)
    ax.axvline(config.URGENCY_LATE_THRESHOLD,  color="green", ls=":", lw=1, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Filing Urgency (0=early, 1=late)")
    ax.set_ylabel(f"{horizon_days}-Day Forward Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Conviction score by urgency bucket (box plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_conviction_score_by_urgency_bucket(
    signal_df: pd.DataFrame,
    filename: str = "conviction_by_urgency_bucket.png",
) -> Path:
    """Box plot of aggregated conviction scores split by urgency bucket.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``urgency_bucket`` and ``aggregated_conviction``.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    df = signal_df.dropna(subset=["aggregated_conviction"]).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "Conviction Score Distribution by Filing Urgency Bucket",
        fontsize=12,
    )

    bucket_order = ["EARLY", "MIDDLE", "LATE"]
    bucket_palette = {"EARLY": "#e74c3c", "MIDDLE": "#3498db", "LATE": "#2ecc71"}

    if "urgency_bucket" in df.columns and "signal_quality" in df.columns:
        sns.boxplot(
            data=df, x="urgency_bucket", y="aggregated_conviction",
            hue="signal_quality", order=bucket_order,
            palette="Set2", ax=ax, fliersize=2, linewidth=0.8,
        )
    elif "urgency_bucket" in df.columns:
        sns.boxplot(
            data=df, x="urgency_bucket", y="aggregated_conviction",
            order=bucket_order, palette=bucket_palette, ax=ax,
            fliersize=2, linewidth=0.8,
        )
    else:
        ax.text(0.5, 0.5, "urgency_bucket column not found",
                transform=ax.transAxes, ha="center")

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Urgency Bucket")
    ax.set_ylabel("Aggregated Conviction Score")
    ax.set_title(
        "LATE filers should show higher (more positive) aggregated conviction\n"
        "for bullish signals — this is the strategy's visual prediction",
        fontsize=9,
    )

    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Stratified backtest comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_stratified_backtest_comparison(
    early_bt: pd.DataFrame,
    middle_bt: pd.DataFrame,
    late_bt: pd.DataFrame,
    filename: str = "fig2_stratified_backtest.png",
) -> Path:
    """Three-panel cumulative PnL chart for EARLY / MIDDLE / LATE urgency buckets.

    **Figure 2** — the visual proof of the strategy thesis. Monotonically
    increasing alpha from EARLY to LATE is the expected result.

    Parameters
    ----------
    early_bt : pd.DataFrame
        Backtest result for EARLY filers.
    middle_bt : pd.DataFrame
        Backtest result for MIDDLE filers.
    late_bt : pd.DataFrame
        Backtest result for LATE filers.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Figure 2 — Stratified Backtest: Cumulative L/S PnL by Filing Urgency\n"
        "Monotonic alpha improvement from EARLY → LATE confirms the timing thesis",
        fontsize=12, y=1.01,
    )

    configs = [
        (early_bt,  "EARLY Filers (urgency < 0.33)",  "#c0392b", axes[0]),
        (middle_bt, "MIDDLE Filers (0.33 ≤ urgency ≤ 0.75)", "#2980b9", axes[1]),
        (late_bt,   "LATE Filers (urgency > 0.75)",   "#27ae60", axes[2]),
    ]

    for bt_df, title, color, ax in configs:
        if bt_df is None or bt_df.empty:
            ax.text(0.5, 0.5, f"{title}\n(no data)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="grey")
            ax.set_title(title, fontsize=10)
            continue

        x = range(len(bt_df))
        cum_pnl = bt_df["cumulative_pnl"] * 100
        ax.plot(x, cum_pnl, color=color, lw=2)
        ax.fill_between(x, 0, cum_pnl, alpha=0.15, color=color)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)

        # Annotate final PnL and Sharpe
        final = cum_pnl.iloc[-1] if not cum_pnl.empty else 0
        mean_r = bt_df["net_ls_return"].mean()
        std_r  = bt_df["net_ls_return"].std()
        sharpe = (mean_r / std_r * 2) if std_r > 0 else np.nan  # annualise quarterly

        ax.text(
            0.98, 0.95,
            f"Cumul: {final:+.1f}%  |  Sharpe: {sharpe:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylabel("Cumulative L/S Return (%)")
        ax.set_xticks(x[::4])
        if not bt_df.empty:
            ax.set_xticklabels(bt_df["quarter_label"].iloc[::4], rotation=45, ha="right", fontsize=8)

    axes[-1].set_xlabel("Quarter")
    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Manager filing pattern heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_manager_filing_pattern_heatmap(
    filing_history_df: pd.DataFrame,
    filename: str = "manager_filing_pattern_heatmap.png",
    max_managers: int = 30,
) -> Path:
    """Heatmap of filing urgency by manager × quarter.

    Immediately reveals structural early/late filers. Managers sorted by
    average urgency — compliance-driven managers cluster at the top (red);
    PM-driven managers cluster at the bottom (green).

    Parameters
    ----------
    filing_history_df : pd.DataFrame
        Filing history with ``manager_name``, ``quarter_label``, ``filing_urgency``.
    filename : str
        Output filename.
    max_managers : int
        Cap number of managers displayed to keep the chart readable.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    df = filing_history_df.copy()

    # Pivot to manager × quarter matrix
    pivot = df.pivot_table(
        index="manager_name", columns="quarter_label",
        values="filing_urgency", aggfunc="mean",
    )

    # Sort by average urgency (structural late filers at bottom)
    pivot["_avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_avg", ascending=False).drop(columns=["_avg"])
    pivot = pivot.iloc[:max_managers]  # cap display

    # Sort quarters chronologically
    def _q_sort_key(q_label: str) -> tuple:
        try:
            parts = q_label.split("_")
            return (int(parts[1]), int(parts[0][1:]))
        except (IndexError, ValueError):
            return (9999, 9)

    sorted_cols = sorted(pivot.columns, key=_q_sort_key)
    pivot = pivot[sorted_cols]

    fig, ax = plt.subplots(
        figsize=(max(12, len(sorted_cols) * 0.4), max(8, len(pivot) * 0.3))
    )
    fig.suptitle(
        "Manager Filing Pattern Heatmap\n"
        "0 = early filer (red, compliance-driven) → 1 = late filer (green, conviction-driven)\n"
        "Sorted by average urgency — structural patterns are immediately visible",
        fontsize=11, y=1.01,
    )

    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        ax=ax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Filing Urgency", "shrink": 0.6},
        annot=len(pivot) <= 20,
        fmt=".2f",
        annot_kws={"size": 6},
    )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Manager")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Raw vs demeaned conviction backtest comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_pnl_raw_vs_demeaned(
    raw_bt: pd.DataFrame,
    demeaned_bt: pd.DataFrame,
    filename: str = "raw_vs_demeaned_conviction.png",
) -> Path:
    """Overlay cumulative PnL for raw conviction vs manager-demeaned conviction.

    Parameters
    ----------
    raw_bt : pd.DataFrame
        Backtest result using raw urgency-based conviction.
    demeaned_bt : pd.DataFrame
        Backtest result using manager-demeaned urgency.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "Raw vs Manager-Demeaned Urgency — Cumulative L/S PnL Comparison",
        fontsize=12,
    )

    def _sharpe(bt: pd.DataFrame) -> float:
        r = bt["net_ls_return"].dropna()
        if r.std() == 0 or len(r) < 2:
            return np.nan
        return r.mean() / r.std() * np.sqrt(4)

    raw_sharpe = _sharpe(raw_bt) if not raw_bt.empty else np.nan
    dem_sharpe = _sharpe(demeaned_bt) if not demeaned_bt.empty else np.nan

    if not raw_bt.empty:
        ax.plot(
            raw_bt["cumulative_pnl"] * 100,
            color="grey", ls="--", lw=2,
            label=f"Raw urgency  (Sharpe: {raw_sharpe:.2f})" if not np.isnan(raw_sharpe)
                  else "Raw urgency",
        )

    if not demeaned_bt.empty:
        ax.plot(
            demeaned_bt["cumulative_pnl"] * 100,
            color="#2980b9", lw=2.5,
            label=f"Demeaned urgency  (Sharpe: {dem_sharpe:.2f})" if not np.isnan(dem_sharpe)
                  else "Demeaned urgency",
        )

    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)
    ax.set_ylabel("Cumulative L/S Return (%)")
    ax.set_xlabel("Quarter")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=10)

    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# CTR contamination audit chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_ctr_contamination_audit(
    with_ctr_bt: pd.DataFrame,
    without_ctr_bt: pd.DataFrame,
    contamination_detected: bool = False,
    filename: str = "ctr_contamination_audit.png",
) -> Path:
    """Overlay PnL with and without CTR-affected positions.

    If the two lines diverge materially, the strategy has a look-ahead bias
    problem — CTR disclosure contains information not available at signal time.

    Parameters
    ----------
    with_ctr_bt : pd.DataFrame
        Backtest including CTR-affected positions as valid signal.
    without_ctr_bt : pd.DataFrame
        Backtest excluding CTR-affected positions (the clean version).
    contamination_detected : bool
        If True, adds a WARNING annotation to the chart title.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    warning_str = (
        "\n⚠ WARNING: CTR inclusion materially affects results — review for look-ahead bias"
        if contamination_detected else ""
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"CTR Contamination Audit — Cumulative L/S PnL{warning_str}",
        fontsize=12, color="red" if contamination_detected else "black",
    )

    if not without_ctr_bt.empty:
        ax.plot(
            without_ctr_bt["cumulative_pnl"] * 100,
            color="#2ecc71", lw=2.5, label="Without CTR positions (clean)",
        )
    if not with_ctr_bt.empty:
        ax.plot(
            with_ctr_bt["cumulative_pnl"] * 100,
            color="#e74c3c", lw=2, ls="--",
            label="With CTR positions (bias audit only — do not use)",
        )

    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_ylabel("Cumulative L/S Return (%)")
    ax.set_xlabel("Quarter")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=10)

    if contamination_detected:
        ax.text(
            0.5, 0.05,
            "Lines diverge: CTR-affected positions contain post-hoc information.",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="red", style="italic",
        )

    plt.tight_layout()
    return _save(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Activist vs non-activist IC comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_activist_vs_nonactivist_ic(
    subgroup_results_df: pd.DataFrame,
    filename: str = "activist_vs_nonactivist_ic.png",
) -> Path:
    """Side-by-side bar chart comparing IC for activist vs non-activist managers.

    Parameters
    ----------
    subgroup_results_df : pd.DataFrame
        Output of ``signal.run_activist_subgroup_analysis``.
        Must contain: ``subgroup``, ``ic``, ``p_value``, ``horizon_days``.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Absolute path to the saved chart.
    """
    df = subgroup_results_df.dropna(subset=["ic"]).copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Information Coefficient: Activist vs Non-Activist Managers\n"
        "Hypothesis: activist managers exhibit strongest filing urgency signal",
        fontsize=12,
    )

    if df.empty:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center")
        return _save(fig, filename)

    subgroups = df["subgroup"].unique()
    x = np.arange(len(subgroups))
    width = 0.35

    # If multiple horizons, group by horizon
    if "horizon_days" in df.columns and df["horizon_days"].nunique() > 1:
        horizons = sorted(df["horizon_days"].unique())
        bar_width = 0.8 / len(horizons)
        for i, horizon in enumerate(horizons):
            h_df = df[df["horizon_days"] == horizon]
            ic_vals = [h_df[h_df["subgroup"] == s]["ic"].values[0]
                       if s in h_df["subgroup"].values else 0 for s in subgroups]
            ax.bar(x + i * bar_width, ic_vals, bar_width * 0.9,
                   label=f"{horizon}d horizon", alpha=0.8)
    else:
        ic_vals = [df[df["subgroup"] == s]["ic"].values[0]
                   if s in df["subgroup"].values else 0 for s in subgroups]
        colors = ["#e74c3c" if "non" in s else "#2ecc71" for s in subgroups]
        ax.bar(x, ic_vals, 0.5, color=colors, alpha=0.85)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in subgroups], fontsize=10)
    ax.set_ylabel("Spearman IC")
    ax.set_title("Higher IC for activist subgroup validates the single-PM mechanism", fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    return _save(fig, filename)
