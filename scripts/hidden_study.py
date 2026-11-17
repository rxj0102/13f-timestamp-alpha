"""Run the hidden-positions study on data/snapshot/ and write results/hidden_summary.json.

    python scripts/hidden_study.py

Also writes results/hidden_pairs.csv.gz (every manager-quarter and horizon) and
results/figures/hidden.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import hidden as H  # noqa: E402

RES = ROOT / "results"


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def summarize(pairs: pd.DataFrame) -> dict:
    out = {}
    for (anchor, k), g in pairs.groupby(["anchor", "k"]):
        out[f"{anchor}|{k}"] = {
            "hidden_ew": H.t_test_by_period(g, "hidden_ew"), "shown_ew": H.t_test_by_period(g, "shown_ew"),
            "diff_ew": H.t_test_by_period(g, "diff_ew"), "diff_vw": H.t_test_by_period(g, "diff_vw"),
            "median_hidden_positions": float(g.n_hidden.median()), "median_shown_positions": float(g.n_shown.median())}
    return out


def figure(pairs: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    labels, means, halves, colors = [], [], [], []
    for anchor, name, color in (("report", "from the report date", "#2a78d6"), ("reveal", "from the reveal", "#eb6834")):
        for k in (1, 2, 4):
            g = pairs[(pairs.anchor == anchor) & (pairs.k == k)]
            r = H.t_test_by_period(g, "diff_ew")
            labels.append(f"{k}Q {name}")
            means.append(r["mean"] * 100)
            halves.append(abs(r["mean"] / r["t"]) * 1.96 * 100 if r["t"] else 0)
            colors.append(color)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(y, means, color=colors, height=0.6)
    ax.errorbar(means, y, xerr=halves, fmt="none", ecolor=ink, elinewidth=1, capsize=3)
    ax.axvline(0, color=ink, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hidden minus shown, equal weighted, percent", color=muted)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Return of hidden positions over the same manager's shown positions, 95% interval",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hidden.png", dpi=150)
    plt.close(fig)


def main() -> None:
    pos, amend, prices = H.load()
    suspect = H.suspect_panel(prices)
    pairs = H.pair_returns(pos, amend, H.price_panel(prices, adjust=True))
    excluded = H.pair_returns(pos, amend, H.price_panel(prices, adjust=False), suspect)
    priced = prices[prices.filers >= H.MIN_FILERS]
    hid_pos = pos[pos.side == "hidden"]
    top = pos.groupby("manager").period.nunique().sort_values(ascending=False)
    lag = (amend.groupby(["CIK", "period"]).filed.min().reset_index().pipe(lambda d: (d.filed - d.period).dt.days))
    payload = rounded({
        "sample": {"manager_quarters": int(pos.groupby(["CIK", "period"]).ngroups), "managers": int(pos.CIK.nunique()),
                   "amendments": int(len(amend)), "first_period": str(pos.period.min().date()),
                   "last_period": str(pos.period.max().date()),
                   "hidden_positions": int(len(hid_pos)), "shown_positions": int((pos.side == "shown").sum()),
                   "securities": int(pos.CUSIP.nunique()),
                   "share_hidden_priced": float(hid_pos.CUSIP.isin(set(priced.CUSIP)).mean()),
                   "median_days_to_reveal": float(lag.median()), "top_managers": list(top.index[:6]),
                   "top6_share_of_quarters": float(top.iloc[:6].sum() / top.sum()), "min_filers": H.MIN_FILERS},
        "results": summarize(pairs), "split_excluded": summarize(excluded),
        "suspect_quarters_share": float(suspect.to_numpy().mean()),
        "primary": f"{H.PRIMARY[0]}|{H.PRIMARY[1]}",
    })
    RES.mkdir(exist_ok=True)
    (RES / "hidden_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    pairs.round(8).to_csv(RES / "hidden_pairs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    figure(pairs, RES / "figures")
    print(json.dumps(payload["sample"], indent=1))
    for key, v in payload["results"].items():
        d, w, u = v["diff_ew"], v["diff_vw"], payload["split_excluded"][key]["diff_ew"]
        print(f"{key:9s} hidden {v['hidden_ew']['mean']:+.3f} shown {v['shown_ew']['mean']:+.3f} diff {d['mean']:+.4f} t {d['t']:+.2f} "
              f"q {d['quarters']} pairs {d['pairs']} med {d['median_pair']:+.4f} pos {d['share_positive']:.2f} | vw {w['mean']:+.4f} t {w['t']:+.2f} | excluded {u['mean']:+.4f} t {u['t']:+.2f}")


if __name__ == "__main__":
    main()
