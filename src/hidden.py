"""Do the positions hedge funds hide from their 13F earn more than the ones they show?

Design, fixed before any return was computed:

  hidden     holdings first disclosed in a 13F-HR/A that adds new holdings and responds to a
             confidential treatment request that was denied or has expired
  shown      holdings in the same manager's original 13F-HR for the same quarter, which states
             that confidential information was omitted. A security on both lists is dropped
             from shown.
  unit       one manager and report quarter; several amendments for it are pooled
  prices     quarter-end price of each security: the median of value / shares across every
             13F filer holding it, used only when MIN_FILERS or more filers report it. A
             security that stops appearing exits at its last price, so acquired and delisted
             companies stay in.
  splits     filers that hold a security in two consecutive quarters mostly do not trade it, so
             their median ratio of shares held equals the split ratio when there is a split.
             Where it is within HOLDER_TOL of a common ratio, earlier prices are divided by it.
             As a check, prices are left unadjusted and a security is instead dropped from any
             window containing a suspected split: shares per filer jumping by MIN_JUMP or more
             while the price moves the other way.
  spikes     an isolated quarter-end price more than SPIKE times away from both neighbours, in
             the same direction, is treated as missing
  returns    price returns, equal weighted within a side; value weighted as a check
  primary    hidden minus shown over the 2 quarters after the report date, about the median
             time a position stays hidden
  secondary  1 and 4 quarters after the report date; and 1, 2 and 4 quarters from the first
             quarter end on or after the amendment is filed, which is what an outsider could trade
  inference  the differences are averaged across managers within each report quarter, then
             tested across quarters
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "snapshot"
MIN_FILERS = 3
SPLITS = tuple(sorted({float(n) for n in range(2, 51)} | {1.5, 2.5}))
HOLDER_TOL = 0.03         # continuing holders' median share ratio within 3% of a split ratio
MIN_HOLDERS = 5           # at least this many filers must hold the security in both quarters
MIN_JUMP = 1.45           # check only: shares per filer jumping this much against the price
SPIKE = 5.0               # a price more than 5x away from both neighbours, the same way, is dropped
PRIMARY = ("report", 2)
HORIZONS = [("report", 1), ("report", 2), ("report", 4), ("reveal", 1), ("reveal", 2), ("reveal", 4)]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pos = pd.read_csv(SNAP / "positions.csv.gz", dtype={"CIK": str, "CUSIP": str}, parse_dates=["period"])
    amend = pd.read_csv(SNAP / "amendments.csv.gz", dtype={"CIK": str}, parse_dates=["period", "filed"])
    prices = pd.read_csv(SNAP / "prices.csv.gz", dtype={"CUSIP": str}, parse_dates=["period"])
    return pos, amend, prices


def quarter_ends(start: str = "2013-03-31", end: str = "2026-12-31") -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="QE")


def split_ratio(holder_ratio: float, holders_both: float) -> float:
    """The split ratio implied by continuing holders, or 1.0 when there is none."""
    if not (holders_both >= MIN_HOLDERS) or not (holder_ratio > 0):
        return 1.0
    up = max(holder_ratio, 1 / holder_ratio)
    k = min(SPLITS, key=lambda s: abs(np.log(up / s)))
    if abs(up / k - 1) > HOLDER_TOL:
        return 1.0
    return k if holder_ratio > 1 else 1 / k


def split_factors(p: pd.DataFrame) -> pd.Series:
    """Factor to multiply each CUSIP-quarter price by so that prices are comparable over time.

    p holds one CUSIP's rows sorted by period, with holder_ratio and holders_both. A split of
    ratio k in quarter i divides all earlier prices by k.
    """
    r = [split_ratio(a, b) for a, b in zip(p["holder_ratio"], p["holders_both"])]
    f = np.ones(len(p))
    for i in range(1, len(p)):
        if r[i] != 1.0:
            f[:i] /= r[i]
    return pd.Series(f, index=p.index)


def drop_spikes(s: pd.Series) -> pd.Series:
    """Set to NaN a price more than SPIKE times above, or below, both of its neighbours."""
    prev, nxt = s.shift(1), s.shift(-1)
    up = (s > SPIKE * prev) & (s > SPIKE * nxt)
    down = (s < prev / SPIKE) & (s < nxt / SPIKE)
    return s.mask(up | down)


def suspect_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """True for a CUSIP-quarter that may contain a split (see the module docstring)."""
    q = quarter_ends()
    p = prices[(prices["filers"] >= MIN_FILERS) & prices["period"].isin(q)].sort_values(["CUSIP", "period"]).copy()
    spf = p["shares_held"] / p["filers"]
    r_spf = spf / spf.groupby(p["CUSIP"]).shift()
    r_p = p["price"] / p["price"].groupby(p["CUSIP"]).shift()
    p["suspect"] = ((r_spf >= MIN_JUMP) & (r_p < 1)) | ((r_spf <= 1 / MIN_JUMP) & (r_p > 1))
    return p.pivot(index="period", columns="CUSIP", values="suspect").reindex(q).fillna(False).astype(bool)


def price_panel(prices: pd.DataFrame, adjust: bool = True) -> pd.DataFrame:
    """Wide table of prices: quarter ends by CUSIP, split adjusted, NaN where too few filers."""
    q = quarter_ends()
    p = prices[(prices["filers"] >= MIN_FILERS) & prices["period"].isin(q)].sort_values(["CUSIP", "period"]).copy()
    if adjust:
        p["factor"] = p.groupby("CUSIP", group_keys=False)[["holder_ratio", "holders_both"]].apply(split_factors)
        p["price"] = p["price"] * p["factor"]
    wide = p.pivot(index="period", columns="CUSIP", values="price").reindex(q)
    return wide.apply(lambda s: drop_spikes(s.dropna()).reindex(s.index))


def returns_between(panel: pd.DataFrame, cusips: list[str], start: pd.Timestamp, k: int,
                    suspect: pd.DataFrame | None = None) -> pd.Series:
    """Price return from quarter end start to k quarters later; a security that stops appearing
    exits at its last price before the end. Securities without a start price are dropped, and so
    are securities with a suspected split inside the window when suspect is given."""
    idx = panel.index
    if start not in idx:
        return pd.Series(dtype=float)
    i = idx.get_loc(start)
    if i + k >= len(idx):
        return pd.Series(dtype=float)
    cols = [c for c in cusips if c in panel.columns]
    window = panel.iloc[i: i + k + 1][cols]
    p0 = window.iloc[0]
    window = window.loc[:, p0.notna()]
    if suspect is not None:
        cols_s = [c for c in window.columns if c in suspect.columns]
        bad = suspect.iloc[i + 1: i + k + 1][cols_s].any()
        window = window.drop(columns=list(bad[bad].index))
    if window.empty:
        return pd.Series(dtype=float)
    end = window.ffill().iloc[-1]
    if window.index[-1] > panel.dropna(how="all").index[-1]:
        return pd.Series(dtype=float)
    return end / window.iloc[0] - 1.0


def first_quarter_end_on_or_after(d: pd.Timestamp) -> pd.Timestamp:
    return (d + pd.offsets.QuarterEnd(0)).normalize()


def pair_returns(pos: pd.DataFrame, amend: pd.DataFrame, panel: pd.DataFrame,
                 suspect: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per manager-quarter and horizon: hidden and shown returns, equal and value weighted."""
    reveal = amend.groupby(["CIK", "period"])["filed"].min()
    rows = []
    for (cik, period), g in pos.groupby(["CIK", "period"]):
        hid = g[g.side == "hidden"].set_index("CUSIP")
        sho = g[(g.side == "shown") & ~g.CUSIP.isin(hid.index)].set_index("CUSIP")
        if hid.empty or sho.empty:
            continue
        starts = {"report": period, "reveal": first_quarter_end_on_or_after(reveal[(cik, period)])}
        for anchor, k in HORIZONS:
            rh = returns_between(panel, list(hid.index), starts[anchor], k, suspect)
            rs = returns_between(panel, list(sho.index), starts[anchor], k, suspect)
            if rh.empty or rs.empty:
                continue
            wh, ws = hid.loc[rh.index, "value"].clip(lower=0), sho.loc[rs.index, "value"].clip(lower=0)
            rows.append({"CIK": cik, "manager": g.manager.iloc[0], "period": period, "anchor": anchor, "k": k,
                         "start": starts[anchor], "n_hidden": len(rh), "n_shown": len(rs),
                         "hidden_ew": rh.mean(), "shown_ew": rs.mean(),
                         "hidden_vw": float((rh * wh).sum() / wh.sum()) if wh.sum() > 0 else np.nan,
                         "shown_vw": float((rs * ws).sum() / ws.sum()) if ws.sum() > 0 else np.nan})
    out = pd.DataFrame(rows)
    out["diff_ew"] = out.hidden_ew - out.shown_ew
    out["diff_vw"] = out.hidden_vw - out.shown_vw
    return out


def t_test_by_period(df: pd.DataFrame, col: str, lags: int = 0) -> dict:
    """Average across managers within each report quarter, then a t-test across quarters.

    Windows longer than one quarter overlap from one report quarter to the next (a four
    quarter window starting in Q1 shares three quarters with the one starting in Q2), so
    the quarterly series is autocorrelated. Pass lags = k - 1 for a k quarter window: the
    standard error is then Newey-West with that many lags (Bartlett weights). lags = 0 is
    the plain t-test.
    """
    by = df.dropna(subset=[col]).groupby("period")[col].mean().sort_index()
    n = len(by)
    m = float(by.mean())
    e = by.to_numpy() - m
    s2 = float(e @ e) / n
    for j in range(1, min(lags, n - 1) + 1):
        s2 += 2 * (1 - j / (lags + 1)) * float(e[j:] @ e[:-j]) / n
    if lags == 0:
        s2 = float(e @ e) / (n - 1) if n > 1 else float("nan")
    t = m / sqrt(s2 / n) if s2 > 0 else float("nan")
    x = df[col].dropna()
    return {"mean": m, "t": t, "quarters": n, "pairs": int(len(x)), "median_pair": float(x.median()),
            "share_positive": float((x > 0).mean())}
