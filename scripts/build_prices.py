"""Quarter-end prices for every security in the sample, from 13F filings themselves.

    export EDGAR_USER_AGENT="Your Name your.address@example.com"
    python scripts/build_prices.py

Every 13F holding reports a market value and a share count at quarter end, so value divided by
shares is that quarter's price. The median across all filers holding the security gives a price
that exists even for companies later acquired or delisted, which free price sources drop.

Two kinds of filing error are fixed before the median is taken:
  units   VALUE is in thousands of dollars before 2023-01-03 and in dollars after, and some
          filers get it wrong. A filing whose prices sit about 1,000 times above or below the
          other filers' (median log10 ratio beyond 2.5, over at least 5 holdings) is rescaled.
  splits  for each security, filers that hold it in two consecutive quarters are compared. Their
          median ratio of shares held is the split ratio when there is one, because most holders
          do not trade in a given quarter. It is stored as holder_ratio.

Output data/raw/f13/prices.csv.gz: CUSIP, period, price, filers, shares_held, holder_ratio,
holders_both.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_confidential import INDEX_PAGE, OUT, fetch, member, table, user_agent  # noqa: E402

DOLLARS_FROM = pd.Timestamp("2023-01-03")


def wanted_cusips() -> set[str]:
    cusips = set()
    for f in OUT.glob("holdings_*.csv.gz"):
        cusips |= set(pd.read_csv(f, dtype=str, usecols=["CUSIP"]).CUSIP.dropna().str.upper())
    return cusips


def rows_from(raw: bytes, cusips: set[str]) -> pd.DataFrame:
    """Holdings of the wanted securities in original 13F-HR filings, with naive prices."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    sub = table(z, "SUBMISSION.tsv", usecols=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    sub = sub[sub.SUBMISSIONTYPE == "13F-HR"]
    sub["unit"] = np.where(pd.to_datetime(sub.FILING_DATE, format="%d-%b-%Y") >= DOLLARS_FROM, 1.0, 1000.0)
    meta = sub.set_index("ACCESSION_NUMBER")[["CIK", "PERIODOFREPORT", "unit"]]
    parts = []
    with z.open(member(z, "INFOTABLE.tsv")) as fh:
        for c in pd.read_csv(fh, sep="\t", dtype=str, quoting=3, on_bad_lines="skip", chunksize=1_000_000,
                             usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]):
            c = c[(c.SSHPRNAMTTYPE == "SH") & c.PUTCALL.isna() & c.ACCESSION_NUMBER.isin(meta.index)]
            c = c.assign(CUSIP=c.CUSIP.str.upper())
            c = c[c.CUSIP.isin(cusips)]
            v, s = pd.to_numeric(c.VALUE, errors="coerce"), pd.to_numeric(c.SSHPRNAMT, errors="coerce")
            m = meta.loc[c.ACCESSION_NUMBER]
            c = c.assign(cik=m.CIK.to_numpy(), period=m.PERIODOFREPORT.to_numpy(), shares=s,
                         price=v * m.unit.to_numpy() / s)
            parts.append(c[(c.shares > 0) & (c.price > 0)][["ACCESSION_NUMBER", "cik", "CUSIP", "period", "price", "shares"]])
    return pd.concat(parts, ignore_index=True)


def fix_units(d: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rescale filings whose prices sit about 1,000 times away from the other filers'."""
    cons = d.groupby("CUSIP").price.transform("median")
    off = np.log10(d.price / cons)
    stat = off.groupby(d.ACCESSION_NUMBER).agg(["median", "size"])
    bad = stat[(stat["size"] >= 5) & (stat["median"].abs() > 2.5)]
    scale = (10.0 ** (-3 * np.sign(bad["median"]))).reindex(d.ACCESSION_NUMBER).fillna(1.0).to_numpy()
    return d.assign(price=d.price * scale), int(len(bad))


def main() -> None:
    ua = user_agent()
    cusips = wanted_cusips()
    page = fetch(INDEX_PAGE, ua).decode("utf-8", "replace")
    urls = sorted({"https://www.sec.gov" + u for u in re.findall(r'href="(/files/[^"]*form13f[^"]*\.zip)"', page)})
    by_period = OUT / "by_period"
    by_period.mkdir(exist_ok=True)
    done = by_period / "done.txt"
    seen = set(done.read_text().split()) if done.exists() else set()
    for u in urls:
        name = u.rsplit("/", 1)[1]
        if name in seen:
            continue
        rows = rows_from(fetch(u, ua), cusips)
        for period, g in rows.groupby("period"):
            target = by_period / (pd.Timestamp(period).strftime("%Y-%m-%d") + ".csv.gz")
            g.to_csv(target, mode="a", header=not target.exists(), index=False, compression="gzip")
        with done.open("a") as f:
            f.write(name + "\n")
        print(f"{name}: {len(rows)} holdings", flush=True)
    aggregate(by_period)


def aggregate(by_period: Path) -> None:
    out, prev, fixed = [], None, 0
    quarter_ends = set(pd.date_range("2013-03-31", "2026-12-31", freq="QE").strftime("%Y-%m-%d"))
    files = sorted(f for f in by_period.glob("*.csv.gz") if f.name[:10] in quarter_ends)
    for f in files:
        d = pd.read_csv(f, dtype={"CUSIP": str, "cik": str})
        d = d.drop_duplicates(["ACCESSION_NUMBER", "CUSIP", "shares", "price"])
        d, n_bad = fix_units(d)
        fixed += n_bad
        g = d.groupby("CUSIP")
        res = pd.DataFrame({"price": g.price.median(), "filers": g.cik.nunique(), "shares_held": g.shares.sum()})
        holder = d.groupby(["CUSIP", "cik"]).shares.sum()
        if prev is not None:
            both = holder.to_frame("now").join(prev.to_frame("before"), how="inner")
            ratio = (both.now / both.before).groupby(level="CUSIP")
            res["holder_ratio"] = ratio.median()
            res["holders_both"] = ratio.size()
        prev = holder
        out.append(res.reset_index().assign(period=f.name[:10]))
    out = pd.concat(out, ignore_index=True)
    out["period"] = pd.to_datetime(out.period)
    out = out[["CUSIP", "period", "price", "filers", "shares_held", "holder_ratio", "holders_both"]]
    out.sort_values(["CUSIP", "period"]).to_csv(OUT / "prices.csv.gz", index=False)
    print(f"{len(out)} CUSIP-quarters for {out.CUSIP.nunique()} CUSIPs; {fixed} filings rescaled for units")


if __name__ == "__main__":
    main()
