"""Freeze the inputs of the hidden-positions study into data/snapshot/.

    python scripts/build_confidential.py   # needs EDGAR_USER_AGENT
    python scripts/build_prices.py         # needs EDGAR_USER_AGENT
    python scripts/prepare_hidden.py

Writes
  data/snapshot/positions.csv.gz   one row per manager, report period, security and side
                                   (hidden or shown), with the reported value
  data/snapshot/amendments.csv.gz  one row per hidden-holdings amendment, with its filing date
  data/snapshot/prices.csv.gz      quarter-end prices of those securities from all 13F filers
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "f13"
SNAP = ROOT / "data" / "snapshot"


def main() -> None:
    f = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(RAW.glob("filings_*.csv.gz"))], ignore_index=True)
    h = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(RAW.glob("holdings_*.csv.gz"))], ignore_index=True)
    f["period"] = pd.to_datetime(f.PERIODOFREPORT, format="%d-%b-%Y")
    f["filed"] = pd.to_datetime(f.FILING_DATE, format="%d-%b-%Y")
    f = f.drop_duplicates("ACCESSION_NUMBER")
    hidden, shown = f[f.kind == "hidden"], f[f.kind == "shown"]
    # an original must exist for the same manager and period; keep the latest original if refiled
    shown = shown.sort_values("filed").drop_duplicates(["CIK", "period"], keep="last")
    hidden = hidden[hidden.set_index(["CIK", "period"]).index.isin(shown.set_index(["CIK", "period"]).index)]
    h = h[(h.SSHPRNAMTTYPE == "SH") & h.PUTCALL.isna()].assign(CUSIP=lambda d: d.CUSIP.str.upper())
    h["value"] = pd.to_numeric(h.VALUE, errors="coerce")
    keys = pd.concat([hidden.assign(side="hidden"), shown.assign(side="shown")])[
        ["ACCESSION_NUMBER", "CIK", "FILINGMANAGER_NAME", "period", "filed", "side"]]
    pos = h.merge(keys, on="ACCESSION_NUMBER")
    pos = pos[pos.set_index(["CIK", "period"]).index.isin(hidden.set_index(["CIK", "period"]).index)]
    # value units differ by filing date; keep each side's own units, used only for weights within a side
    pos = pos.groupby(["CIK", "period", "side", "CUSIP"], as_index=False).agg(
        manager=("FILINGMANAGER_NAME", "first"), value=("value", "sum"), issuer=("NAMEOFISSUER", "first"))
    amend = hidden[["ACCESSION_NUMBER", "CIK", "FILINGMANAGER_NAME", "period", "filed", "DATEDENIEDEXPIRED"]]
    prices = pd.read_csv(RAW / "prices.csv.gz", dtype={"CUSIP": str}, parse_dates=["period"])
    prices = prices[prices.CUSIP.isin(set(pos.CUSIP))]
    SNAP.mkdir(parents=True, exist_ok=True)
    gz = {"method": "gzip", "mtime": 0}
    pos.sort_values(["period", "CIK", "side", "CUSIP"]).to_csv(SNAP / "positions.csv.gz", index=False, compression=gz)
    amend.sort_values(["period", "CIK", "filed"]).to_csv(SNAP / "amendments.csv.gz", index=False, compression=gz)
    prices.assign(price=prices.price.round(6)).to_csv(SNAP / "prices.csv.gz", index=False, compression=gz)
    print(f"{len(pos)} positions, {len(amend)} amendments, {len(prices)} prices")


if __name__ == "__main__":
    main()
