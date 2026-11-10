"""Rebuild data/reference/manager_universe.csv from SEC EDGAR.

Inputs : data/reference/manager_seed.csv  (manager_name, cik, manager_type, primary_strategy)
Outputs: data/reference/manager_universe.csv
         results/filing_history.csv        (every original 13F-HR, report periods 2014Q1 to 2023Q4)

Every numeric column is measured from EDGAR:
  * cik and edgar_name come from data.sec.gov/submissions
  * filing-day statistics come from the filingDate and reportDate of each original 13F-HR
  * reported_13f_value_billion and reported_positions come from the cover page
    (tableValueTotal, tableEntryTotal) of the 13F-HR for the period ending 2022-12-31.
    This is the reportable long US equity book, not total AUM.
include_flag applies the filters in config.py to these measured values.

Usage: python scripts/build_manager_universe.py   (needs network access to sec.gov)
Set EDGAR_USER_AGENT to your name and email, as the SEC requires.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

UA = {"User-Agent": os.getenv("EDGAR_USER_AGENT", config.EDGAR_USER_AGENT)}
PERIOD_START, PERIOD_END = "2014-01-01", "2023-12-31"
SIZE_PERIOD = "2022-12-31"
CACHE = ROOT / "data" / "raw" / "edgar_cache"


def get(url: str, as_json: bool = True):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / re.sub(r"[^A-Za-z0-9]+", "_", url)[-180:]
    if key.exists():
        raw = key.read_text()
    else:
        for attempt in range(4):
            try:
                raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        time.sleep(config.REQUEST_DELAY_SECONDS)
        key.write_text(raw)
    return json.loads(raw) if as_json else raw


def filings_for(cik: str) -> tuple[str, pd.DataFrame]:
    sub = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        blocks.append(get(f"https://data.sec.gov/submissions/{f['name']}"))
    df = pd.concat([pd.DataFrame({k: b[k] for k in ("form", "filingDate", "reportDate", "accessionNumber")}) for b in blocks])
    df = df[df["form"] == "13F-HR"].copy()
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    return sub["name"], df


def cover_page(cik: str, accession: str) -> tuple[float, int]:
    nodash = accession.replace("-", "")
    idx = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/index.json")
    names = [i["name"] for i in idx["directory"]["item"]]
    doc = "primary_doc.xml" if "primary_doc.xml" in names else next(n for n in names if n.lower().endswith(".xml") and "info" not in n.lower())
    xml = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{doc}", as_json=False)
    value = float(re.search(r"<(?:\w+:)?tableValueTotal>([\d.]+)<", xml).group(1))
    entries = int(re.search(r"<(?:\w+:)?tableEntryTotal>(\d+)<", xml).group(1))
    return value, entries


def main() -> None:
    seed = pd.read_csv(ROOT / "data" / "reference" / "manager_seed.csv", dtype={"cik": str})
    hist_rows, out = [], []
    for _, s in seed.iterrows():
        cik = s.cik.zfill(10)
        name, df = filings_for(cik)
        win = df[(df.reportDate >= PERIOD_START) & (df.reportDate <= PERIOD_END)].sort_values("reportDate")
        win = win.drop_duplicates("reportDate", keep="first")  # first original filing per period
        win["days_after_quarter_end"] = (win.filingDate - win.reportDate).dt.days
        for _, w in win.iterrows():
            hist_rows.append({"cik": cik, "manager_name": s.manager_name, "report_date": w.reportDate.date(),
                              "filing_date": w.filingDate.date(), "days_after_quarter_end": w.days_after_quarter_end,
                              "accession": w.accessionNumber})
        value_bn, positions = float("nan"), pd.NA
        size_row = df[df.reportDate == SIZE_PERIOD].sort_values("filingDate").head(1)
        if len(size_row):
            v, positions = cover_page(cik, size_row.accessionNumber.iloc[0])
            # Filings after 2023-01-03 report tableValueTotal in dollars, earlier ones in thousands
            value_bn = v / 1e9 if size_row.filingDate.iloc[0] >= pd.Timestamp("2023-01-03") else v / 1e6
        d = win["days_after_quarter_end"]
        reasons = []
        if not (config.MIN_AUM_BILLION <= value_bn <= config.MAX_AUM_BILLION):
            reasons.append(f"13F value {value_bn:.1f}B outside {config.MIN_AUM_BILLION:g}-{config.MAX_AUM_BILLION:g}B" if value_bn == value_bn else "no 13F-HR for 2022Q4")
        if positions is not pd.NA and not (config.MIN_POSITIONS <= positions <= config.MAX_POSITIONS):
            reasons.append(f"{positions} positions outside {config.MIN_POSITIONS}-{config.MAX_POSITIONS}")
        if s.manager_type not in config.MANAGER_TYPE_FILTER:
            reasons.append(f"type {s.manager_type}")
        if len(win) < 8:
            reasons.append(f"only {len(win)} filings 2014-2023")
        out.append({
            "cik": cik, "manager_name": s.manager_name, "edgar_name": name,
            "manager_type": s.manager_type, "primary_strategy": s.primary_strategy,
            "reported_13f_value_billion": round(value_bn, 2) if value_bn == value_bn else "",
            "reported_positions": positions if positions is not pd.NA else "",
            "n_filings_2014_2023": len(win),
            "median_filing_day": int(d.median()) if len(d) else "",
            "share_filed_day_35_plus": round(float((d >= 35).mean()), 2) if len(d) else "",
            "share_filed_by_day_15": round(float((d <= 15).mean()), 2) if len(d) else "",
            "include_flag": not reasons,
            "exclusion_reason": "; ".join(reasons),
        })
        print(f"{s.manager_name:38s} {len(win):3d} filings  median day {out[-1]['median_filing_day']}  include={not reasons}")

    header = """# Manager Universe, 13F Timestamp Strategy
# Generated by scripts/build_manager_universe.py from SEC EDGAR. Do not edit by hand.
# manager_type and primary_strategy are the author's classification (data/reference/manager_seed.csv).
# Every other column is measured from EDGAR:
#   reported_13f_value_billion, reported_positions: 13F-HR cover page for the period ending 2022-12-31
#     (reportable long US equity book, not total AUM)
#   n_filings_2014_2023, median_filing_day, share_filed_*: original 13F-HR filings, report periods 2014Q1 to 2023Q4
# include_flag applies the config.py filters (MIN/MAX_AUM_BILLION on the 13F value, MIN/MAX_POSITIONS,
#   MANAGER_TYPE_FILTER) and requires at least 8 filings in the window.
"""
    path = ROOT / "data" / "reference" / "manager_universe.csv"
    path.write_text(header + pd.DataFrame(out).to_csv(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(hist_rows).to_csv(ROOT / "results" / "filing_history.csv", index=False)
    print(f"wrote {path} and results/filing_history.csv ({len(hist_rows)} filings)")


if __name__ == "__main__":
    main()
