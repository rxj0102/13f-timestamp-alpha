"""Rebuild data/reference/ctr_log.csv from SEC EDGAR.

A position that a manager kept confidential is disclosed later in a 13F-HR/A whose amendment type is
NEW HOLDINGS. For every manager in data/reference/manager_seed.csv, this script collects all such amendments
for report periods 2014Q1 to 2023Q4, reads their information tables, and maps CUSIPs to tickers with OpenFIGI.

Columns kept compatible with src/features.py apply_ctr_filter (quarter_label, ticker_cusip), plus provenance:
accession, amendment filing date, conf_denied_expired flag from the cover page, issuer name and CUSIP.
A NEW HOLDINGS amendment can also correct an omission that was never confidential; conf_denied_expired
marks the filings that state they respond to a denied or expired confidential treatment request.

Usage: EDGAR_USER_AGENT="Name email" python scripts/build_ctr_log.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_manager_universe as b  # noqa: E402  (shared cached EDGAR getter)

QLAB = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}


def figi(cusips: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(cusips), 10):
        chunk = cusips[i:i + 10]
        req = urllib.request.Request("https://api.openfigi.com/v3/mapping", method="POST",
                                     headers={"Content-Type": "application/json"},
                                     data=json.dumps([{"idType": "ID_CUSIP", "idValue": c} for c in chunk]).encode())
        for attempt in range(5):
            try:
                res = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except Exception:
                time.sleep(15 * (attempt + 1))
        else:
            res = [{}] * len(chunk)
        for c, r in zip(chunk, res):
            data = r.get("data") or []
            us = [d for d in data if d.get("exchCode") == "US"] or data
            out[c] = us[0]["ticker"] if us else ""
        time.sleep(2.6)  # 25 requests per minute without an API key
    return out


def main() -> None:
    seed = pd.read_csv(ROOT / "data" / "reference" / "manager_seed.csv", dtype={"cik": str})
    rows = []
    for _, s in seed.iterrows():
        cik = s.cik.zfill(10)
        sub = b.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        blocks = [sub["filings"]["recent"]] + [b.get(f"https://data.sec.gov/submissions/{f['name']}") for f in sub["filings"].get("files", [])]
        df = pd.concat([pd.DataFrame({k: x[k] for k in ("form", "reportDate", "filingDate", "accessionNumber")}) for x in blocks])
        df = df[(df.form == "13F-HR/A") & df.reportDate.between(b.PERIOD_START, b.PERIOD_END)]
        for _, a in df.iterrows():
            nodash = a.accessionNumber.replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}"
            items = [i["name"] for i in b.get(f"{base}/index.json")["directory"]["item"] if i["name"].lower().endswith(".xml")]
            cover = next((b.get(f"{base}/{n}", as_json=False) for n in items if n == "primary_doc.xml"), "")
            atype = re.search(r"<(?:\w+:)?amendmentType>(.*?)<", cover)
            if not atype or "NEW HOLDINGS" not in atype.group(1).upper():
                continue
            denied = bool(re.search(r"<(?:\w+:)?confDeniedExpired>\s*true", cover, re.I))
            for n in items:
                if n == "primary_doc.xml":
                    continue
                x = b.get(f"{base}/{n}", as_json=False)
                for issuer, cusip in re.findall(r"<(?:\w+:)?nameOfIssuer>(.*?)</(?:\w+:)?nameOfIssuer>.*?<(?:\w+:)?cusip>(\w+)</", x, re.S):
                    rd = pd.Timestamp(a.reportDate)
                    fd = pd.Timestamp(a.filingDate)
                    rows.append({"cik": cik, "manager_name": s.manager_name,
                                 "quarter_label": f"{QLAB[rd.month]}_{rd.year}", "cusip": cusip.upper(),
                                 "issuer": issuer.strip(), "accession": a.accessionNumber,
                                 "amendment_filing_date": fd.date(), "conf_denied_expired": denied,
                                 "disclosure_lag_quarters": (fd.year - rd.year) * 4 + (fd.month - rd.month) // 3})
        print(f"{s.manager_name:38s} new-holdings amendments rows so far: {len(rows)}", flush=True)
    out = pd.DataFrame(rows).drop_duplicates(["cik", "quarter_label", "cusip"])
    tick = figi(sorted(out.cusip.unique())) if len(out) else {}
    out["ticker_cusip"] = [tick.get(c) or c for c in out.cusip]
    out["ctr_granted"] = True
    out["ctr_subsequently_disclosed"] = True
    cols = ["cik", "manager_name", "quarter_label", "ticker_cusip", "ctr_granted", "ctr_subsequently_disclosed",
            "disclosure_lag_quarters", "conf_denied_expired", "issuer", "cusip", "accession", "amendment_filing_date"]
    header = """# CTR Log, positions first disclosed in 13F-HR/A NEW HOLDINGS amendments
# Generated by scripts/build_ctr_log.py from SEC EDGAR. Do not edit by hand.
# One row per manager, report quarter and security. ticker_cusip is the OpenFIGI ticker, or the CUSIP if unmapped.
# conf_denied_expired is True when the cover page says the amendment responds to a denied or expired
# confidential treatment request; other NEW HOLDINGS amendments may be corrections of omissions.
"""
    (ROOT / "data" / "reference" / "ctr_log.csv").write_text(header + out[cols].sort_values(["quarter_label", "manager_name"]).to_csv(index=False))
    print(f"wrote {len(out)} rows; conf_denied_expired rows: {int(out.conf_denied_expired.sum())}; managers: {out.cik.nunique()}")


if __name__ == "__main__":
    main()
