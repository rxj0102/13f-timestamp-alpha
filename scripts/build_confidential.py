"""Collect hidden and shown holdings from SEC's Form 13F data sets.

    export EDGAR_USER_AGENT="Your Name your.address@example.com"
    python scripts/build_confidential.py

SEC publishes every Form 13F as quarterly tables from 2013 (https://www.sec.gov/data-research/
sec-markets-data/form-13f-data-sets). For each file the script keeps:

  hidden   holdings in 13F-HR/A filings whose cover page says the amendment adds new holdings
           and responds to a confidential treatment request that was denied or has expired
  shown    holdings in original 13F-HR filings whose summary page says confidential
           information was omitted

and writes them to data/raw/f13/holdings.csv.gz, with a filing index beside it. Files are
downloaded one at a time and deleted once read.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "f13"
INDEX_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
KEEP = ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]


def user_agent() -> str:
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if "@" not in ua:
        raise SystemExit('Set EDGAR_USER_AGENT="Your Name your.address@example.com" (SEC requires it).')
    return ua


def fetch(url: str, ua: str) -> bytes:
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=600).read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * 2 ** attempt)


def member(z: zipfile.ZipFile, name: str) -> str:
    """The archive member called name, wherever it sits; some files nest the tables in a folder."""
    hits = [n for n in z.namelist() if n.rsplit("/", 1)[-1].upper() == name.upper()]
    if len(hits) != 1:
        raise KeyError(f"{name}: found {hits}")
    return hits[0]


def table(z: zipfile.ZipFile, name: str, **kw) -> pd.DataFrame:
    with z.open(member(z, name)) as f:
        return pd.read_csv(f, sep="\t", dtype=str, quoting=3, on_bad_lines="skip", **kw)


def read_one(raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zipfile.ZipFile(io.BytesIO(raw))
    sub = table(z, "SUBMISSION.tsv")
    cov = table(z, "COVERPAGE.tsv", usecols=["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTTYPE", "CONFDENIEDEXPIRED",
                                              "DATEDENIEDEXPIRED", "FILINGMANAGER_NAME"])
    summ = table(z, "SUMMARYPAGE.tsv", usecols=["ACCESSION_NUMBER", "ISCONFIDENTIALOMITTED"])
    f = sub.merge(cov, on="ACCESSION_NUMBER", how="left").merge(summ, on="ACCESSION_NUMBER", how="left")
    hidden = (f.SUBMISSIONTYPE == "13F-HR/A") & (f.AMENDMENTTYPE == "NEW HOLDINGS") & (f.CONFDENIEDEXPIRED == "Y")
    shown = (f.SUBMISSIONTYPE == "13F-HR") & (f.ISCONFIDENTIALOMITTED == "Y")
    f = f[hidden | shown].copy()
    f["kind"] = ["hidden" if h else "shown" for h in hidden[hidden | shown]]
    keep = set(f.ACCESSION_NUMBER)
    parts = []
    with z.open(member(z, "INFOTABLE.tsv")) as fh:
        for chunk in pd.read_csv(fh, sep="\t", dtype=str, quoting=3, on_bad_lines="skip", usecols=KEEP, chunksize=500_000):
            parts.append(chunk[chunk.ACCESSION_NUMBER.isin(keep)])
    return f, pd.concat(parts, ignore_index=True)


def main() -> None:
    ua = user_agent()
    OUT.mkdir(parents=True, exist_ok=True)
    page = fetch(INDEX_PAGE, ua).decode("utf-8", "replace")
    urls = sorted({"https://www.sec.gov" + u for u in re.findall(r'href="(/files/[^"]*form13f[^"]*\.zip)"', page)})
    done = OUT / "done.txt"
    seen = set(done.read_text().split()) if done.exists() else set()
    for u in urls:
        name = u.rsplit("/", 1)[1]
        if name in seen:
            continue
        filings, holdings = read_one(fetch(u, ua))
        filings.to_csv(OUT / f"filings_{name[:-4]}.csv.gz", index=False)
        holdings.to_csv(OUT / f"holdings_{name[:-4]}.csv.gz", index=False)
        with done.open("a") as d:
            d.write(name + "\n")
        print(f"{name}: {int((filings.kind == 'hidden').sum())} hidden and {int((filings.kind == 'shown').sum())} shown filings, "
              f"{len(holdings)} holdings", flush=True)
    print("all files read")


if __name__ == "__main__":
    main()
