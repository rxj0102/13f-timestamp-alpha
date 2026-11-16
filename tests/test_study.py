"""Checks on the deadline arithmetic, the amendment summary and the committed results."""
import json
from pathlib import Path

import pandas as pd
import pytest

import filing_timing_study as fts

ROOT = Path(__file__).resolve().parents[1]
Q = pd.Timestamp


@pytest.mark.parametrize("report,filed,expected", [
    ("2023-03-31", "2023-05-15", 0),     # day 45 is a Monday: on the deadline
    ("2023-03-31", "2023-05-12", 1),     # the Friday before
    ("2023-03-31", "2023-05-08", 5),
    ("2023-03-31", "2023-05-16", -1),    # late
    ("2023-06-30", "2023-08-14", 0),     # day 45 is a Monday
    ("2022-09-30", "2022-11-14", 0),     # day 45 is a Monday
    ("2021-12-31", "2022-02-14", 0),     # day 45 is a Monday
    ("2020-06-30", "2020-08-14", 0),     # day 45 is a Friday
    ("2019-06-30", "2019-08-14", 0),     # day 45 is a Wednesday
    ("2023-09-30", "2023-11-14", 0),     # day 45 is a Tuesday
    ("2018-06-30", "2018-08-14", 0),
])
def test_business_days_before_deadline(report, filed, expected):
    assert fts.business_days_before_deadline(Q(report), Q(filed)) == expected


def test_weekend_deadline_moves_to_monday():
    # 2023-12-31 + 45 days = 2024-02-14, a Wednesday; 2022-12-31 + 45 = 2023-02-14, a Tuesday.
    # 2017-03-31 + 45 = 2017-05-15, a Monday. 2016-12-31 + 45 = 2017-02-14.
    # 2015-12-31 + 45 = 2016-02-14, a Sunday, and 2016-02-15 is Presidents Day: deadline Tuesday 2016-02-16.
    assert fts.business_days_before_deadline(Q("2015-12-31"), Q("2016-02-16")) == 0
    assert fts.business_days_before_deadline(Q("2015-12-31"), Q("2016-02-12")) == 1
    assert fts.business_days_before_deadline(Q("2015-12-31"), Q("2016-02-17")) == -1


def test_confidential_summary():
    ctr = pd.DataFrame({
        "cik": [1, 1, 1, 2, 3, 3],
        "manager_name": ["A", "A", "A", "B", "C", "C"],
        "accession": ["x", "x", "x", "y", "z", "w"],
        "quarter_label": ["Q1_2014", "Q1_2014", "Q1_2014", "Q2_2015", "Q3_2016", "Q4_2016"],
        "disclosure_lag_quarters": [1, 1, 1, 5, 2, 0],
        "conf_denied_expired": [False, False, False, True, True, False],
    })
    s = fts.confidential(ctr)
    assert s["n_rows"] == 6 and s["n_managers"] == 3 and s["n_amendments"] == 4
    assert s["top2_managers"] == ["A", "C"] and s["top2_share"] == pytest.approx(5 / 6)
    assert s["n_confirmed"] == 2 and s["confirmed_managers"] == 2 and s["confirmed_years"] == "2015 to 2016"
    assert s["share_lag_over_4"] == pytest.approx(1 / 6)
    assert s["n_large_amendments"] == 0 and s["median_rows_other_amendments"] == 1


def test_committed_history_agrees_with_the_summary():
    h = pd.read_csv(ROOT / "results" / "filing_history.csv", parse_dates=["report_date", "filing_date"])
    s = json.loads((ROOT / "results" / "filing_timing_summary.json").read_text())
    assert len(h) == s["n_filings"] and h["cik"].nunique() == s["n_managers"]
    assert ((h["filing_date"] - h["report_date"]).dt.days == h["days_after_quarter_end"]).all()
    b = [fts.business_days_before_deadline(r, f) for r, f in zip(h["report_date"], h["filing_date"])]
    assert (pd.Series(b) == h["bdays_before_deadline"]).all()


def test_amendment_log_has_no_constant_columns():
    ctr = pd.read_csv(ROOT / "data" / "reference" / "ctr_log.csv", comment="#")
    assert "ctr_granted" not in ctr.columns and "ctr_subsequently_disclosed" not in ctr.columns
    assert ctr["conf_denied_expired"].nunique() == 2


def test_headline_guards_fire_when_results_change():
    s = json.loads((ROOT / "results" / "filing_timing_summary.json").read_text())
    s["n_early_bucket"] = 40
    with pytest.raises(SystemExit):
        fts.headline(s)
