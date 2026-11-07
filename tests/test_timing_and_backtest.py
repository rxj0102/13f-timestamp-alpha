"""Tests for filing-urgency direction, quarter/business-day utilities,
conviction scoring and the long/short backtest mechanics."""

import numpy as np
import pandas as pd
import pytest

import config
from src.backtest import (
    _compute_drawdown,
    _compute_position_return,
    run_long_short_backtest,
)
from src.features import compute_conviction_score, compute_filing_urgency
from src.utils import (
    days_to_deadline,
    get_quarter_end,
    get_quarter_label,
    lag_dataframe,
    next_business_day,
)


# ---------------------------------------------------------------- utilities
def test_quarter_label_round_trip():
    assert get_quarter_label(pd.Timestamp("2023-03-31")) == "Q1_2023"
    assert get_quarter_label(pd.Timestamp("2022-10-01")) == "Q4_2022"
    assert get_quarter_end("Q1_2023") == pd.Timestamp("2023-03-31")
    assert get_quarter_end("Q4_2022") == pd.Timestamp("2022-12-31")


def test_quarter_end_rejects_bad_label():
    with pytest.raises(ValueError):
        get_quarter_end("2023Q1")


def test_next_business_day_rolls_weekend_and_holiday():
    assert next_business_day(pd.Timestamp("2023-11-09")) == pd.Timestamp("2023-11-09")
    assert next_business_day(pd.Timestamp("2023-11-10")) == pd.Timestamp("2023-11-13")  # observed Veterans Day
    assert next_business_day(pd.Timestamp("2023-11-11")) == pd.Timestamp("2023-11-13")
    assert next_business_day(pd.Timestamp("2023-09-02")) == pd.Timestamp("2023-09-05")  # Sat + Labor Day


def test_days_to_deadline_sign():
    d = pd.Timestamp("2023-05-15")
    assert days_to_deadline(pd.Timestamp("2023-05-10"), d) == 5
    assert days_to_deadline(pd.Timestamp("2023-05-20"), d) == -5


def test_lag_dataframe_shifts_numeric_only():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})
    out = lag_dataframe(df, 1)
    assert np.isnan(out["x"].iloc[0]) and out["x"].iloc[1] == 1.0
    assert list(out["name"]) == ["a", "b", "c"]


# ------------------------------------------------------------ filing urgency
def _history(days_after_qend):
    qend = pd.Timestamp("2023-03-31")
    deadline = qend + pd.Timedelta(days=config.FILING_DEADLINE_DAYS)
    hist = pd.DataFrame(
        {
            "quarter_label": ["Q1_2023"] * len(days_after_qend),
            "filing_date_only": [qend + pd.Timedelta(days=d) for d in days_after_qend],
        }
    )
    deadlines = pd.DataFrame({"quarter_label": ["Q1_2023"], "filing_deadline": [deadline]})
    return hist, deadlines


def test_urgency_increases_with_later_filing():
    """Filing later in the 45-day window must give HIGHER urgency (README spec)."""
    hist, deadlines = _history([5, 20, 40])
    out = compute_filing_urgency(hist, deadlines)
    u = out["filing_urgency"].to_numpy()
    assert u[0] < u[1] < u[2]
    np.testing.assert_allclose(u, [5 / 45, 20 / 45, 40 / 45])


def test_urgency_buckets_follow_thresholds():
    hist, deadlines = _history([3, 25, 44])
    out = compute_filing_urgency(hist, deadlines)
    assert list(out["urgency_bucket"]) == ["EARLY", "MIDDLE", "LATE"]


def test_urgency_clamped_to_unit_interval():
    hist, deadlines = _history([-3, 60])  # before quarter end / after deadline
    out = compute_filing_urgency(hist, deadlines)
    assert out["filing_urgency"].between(0.0, 1.0).all()
    assert out["filing_urgency"].iloc[1] == 1.0


# --------------------------------------------------------------- conviction
def test_conviction_sign_and_flat_zero():
    df = pd.DataFrame(
        {
            "filing_urgency": [0.8, 0.8, 0.8, 0.8, 0.8],
            "position_type": ["NEW", "INCREASED", "REDUCED", "EXITED", "FLAT"],
            "delta_pct": [np.inf, 0.25, -0.5, -1.0, 0.0],
        }
    )
    c = compute_conviction_score(df)["conviction_score"].to_numpy()
    assert c[0] > 0 and c[1] > 0 and c[2] < 0 and c[3] < 0 and c[4] == 0
    assert c[1] == pytest.approx(0.8 * np.log1p(0.25))
    assert c[0] == pytest.approx(0.8 * np.log1p(10.0))  # inf capped


def test_conviction_scales_with_urgency():
    base = {"position_type": ["INCREASED"], "delta_pct": [0.25]}
    lo = compute_conviction_score(pd.DataFrame({**base, "filing_urgency": [0.1]}))
    hi = compute_conviction_score(pd.DataFrame({**base, "filing_urgency": [0.9]}))
    assert hi["conviction_score"].iloc[0] > lo["conviction_score"].iloc[0] > 0


# ----------------------------------------------------------------- backtest
def _prices(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)),
        index=idx,
        columns=["AAA", "BBB"],
    )
    return px


def test_position_return_uses_only_prices_from_entry_onward():
    px = _prices()
    entry = px.index[50]
    base = _compute_position_return("AAA", entry, 20, px, "LONG")
    px2 = px.copy()
    px2.iloc[:50] *= 3.0  # scramble the past
    assert _compute_position_return("AAA", entry, 20, px2, "LONG") == pytest.approx(base)
    assert base == pytest.approx(px["AAA"].iloc[70] / px["AAA"].iloc[50] - 1)


def test_position_return_short_is_negative_of_long_and_nan_when_short_history():
    px = _prices()
    entry = px.index[50]
    lg = _compute_position_return("AAA", entry, 20, px, "LONG")
    sh = _compute_position_return("AAA", entry, 20, px, "SHORT")
    assert sh == pytest.approx(-lg)
    assert np.isnan(_compute_position_return("AAA", px.index[-5], 20, px, "LONG"))
    assert np.isnan(_compute_position_return("ZZZ", entry, 20, px, "LONG"))


def test_drawdown_is_fraction_of_peak_equity():
    equity = pd.Series([1.0, 1.5, 2.0, 1.0, 1.2])
    dd = _compute_drawdown(equity)
    assert dd.iloc[2] == 0.0
    assert dd.iloc[3] == pytest.approx(-0.5)
    assert (dd <= 0).all()


def test_long_short_backtest_accounting():
    px = _prices()
    filing = px.index[40]
    sig = pd.DataFrame(
        {
            "quarter_label": ["Q1_2020", "Q1_2020"],
            "ticker": ["AAA", "BBB"],
            "position_side": ["LONG", "SHORT"],
            "filing_date_only": [filing, filing],
        }
    )
    res = run_long_short_backtest(sig, px, hold_days=20, tc_bps=10)
    row = res.iloc[0]
    exp_long = px["AAA"].iloc[60] / px["AAA"].iloc[40] - 1
    exp_short = -(px["BBB"].iloc[60] / px["BBB"].iloc[40] - 1)
    assert row["long_return"] == pytest.approx(exp_long)
    assert row["short_return"] == pytest.approx(exp_short)
    assert row["tc_drag"] == pytest.approx(4 * 10 * 1e-4)
    assert row["net_ls_return"] == pytest.approx(exp_long + exp_short - row["tc_drag"])
    assert res["drawdown"].max() <= 0
