"""Checks on the hidden-positions study: splits, spikes, windows and pairing."""
import numpy as np
import pandas as pd
import pytest

from src import hidden as H

Q = H.quarter_ends()


def test_split_ratio_from_continuing_holders():
    assert H.split_ratio(4.02, 50) == 4.0
    assert H.split_ratio(0.1002, 30) == pytest.approx(0.1)          # 1-for-10 reverse split
    assert H.split_ratio(1.5, 12) == 1.5
    assert H.split_ratio(1.12, 50) == 1.0                            # ordinary buying
    assert H.split_ratio(2.2, 50) == 1.0                             # 10% off any ratio
    assert H.split_ratio(4.0, 3) == 1.0                              # too few holders to tell
    assert H.split_ratio(np.nan, np.nan) == 1.0


def test_split_factors_rescale_earlier_quarters():
    p = pd.DataFrame({"holder_ratio": [np.nan, 1.0, 7.0, 1.01, 4.0], "holders_both": [np.nan, 50, 50, 50, 50]})
    f = H.split_factors(p).to_numpy()
    assert f == pytest.approx([1 / 28, 1 / 28, 1 / 4, 1 / 4, 1.0])


def test_drop_spikes():
    s = pd.Series([10.0, 10.5, 11000.0, 11.0, 0.002, 12.0, 60.0])
    out = H.drop_spikes(s)
    assert np.isnan(out[2]) and np.isnan(out[4])
    assert out[6] == 60.0                                            # last value has one neighbour only
    assert out[[0, 1, 3, 5]].tolist() == [10.0, 10.5, 11.0, 12.0]


def panel(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values, index=Q[: len(next(iter(values.values())))]).reindex(Q)


def test_returns_between_and_exit_at_last_price():
    p = panel({"A": [10, 11, 12, 13, 14], "B": [10, 20, np.nan, np.nan, np.nan], "C": [np.nan, 5, 6, 7, 8]})
    r = H.returns_between(p, ["A", "B", "C", "D"], Q[0], 2)
    assert r["A"] == pytest.approx(0.2) and r["B"] == pytest.approx(1.0)     # B exits at its last price
    assert "C" not in r and "D" not in r                                     # no start price, no data


def test_returns_between_drops_suspected_splits():
    p = panel({"A": [10, 11, 12, 13, 14], "B": [10, 5, 5, 5, 5]})
    s = pd.DataFrame(False, index=Q, columns=["A", "B"])
    s.loc[Q[1], "B"] = True
    r = H.returns_between(p, ["A", "B"], Q[0], 2, s)
    assert list(r.index) == ["A"]
    assert "B" in H.returns_between(p, ["A", "B"], Q[1], 2, s)               # split before the window


def test_window_must_end_inside_the_data():
    p = panel({"A": [10, 11, 12]})
    assert H.returns_between(p, ["A"], Q[1], 4).empty


def test_reveal_quarter():
    assert H.first_quarter_end_on_or_after(pd.Timestamp("2020-02-14")) == pd.Timestamp("2020-03-31")
    assert H.first_quarter_end_on_or_after(pd.Timestamp("2020-03-31")) == pd.Timestamp("2020-03-31")


def test_pairing_drops_shared_securities_and_weights_by_value():
    period = Q[0]
    pos = pd.DataFrame({"CIK": "1", "manager": "M", "period": period,
                        "side": ["hidden", "hidden", "shown", "shown", "shown"],
                        "CUSIP": ["H1", "S1", "S1", "S2", "S3"], "value": [100, 300, 50, 50, 150]})
    amend = pd.DataFrame({"CIK": ["1"], "period": [period], "filed": [period + pd.Timedelta(days=100)]})
    p = panel({"H1": [10, 20, 20, 20, 20, 20, 20], "S1": [10, 10, 10, 10, 10, 10, 10],
               "S2": [10, 11, 11, 11, 11, 11, 11], "S3": [10, 9, 9, 9, 9, 9, 9]})
    out = H.pair_returns(pos, amend, p)
    r = out[(out.anchor == "report") & (out.k == 1)].iloc[0]
    assert r.n_hidden == 2 and r.n_shown == 2                          # S1 counts as hidden only
    assert r.hidden_ew == pytest.approx((1.0 + 0.0) / 2)
    assert r.hidden_vw == pytest.approx((100 * 1.0 + 300 * 0.0) / 400)
    assert r.shown_ew == pytest.approx((0.1 - 0.1) / 2)
    assert r.shown_vw == pytest.approx((50 * 0.1 + 150 * -0.1) / 200)
    rv = out[(out.anchor == "reveal") & (out.k == 1)].iloc[0]
    assert rv.start == Q[2]                                            # filed 100 days after quarter end


def test_t_test_by_period_counts_quarters_not_pairs():
    df = pd.DataFrame({"period": [Q[0]] * 50 + [Q[1], Q[2], Q[3]], "x": [1.0] * 50 + [-1.0, -1.0, -1.0]})
    r = H.t_test_by_period(df, "x")
    assert r["quarters"] == 4 and r["pairs"] == 53 and r["mean"] == pytest.approx(-0.5)


# ---------- the frozen prices against known values ----------

KNOWN = {  # CUSIP: (date, split-adjusted close in today's shares)
    "037833100": ("2014-03-31", 536.74 / 28),    # Apple: 7-for-1 in 2014, 4-for-1 in 2020
    "67066G104": ("2021-03-31", 533.93 / 40),    # Nvidia: 4-for-1 in 2021, 10-for-1 in 2024
    "88160R101": ("2020-06-30", 1079.81 / 15),   # Tesla: 5-for-1 in 2020, 3-for-1 in 2022
    "02079K305": ("2021-12-31", 2897.04 / 20),   # Alphabet: 20-for-1 in 2022
    "023135106": ("2021-12-31", 3334.34 / 20),   # Amazon: 20-for-1 in 2022
    "369604301": ("2021-12-31", 94.47),          # GE: new CUSIP after its 2021 reverse split, so none here
    "594918104": ("2015-12-31", 55.48),          # Microsoft: no split
    "30231G102": ("2019-12-31", 69.78),          # Exxon Mobil: no split
}


@pytest.fixture(scope="module")
def frozen_panel():
    _, _, prices = H.load()
    return H.price_panel(prices)


@pytest.mark.parametrize("cusip", sorted(KNOWN))
def test_known_prices(frozen_panel, cusip):
    date, price = KNOWN[cusip]
    assert frozen_panel.loc[pd.Timestamp(date), cusip] == pytest.approx(price, rel=0.005)


def test_t_test_by_period_uses_newey_west_for_overlapping_windows():
    # A series that repeats each value for three quarters, as overlapping windows do,
    # has a far larger long run variance than its plain variance suggests.
    vals = np.repeat([0.05, -0.02, 0.04, 0.01, -0.03, 0.06, 0.02, -0.01], 3)
    df = pd.DataFrame({"period": pd.period_range("2015Q1", periods=len(vals), freq="Q").astype(str), "x": vals})
    plain = H.t_test_by_period(df, "x")
    nw = H.t_test_by_period(df, "x", lags=3)
    e = vals - vals.mean()
    n = len(e)
    s2 = e @ e / n + sum(2 * (1 - j / 4) * (e[j:] @ e[:-j]) / n for j in range(1, 4))
    assert nw["t"] == pytest.approx(vals.mean() / np.sqrt(s2 / n))
    assert abs(nw["t"]) < abs(plain["t"])
    assert plain["t"] == pytest.approx(vals.mean() / (vals.std(ddof=1) / np.sqrt(n)))
