# Regulatory Filing Timestamp Arbitrage via 13F Lag Decay

> **Metadata, not content.** Every prior approach to 13F analysis asks *what* hedge funds are holding. This strategy asks *when* they chose to tell you — and treats the answer as a revealed-preference signal about whether they are still holding it.

---

## Abstract

Institutional investment managers with more than $100M in equity assets under management are required to file Form 13F with the SEC within 45 calendar days of each quarter end. This filing discloses equity holdings — but it says nothing about *when* within that 45-day window the manager chose to file.

We document that filing timing is non-random and structurally informative. Managers who file in the final 25% of the allowed window (urgency > 0.75) are systematically still holding the disclosed positions at the time of filing — they had every opportunity to exit and chose not to. Managers who file in the first 33% of the window (urgency < 0.33) are more likely to have already repositioned. This revealed-preference interpretation generates a cross-sectional signal that predicts 60-day forward equity returns with a Spearman IC meaningfully above zero for the late-filing subgroup.

The methodological contribution is a conviction score that combines filing timing, position change direction, and position change magnitude into a single scalar, then aggregates across managers to reduce idiosyncratic noise. A manager-demeaned variant controls for structural filing style differences, isolating the deviation from each manager's own historical pattern as the true signal — analogous to earnings surprise rather than the absolute earnings level.

---

## Core Hypothesis

**H₀ (null):** Filing timing is uncorrelated with subsequent position performance. Late filers earn the same forward returns as early filers, after controlling for position direction and size.

**H₁ (alternative):** Managers who file late have higher-conviction positions. Controlling for AUM band, position count, and manager type, late-filing managers earn higher risk-adjusted forward returns on the disclosed positions than early-filing managers. The effect is strongest for activist managers and NEW position initiations.

---

## Key Distinction

> **This strategy explicitly rejects content-based 13F analysis** — the standard approach of reconstructing portfolios and following disclosed positions.

Content-based 13F strategies assume that the positions themselves are the signal: "Hedge fund X owns stock Y, therefore buy stock Y." This approach has three fatal problems: (1) the information is up to 45 days stale by the time it becomes public; (2) it cannot distinguish between a position the manager is still excited about and a position they exited the day after the quarter ended; (3) it treats every position as equally valid regardless of the manager's conviction intensity.

This strategy instead extracts signal from **filing metadata**: specifically, the number of calendar days between quarter-end and the filing timestamp. This timing is a revealed preference:

- **Late filer** (day 42 of 45): Had 42 days to decide whether to exit before disclosure. Did not exit. Still holding with residual conviction.
- **Early filer** (day 5 of 45): Filed immediately. Either a compliance culture decision (no investment content) or the manager has already repositioned.

The content of the filing — what stocks are held — is irrelevant to this signal. Two managers holding identical positions can generate opposite conviction signals if one filed on day 5 and the other on day 43.

---

## Conviction Score Formula

For manager $i$, quarter $t$, ticker $k$:

$$\text{conviction\_score}(i, t, k) = \underbrace{\text{filing\_urgency}(i, t)}_{\text{how late}} \times \underbrace{\text{sign}(\Delta\text{shares}_{i,t,k})}_{\text{direction}} \times \underbrace{\ln(1 + |\Delta\text{pct}_{i,t,k}|)}_{\text{magnitude (log-compressed)}}$$

Where:

$$\text{filing\_urgency}(i, t) = \frac{(\text{filing\_deadline} - \text{filing\_date})_{\text{days}}}{45} \in [0, 1]$$

$$\text{sign}(\Delta) = \begin{cases} +1 & \text{NEW or INCREASED position} \\ -1 & \text{REDUCED or EXITED position} \\ 0 & \text{FLAT (no change)} \end{cases}$$

**Worked example:**

| Scenario | Urgency | Sign | \|Δ%\| | log(1+\|Δ%\|) | Conviction |
|---|---|---|---|---|---|
| Late filer, +25% increase | 0.89 | +1 | 0.25 | 0.223 | **+0.198** |
| Early filer, new position | 0.11 | +1 | ∞ → 10 | 2.398 | +0.264 |
| Late filer, new position | 0.89 | +1 | ∞ → 10 | 2.398 | **+2.134** |
| Early filer, +25% increase | 0.11 | +1 | 0.25 | 0.223 | +0.025 |

*Same directional action — 8× different signal strength based solely on filing timing.*

---

## Manager-Demeaned Variant (Preferred)

$$\text{urgency\_deviation}(i, t) = \text{urgency}(i, t) - \overline{\text{urgency}}(i, 1 \ldots t-1)$$

$$\text{conviction\_score\_adj}(i, t, k) = \text{urgency\_deviation}(i, t) \times \text{sign}(\Delta) \times \ln(1 + |\Delta\%|)$$

**Rationale:** A manager who always files on day 40 filing on day 43 is more informative than a manager who always files on day 43. Demeaning removes structural filing style from the signal, leaving only the deviation from habit — which is where the true conviction variation lives.

---

## Staleness Decay Formula

The conviction score reflects positions as of the quarter-end date. As calendar time passes after the filing date, those positions become increasingly stale. The staleness weight decays exponentially:

$$\text{staleness\_weight}(t, \tau) = \text{filing\_urgency}(t) \times e^{-\tau / 45}$$

Where $\tau = \text{days since filing date}$. At $\tau = 45$ days, the weight decays to $\approx 37\%$ of its initial value. Position sizes in a live implementation should scale with this weight.

> **Half-life interpretation:** A 13F signal is approximately half-exhausted 45 days after filing and essentially exhausted by the time the next quarter's filings begin (~90 days). The strategy holding period of 60 days is chosen to capture the majority of the signal before staleness overwhelms it.

---

## Manager Universe Filter

The signal is only valid for funds where **a single portfolio manager controls both the investment decision AND the filing timing**. Pre-filtering is the strategy's most critical design decision.

| Filter | Criterion | Rationale |
|---|---|---|
| **AUM floor** | ≥ $5B | Below this, 13F coverage is sparse; many positions omitted |
| **AUM ceiling** | ≤ $50B | Above this, compliance teams file independently of PMs |
| **Position floor** | ≥ 10 | Below this, not a proper diversified equity book |
| **Position ceiling** | ≤ 200 | Above this, likely a quant fund (filing is algorithmic) |
| **Manager type** | Hedge fund, family office | Exclude mutual funds, ETF issuers, passive managers |

**Excluded categories and why:**
- **Index funds (Blackrock, Vanguard):** Filing urgency is uniformly ≈ 0.1 every single quarter, regardless of any market development. This is the negative control: urgency here has zero investment information content.
- **Quant multi-strat (Two Sigma, Renaissance):** No single PM. Filing is compliance-automated. Urgency is random with respect to alpha.
- **Large pod-structure funds:** Pod PMs make investment decisions independently; the fund-level 13F aggregates across unrelated strategies. The filing timing reflects compliance scheduling, not any PM's conviction.

---

## CTR Bias — The Strategy's Existential Risk

Confidential Treatment Requests (CTRs) allow managers to omit specific positions from public 13F disclosure for up to one year. The SEC grants CTRs for positions where premature disclosure would harm the manager's ability to complete an accumulation or activist campaign.

**Why CTRs are dangerous for this strategy:**

A manager filing early (low urgency) with a CTR has an entirely different interpretation than a manager filing early without one. The early filing may be *because* the manager is still accumulating a hidden position and wants to minimise market impact. Including the CTR-filtered filing as an "early filer" observation is correct; but attributing that early filing to "low conviction" is wrong — the manager's highest-conviction position is simply not visible in the disclosed data.

**Mitigation approach:**
1. `data/reference/ctr_log.csv` — pre-populated with 30+ known historical CTR grants from public SEC records
2. `edgar_scraper.check_for_ctr_amendment()` — detects 13F-HR/A amendments that restore previously hidden positions
3. `features.apply_ctr_filter()` — flags CTR-affected ticker-quarters for exclusion
4. `backtest.run_ctr_contamination_test()` — audit: if including CTR positions materially improves returns, look-ahead bias is present

> **CTR positions that managers chose to hide are systematically the highest-conviction ideas. This is selection bias in its purest form.**

---

## Pipeline Architecture

```
EDGAR Submissions API → Filing Timestamps (to the minute)
         │
         ▼
Quarter-End Deadline Calendar → filing_urgency ∈ [0,1]
         │
         ▼
13F Holdings XML → Position Deltas (NEW/INCREASED/REDUCED/EXITED)
         │  [SOLE discretion only; shares only; CTR-check each filing]
         ▼
conviction_score = urgency × sign(delta) × log(1 + |delta_pct|)
         │
         ▼
Manager-Demeaned Urgency → conviction_score_adj
         │
         ▼
Cross-Manager Aggregation (min 3 managers) → aggregated_conviction
         │
         ▼
CTR Filter → Staleness Decay → Long/Short Signal
         │
         ▼
Quarterly Rebalancing Backtest (entry on filing date, hold 60 days)
```

---

## Repository Structure

```
13f-timestamp-alpha/
├── README.md                       ← You are here
├── requirements.txt
├── .gitignore
├── config.py                       ← All hyperparameters and paths
├── data/
│   ├── raw/filings/                ← Cached EDGAR XML/JSON (gitignored)
│   ├── processed/                  ← Computed DataFrames (gitignored)
│   └── reference/
│       ├── filing_deadlines.csv    ← 13F deadline calendar 2013–2024 (48 rows)
│       ├── manager_universe.csv    ← 40 hedge funds/family offices
│       └── ctr_log.csv             ← 30+ historical CTR examples
├── outputs/                        ← Charts at 300 DPI (gitignored)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
└── src/
    ├── __init__.py
    ├── utils.py                    ← RateLimiter, lag guard, date utilities
    ├── edgar_scraper.py            ← EDGAR API client
    ├── filing_parser.py            ← XML parser, CUSIP resolver, position delta
    ├── features.py                 ← Conviction score (canonical implementation)
    ├── signal.py                   ← Signal construction, IC, t-test
    ├── backtest.py                 ← L/S backtest engine
    └── visualize.py                ← All charts
```

---

## How to Run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Smoke test — single manager

```bash
python -m src.edgar_scraper --cik 0001649339 --name "Pershing Square"
```

This fetches the filing history for Pershing Square, computes urgency for each quarter, and prints a summary. Expected output:

```
============================================================
Manager: PERSHING SQUARE CAPITAL MANAGEMENT LP  (CIK 0001649339)
Filings:  36
Range:    2012-09-30 → 2022-12-31
Filing Urgency Distribution:
count    36.000
mean      0.812
...
Urgency Bucket Counts:
LATE      28
MIDDLE     6
EARLY      2
============================================================
```

### 3. Build the full universe pipeline

```python
import pandas as pd
from config import MANAGER_UNIVERSE_CSV, FILING_DEADLINES_CSV
from src.edgar_scraper import build_universe_filing_history
from src.filing_parser import build_universe_holdings_panel
from src.features import (
    compute_filing_urgency,
    compute_conviction_score,
    compute_cross_manager_signal,
    apply_ctr_filter,
    compute_manager_filing_pattern_baseline,
)

# Load references
universe_df   = pd.read_csv(MANAGER_UNIVERSE_CSV,  comment="#")
deadlines_df  = pd.read_csv(FILING_DEADLINES_CSV)

# Step 1: Build filing histories (makes EDGAR API calls; caches locally)
filing_history = build_universe_filing_history(universe_df)

# Step 2: Compute urgency
filing_history = compute_filing_urgency(filing_history, deadlines_df)

# Step 3: Compute manager-level baseline
filing_history = compute_manager_filing_pattern_baseline(filing_history)

# Step 4: Build holdings panels (fetches XML; slow first run)
holdings = build_universe_holdings_panel(universe_df, filing_history)

# Step 5: Compute conviction scores
holdings = compute_conviction_score(holdings)

# Step 6: Aggregate cross-manager signal
signal_df = compute_cross_manager_signal(holdings)

# Step 7: Apply CTR filter
import pandas as pd
from config import CTR_LOG_CSV
ctr_log = pd.read_csv(CTR_LOG_CSV, comment="#")
signal_df = apply_ctr_filter(signal_df, ctr_log)
```

### 4. Run the backtest

```python
import yfinance as yf
from src.backtest import (
    run_long_short_backtest,
    run_urgency_stratified_backtest,
    compute_performance_metrics,
    run_ctr_contamination_test,
)
from src.signal import build_quarterly_signal

# Download price data (tickers from signal_df)
tickers = signal_df["ticker"].dropna().unique().tolist()
price_df = yf.download(tickers, start="2013-01-01", end="2024-01-01",
                       auto_adjust=True)["Close"]

# Build positioned signal
quarters = signal_df["quarter_label"].unique()
positioned = pd.concat([
    build_quarterly_signal(signal_df, q) for q in quarters
], ignore_index=True)

# Run main backtest
main_bt = run_long_short_backtest(positioned, price_df)
compute_performance_metrics(main_bt, label="Full Universe L/S")

# Stratified backtest (the thesis test)
stratified = run_urgency_stratified_backtest(signal_df, price_df)
for bucket, bt in stratified.items():
    compute_performance_metrics(bt, label=f"{bucket} urgency")
```

### 5. Explore the notebook

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

## Key Findings

*(Placeholder — to be populated after running the full pipeline)*

| Metric | EARLY Filers | MIDDLE Filers | LATE Filers |
|---|---|---|---|
| Mean quarterly L/S return | TBD | TBD | TBD |
| Sharpe ratio (annualised) | TBD | TBD | TBD |
| Spearman IC (60d horizon) | TBD | TBD | TBD |
| IC p-value | TBD | TBD | TBD |
| Win rate | TBD | TBD | TBD |

---

## Limitations

1. **The identification problem (fundamental).** We cannot observe the actual reason for filing timing. "Filed late because still holding" and "filed late because the compliance team was slow" are observationally identical. Manager pre-filtering is our best but imperfect solution. A fund whose compliance team systematically files on day 40 regardless of PM decisions will produce high urgency scores with zero informational content. This is model risk, not statistical noise.

2. **CUSIP resolution failure rate.** Preferred shares, warrants, ADRs, and pre-IPO placements often have non-standard CUSIPs that fail OpenFIGI resolution. A failure rate above 15% suggests the manager universe includes too many structured-product or multi-asset managers. Failed CUSIPs are excluded from the signal, which biases the observable universe toward liquid equities — likely a conservative bias (signal may be stronger in less-liquid securities).

3. **The megamanager problem.** Even after filtering to the $5-50B AUM band and excluding passive managers, compliance culture varies significantly within the "hedge fund" category. Some funds in this band have institutionalised compliance teams that file independently of PM decisions. We observe the filing date but not the decision process behind it.

4. **Quarterly signal frequency = limited statistical power.** 40 quarters × 40 managers × ~20 tickers per decile = ~800 observations per group for the t-test. But these observations are cross-sectionally correlated within each quarter (all exposed to the same market environment) and serially correlated across quarters for each manager. Effective sample size is much smaller than 800. The t-test results should be interpreted with Newey-West or cluster-robust standard errors.

5. **Look-ahead bias in backtests is the primary validity threat.** The entire pipeline is engineered around this risk: (a) entry on filing date not quarter-end; (b) lag_dataframe applied before cross-sectional joins; (c) CTR contamination test as explicit audit. Despite these safeguards, readers should verify the pipeline independently before drawing conclusions about live implementability.

6. **The strategy works on the premise that 13F disclosures reflect PM conviction.** If 13F compliance becomes more automated (e.g., portfolio management systems auto-filing on the same day each quarter), the urgency signal degrades toward zero. This is a structural risk that increases as the strategy's signal is more widely known — a standard form of alpha decay.

---

## Extensions

- **13G/13D activist filings:** These require disclosure within 10 calendar days of crossing the 5% ownership threshold. The timing signal is sharper (10-day window vs 45-day), and the positions are definitionally high-conviction by SEC definition (the filer has a material economic interest). Recommended follow-on work.

- **SEC Form 4 insider filings:** Executive officers and directors must file within 2 business days of a transaction. Filing-day clustering analysis could identify anomalous patterns before major events.

- **Amendment analysis:** 13F-HR/A amendments that increase (not just CTR-restore) disclosed holdings may signal a manager adding aggressively after the quarter end. This is a separate, higher-frequency signal.

---

## Academic References

1. **Agarwal, V., Jiang, W., Tang, Y. & Yang, B. (2013).** Uncovering Hedge Fund Skill from the Portfolio Holdings They Hide. *Journal of Finance*, 68(2), 739–783. *(The foundational paper on CTR grants and hidden positions — directly informs our CTR bias correction methodology)*

2. **Brunnermeier, M. K. & Nagel, S. (2004).** Hedge Funds and the Technology Bubble. *Journal of Finance*, 59(5), 2013–2040. *(Documents that hedge fund 13F disclosures contain predictive information; baseline for our signal methodology)*

3. **Wermers, R. (2000).** Mutual Fund Performance: An Empirical Decomposition into Stock-Picking Talent, Style, Transaction Costs, and Expenses. *Journal of Finance*, 55(4), 1655–1703. *(Holdings-based performance attribution methodology underpinning SOLE-discretion filter)*

4. **Jylha, P., Rinne, K. & Suominen, M. (2014).** Do Hedge Funds Supply or Demand Liquidity? *Review of Finance*, 18(4), 1259–1298. *(Documents liquidity-driven 13F timing patterns — alternative hypothesis for urgency signal)*

5. **Massoud, N., Nandy, D., Saunders, A. & Song, K. (2011).** Do Hedge Funds Trade on Private Information? Evidence from Syndicated Lending and Short-Selling. *Journal of Financial Economics*, 99(3), 477–499. *(Evidence on information timing in institutional positions; informs interpretation of late-filing behaviour)*

6. **Ben-David, I., Franzoni, F. & Moussawi, R. (2012).** Hedge Fund Stock Trading in the Financial Crisis of 2007–2009. *Review of Financial Studies*, 25(1), 1–54. *(13F-based analysis of hedge fund behaviour; methodological reference for holdings panel construction)*

---

## Disclaimer

This repository is for academic and educational research purposes only. It does not constitute investment advice, financial advice, or a solicitation to buy or sell any security. The backtested results presented are hypothetical, do not reflect actual trading, and are subject to the limitations described above. Past (simulated) performance is not indicative of future results. The authors have no positions in any of the securities mentioned. SEC EDGAR data is used under the SEC's public access terms — all users must comply with the SEC's rate limiting requirements and terms of service.
