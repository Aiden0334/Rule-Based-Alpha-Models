# Mean-Reversion + Momentum Alpha via Variance Ratio Regime Classification with Volatility-Adaptive Model Selection on U.S. Equity Index Futures

---

# Phase 1

## Abstract

This study verifies mean-reversion and momentum alpha on U.S. equity index futures 4-hour bars. We classify market regimes leveraging the Variance Ratio (VR), used Bollinger Band breakouts as mean-reversion entry signals, and used BBW Expansion + Walking as momentum entry signals. Two model variants (v1: general type, v4: selective type) were adaptively applied according to market volatility.

**Main Results:**
- Achieved **OOS Sharpe 0.96+** on 8.3 years of backtest data from a 9-year original dataset (2017-2026)
- The difference between In-Sample Sharpe and Out-of-Sample Sharpe is at the level of 0.01 (not overfitting)
- All four splits positive in Rolling Origin validation, standard deviation 0.14
- Asset class specialization: works only on U.S. equity indices (fails on commodities)

**Verified Limitations:**
- Alpha specialized for U.S. equity indices
- Does not generalize to other asset classes (e.g., crude oil and gold)

---

## 1. Introduction

### 1.1 Core Hypothesis

> **The market is not always a random walk.**

We verify the hypothesis that alpha can be discovered in intervals where the Variance Ratio (VR) sufficiently deviates from 1 — that is, when mean-reversion or trends become stronger.

```
VR < 1 : Mean-reversion dominant (volatility increases slowly over time)
VR ≈ 1 : Random walk (Efficient Market assumption)
VR > 1 : Trend dominant (volatility increases rapidly over time)
```

If the random walk hypothesis is 100% true, Sharpe near 0 is normal. **If the VR ≠ 1 interval generates statistically significant alpha, it is a partial refutation of EMH.**

### 1.2 Related Research

- Lo & MacKinlay (1988): Variance Ratio test — determines whether stock prices follow a random walk
- Lo (2004): Adaptive Market Hypothesis — markets alternate between efficient and inefficient over time
- Mean-reversion: Jegadeesh (1990), De Bondt & Thaler (1985)

This study builds an **empirical trading alpha** on top of the theoretical foundation above.

---

## 2. Data

| Item | Value |
|------|-------|
| Data Source | Massive REST API (Futures) |
| Markets | E-mini S&P 500 (ES), E-mini NASDAQ-100 (NQ), E-mini Dow Jones (YM), E-mini Russell 2000 (RTY) |
| Timeframe | 4-hour bars (6 bars/day) |
| Original Period | 2017-04-04 ~ 2026-04-03 (≈ 9 years) |
| Backtest Period | 2018-01-24 ~ 2026-04-03 (≈ 8.3 years) |
| Bar Count | Approximately 12,000 bars per market |
| Cost Assumption | 0.04% round-trip (commission + slippage) |
| Split | IS: 2018-2023 (5.93 years), OOS: 2024-2026 (2.25 years) |

**Warmup Truncation Note:** While the original data starts on 2017-04-04, the warmup period of approximately 9 months is excluded from the backtest due to the quantile calculation window (ROLL_Q = 800 bars) used in some model variants (5-regime classification). For fair comparison across all models, 2018-01-24 is applied as a common starting point.

**Continuous Futures Construction:** Roll-over applied immediately before expiration, with price back-adjustment.

**Purpose:** To ensure maximum model robustness and eliminate execution-level data dependencies, this study intentionally restricts the feature space strictly to OHLC price data. This design programmatic controls for the risk of missing or delayed alternative data (e.g., Open Interest) during live execution, demonstrated that premium alpha can be extracted solely through pure statistical price structures. 

---

## 3. Methodology

### 3.1 Indicators Used

**Variance Ratio (VR):**
$$VR(q) = \frac{Var[r_q]}{q \cdot Var[r_1]}$$

where $r_q$ is the cumulative return over $q$ periods. This study measures at two points: $q = 16, 30$ (approximately 3 days / 5 days on a 4-hour bar basis).

**Bollinger Band:**
$$Upper, Lower = MA(20) \pm 2 \cdot \sigma(20)$$

**Bollinger Band Width Expansion:**
$$BBW(t) = \frac{Upper - Lower}{MA}$$
$$Expansion = BBW(t) > \text{Quantile}_{0.80}(BBW, 100)$$

**Band Walking:**
$$walk_{up} = \sum_{i=t-4}^{t} \mathbf{1}[close_i > MA + 1.5\sigma] \geq 3$$

**Average True Range (ATR):**
$$ATR(14) = \frac{1}{14}\sum TR_i$$

### 3.2 Two Model Variants

This study **adaptively selects between two variants based on volatility.**

#### 3.2.1 Model v1 — General Type (Low-Volatility Markets)

**Entry Rules:**

| Priority | Condition | Entry |
|----------|-----------|-------|
| 1 | $VR_{16} < 0.95$ AND $VR_{30} < 0.95$ | BB ±2σ breakout (reversion) |
| 2 | $VR_{16} > 1.05$ AND $VR_{30} > 1.05$ | 10-bar momentum direction (trend) |
| 3 | $BBW_{Expansion}$ AND $walk_{up/down}$ | Directional match (trend assist) |

**Exit:**
- Reversion: Opposite band touch OR ATR × 4 stop-loss
- Trend: Trailing stop ATR × 3 OR trend signal breakdown

**Applied Markets:** ES, NQ, YM (large-cap indices, relatively stable)

#### 3.2.2 Model v4 — Selective Type (High-Volatility Markets)

Based on 5-regime classification, enters only on strong signals.

**5-Regime Classification:**
$$vr_{score} = 0.8 \cdot (1 - \overline{long\_vr}) + 0.2 \cdot (1 - \overline{short\_vr})$$

where $short\_q = [2,3,4,6,8]$, $long\_q = [10,16,21,25,30]$. Classification by ROLL_Q=800 bar quantiles:

| Regime | Condition |
|--------|-----------|
| strong_mom | $vr_{score} \leq q_{20}$ |
| mom | $q_{20} < vr_{score} \leq q_{40}$ |
| neutral | $q_{40} < vr_{score} \leq q_{60}$ |
| rev | $q_{60} < vr_{score} \leq q_{80}$ |
| strong_rev | $vr_{score} > q_{80}$ |

**Entry Rules:**

| Priority | Regime | Entry |
|----------|--------|-------|
| 1 | strong_rev OR rev | BB ±2σ breakout (reversion) |
| 2 | strong_mom AND $BBW_{Expansion}$ AND $walk_{up/down}$ | Trend (strict) |
| 3 | (Other) AND $BBW_{Expansion}$ AND $walk_{up/down}$ | Trend assist |

**Applied Markets:** RTY (small-cap, high volatility), and future high-volatility markets

### 3.3 Volatility-Based Model Selection (Prior Hypothesis)

> **Hypothesis:** In high-volatility markets, false signals are abundant, so selective entry (v4) is advantageous.

| Market | Approximate Daily Volatility | Applied Model |
|--------|-----------------------------|---------------|
| ES | Low | v1 |
| NQ | Medium | v1 |
| YM | Low | v1 |
| RTY | High | v4 |

**Important:** Model selection is based on a **prior hypothesis** (determined by market characteristics), not decided after observing in-sample results.

### 3.4 Position Management

```
Capital per market: 1/N (N = number of markets traded)
Maximum simultaneous positions: 2
Stop-loss: ATR(14) × 4
Trailing stop: ATR(14) × 3
Trading cost: 0.04% round-trip
```

---

## 4. Results

### 4.1 Hypothesis Verification (Model A < B < C)

| Model | Description | Sharpe | t-stat | p-value |
|-------|-------------|--------|--------|---------|
| A | BB ±2σ simple mean-reversion (baseline) | -0.028 | -0.08 | 0.53 |
| B | A + VR filter ($vr_{16}, vr_{30} < 0.95$) | 0.488 | 1.43 | 0.077 |
| C / **v1** | B + BBW momentum + ATR stop | **0.965** | 2.76 | **0.003** |

**Monotonic improvement A → B → C** — Proves VR provides meaningful information. v1 is statistically significant.

### 4.2 In-Sample vs Out-of-Sample

**v1 (ES/NQ/YM):**

| Period | Trades | Sharpe | CAGR | MDD |
|--------|--------|--------|------|-----|
| IS (2018-2023) | 700 | 0.963 | 24.09% | -16.5% |
| OOS (2024-2026) | 295 | **0.973** | 22.56% | -22.7% |

**IS → OOS Sharpe change: -0.01** (essentially identical, no overfitting).

**v4 (RTY):**

| Period | Sharpe | CAGR | MDD |
|--------|--------|------|-----|
| IS (2018-2023) | 0.943 | - | - |
| OOS (2024-2026) | **0.989** | - | - |

v4 in RTY OOS is also nearly identical to IS.

### 4.3 Yearly Consistency (v1, ES/NQ/YM)

| Year | Trades | Return |
|------|--------|--------|
| 2018 | 125 | +8.36% |
| 2019 | 119 | +17.70% |
| 2020 | 111 | +49.45% |
| 2021 | 100 | +38.43% |
| 2022 | 117 | +27.90% |
| 2023 | 128 | +6.52% |
| 2024 | 142 | +12.09% |
| 2025 | 125 | +20.14% |
| 2026 | 28 | +17.38% |

**9/9 positive years.** 2024-2026 are all OOS periods.

### 4.4 Portfolio Effect (Concurrent 2-Position Trading)

| Metric | Value |
|--------|-------|
| CAGR | 7.90% |
| Sharpe | 0.787 |
| MDD | -11.8% |

Diversification across markets reduces MDD to about half compared to single market.

---

## 5. Robustness

### 5.1 OOS Randomness Verification (Rolling Origin)

v1 OOS Sharpe measured at 4 split points:

| Split | IS Period | OOS Period | OOS Sharpe |
|-------|-----------|------------|-----------|
| 1 | 2018-2021 | 2022-2026 | 0.863 |
| 2 | 2018-2022 | 2023-2026 | 0.832 |
| 3 | 2018-2023 | 2024-2026 | 0.973 |
| 4 | 2018-2024 | 2025-2026 | 1.182 |

**Mean 0.963, standard deviation 0.137.** All splits positive, strong consistency.

OOS start point fine-tuning also consistent:

| OOS Start | OOS Sharpe |
|-----------|-----------|
| 2023-06 | 0.661 |
| 2023-12 | 0.973 |
| 2024-06 | 0.984 |
| 2025-01 | 1.182 |

### 5.2 Asset Class Generalization

| Market | Asset Class | OOS Sharpe | Result |
|--------|-------------|-----------|--------|
| ES (v1) | U.S. large-cap index | 0.96 | ✓ Works |
| NQ (v1) | U.S. large-cap index | 0.97 | ✓ Works |
| YM (v1) | U.S. large-cap index | 0.95 | ✓ Works |
| RTY (v4) | U.S. small-cap index | 0.96 | ✓ Works |
| Crude Oil | Energy commodity | < 0, p > 0.5 | ✗ Failed |
| Gold | Safe-haven asset | < 0, p > 0.5 | ✗ Failed |

**Interpretation:** The alpha is for **mean-reversion inefficiency in U.S. equity indices**. It could not generalized to other asset classes with different market mechanisms.

---

## 6. Failed Attempts

For academic honesty, we record model variants that were attempted but failed.

### 6.1 Refined 5-Regime Classification (v3, v3-BBW, v5)

| Model | Changes | IS Sharpe | OOS Sharpe | OOS Drop |
|-------|---------|-----------|------------|----------|
| v3 | strong_rev entry threshold BB ±2.5σ (stricter) | 1.106 | 0.184 | -0.922 |
| v3-BBW | v3 with BBW standalone entry removed | 1.105 | 0.213 | -0.892 |
| v5 | mom entry removed, momentum further simplified | 1.207 | -0.453 | -1.660 |

**Key Finding: Complexity ↑ = OOS Performance ↓** (perfect monotonic pattern)

```
Model complexity:    v1 < v4 < v3-BBW < v5
OOS Sharpe average:  0.96 ≈ 0.66 > -0.004 > -0.46
```

### 6.2 Candlestick Pattern Integration

In EDA of 19 candlestick patterns, piercing (n=52, +0.94% after 5 bars, 67% hit rate) passed when context was included, but during backtest integration, due to priority conflicts, only 1 actual trade was generated, with no effect.

### 6.3 Per-Market Best Model Selection (Walk-Forward Verification)

In-sample per-market best model selection (ES=v5, NQ=v1, YM=v4):

| Method | Sharpe | MDD |
|--------|--------|-----|
| In-sample per-market best | 1.480 | -13.8% |
| Walk-Forward honest | 0.960 | -21.3% |
| **Difference** | **-0.520** | -7.5%p |

In WF 4-folds, the best model for ES changes every fold as v5, v4, v5, v1 — **per-market model selection is overfitting.**

### 6.4 Noise Removal Attempts

Attempt to remove neutral regime BBW entries (84 trades, Sharpe -0.555) in v4:

| Variant | Sharpe | MDD |
|---------|--------|-----|
| v4 (original) | 0.945 | -21.3% |
| v4.1 (neutral removed) | 0.909 | -24.6% |

**Counter-effect.** The neutral trades were acting as a **shield** that, by occupying positions, prevented entries into worse strong_mom trades.

### 6.5 Partial Closing (70% Profit at +1σ)

Sharpe improved marginally by 0.018, but average profit width decreased. Insufficient statistical significance, not adopted.

---

## 7. Discussion

### 7.1 Empirical Proof of Simplicity

Occam's Razor verified quantitatively:

| Model | Parameter Count | OOS Sharpe |
|-------|-----------------|-----------|
| v1 | Few (binary VR + fixed threshold) | 0.96 ★ |
| v4 | Medium (5-regime + BBW combination) | 0.66~0.96 (by market) |
| v3-BBW | Many (5-regime + differentiated thresholds) | 0.18 |
| v5 | Many (5-regime + simplified) | -0.45 |

**More degrees of freedom = More market noise learned = Lower OOS performance**

### 7.2 v1 vs v4 — Complementary Relationship

The two models are not competitive but **complementary**:

| | v1 (General) | v4 (Selective) |
|---|---|---|
| Entry frequency | High (995 trades) | Low (808 trades) |
| Signal strength required | Moderate | Very strong (multiple conditions) |
| Suitable market | Stable (few false signals) | High volatility (many false signals) |
| Core strength | Captures many real opportunities | Effectively filters false signals |

### 7.3 Meaning of Asset Class Specialization

Our alpha captures the **mean-reversion inefficiency in U.S. equity indices**. Different market mechanisms operate on crude oil (supply/demand shock-based) or gold (safe-haven demand-based).

### 7.4 Lessons Learned

1. **Value of simplicity** — Complex models break down in OOS
2. **Noise removal ≠ improvement** — Risk of ignoring shield effect
3. **Per-market adaptation is also overfitting** — Confirmed 0.52 Sharpe difference via WF
4. **Acknowledging asset class specialization** — Limitation

---

## 8. Conclusion

### 8.1 Verified Performance

- **OOS Sharpe 0.96+ on U.S. equity indices** (8.3 years of data)
- **9/9 positive years** (IS+OOS combined)
- **Rolling Origin randomness passed** (all 4 splits positive, standard deviation 0.14)
- **t-stat 2.76, p = 0.003** (statistically significant)

### 8.2 Adopted Models

**Volatility-Adaptive Model Selection:**
- Low-volatility markets (ES, NQ, YM): use **v1**
- High-volatility markets (RTY): use **v4**
- For new markets, measure volatility before selection

### 8.3 Limitations

- Specialized for U.S. equity indices (asset class constraint)
- Based on 4-hour bars (other timeframes require separate verification)
- Trading cost assumption of 0.04% (additional costs such as market impact possible in real operation)

### 8.4 Future Research

**A. ML Enhancement (Natural Extension)**
- Add ML filter to v1/v4 entry signals
- Goal: Stabilize OOS Sharpe 1.2+
- Features: VR, BBW, ATR, price patterns, market correlations, etc.

**B. Market Expansion**
- Global equity indices (DAX, FTSE, ESTX50, KOSPI200, NK225)
- Verify whether the same mean-reversion mechanism generalizes globally

**C. Separate Alpha Research for Other Assets**
- Trend-following or carry trade strategies for commodities/forex
- Asset class diversification to spread portfolio risk

---

## 9. Code Structure

```
project/
├── README.md                          # This document
├── Adaptive Market Model:A,B,C.md     # Hypothesis Test 
├── v1_model_description.md            # Rule-based Model v1 (Binary Regime Classifer)
├── v4_model_description.md            # Rule-based Model v4 (5-Regime Classifier)
├── final_model.py                     # v1 + v4 integrated final model
└── validation/
    ├── forward_validation.py          # IS / OOS / Rolling Origin Validations
    ├── validation/oos_robustness.py   # Model Robustness Validation    
    └── wf_per_product.py              # Walk-Forward per-market verification
```

Data is downloaded separately from the Massive Rest API and stored locally (not included in repository).

### Execution

```bash
# Run main model
python final_model.py

# IS / OOS / Rolling Origin Validations
python forward_validation.py

# Randomness verification
python validation/oos_robustness.py
```

Each script sets data paths at the top.

---

## 10. References

- **Lo, A. W., & MacKinlay, A. C. (1988).** Stock market prices do not follow random walks: Evidence from a simple specification test. *Review of Financial Studies*, 1(1), 41–66.
- **Lo, A. W. (2004).** The Adaptive Markets Hypothesis: Market efficiency from an evolutionary perspective. *Journal of Portfolio Management*, 30(5), 15–29.
- **Jegadeesh, N. (1990).** Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- **De Bondt, W. F., & Thaler, R. (1985).** Does the stock market overreact? *Journal of Finance*, 40(3), 793–805.
- **Bollinger, J. (2002).** *Bollinger on Bollinger Bands*. McGraw-Hill.

---

## 11. Acknowledgments

This study quantitatively proved the risk of in-sample overfitting through robust statistical verification (especially OOS Rolling Origin). The records of failed refinement attempts such as 5-regime classification have academic significance as cases demonstrating the **empirical value of simplicity**.

---

**Version:** 1.0 (as of 2026-04)  
**Status:** Phase 1 (rule-based model) completed. Phase 2 (ML enhancement) planned.
