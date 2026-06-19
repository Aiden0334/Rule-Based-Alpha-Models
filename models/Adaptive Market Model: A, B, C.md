# 1. Core Hypothesis & Theoretical Background

This framework is designed to empirically test the Adaptive Market Hypothesis. The core objective is to disprove the strict efficient market hypothesis by demonstrating the statistically significant alpha can be extracted when the market deviates from a random walk. 

We set up three progressive models to isolate the exact source of alpha: 

**Model A (Baseline)** ──> **Model B (Regime Filtered)** ──> **Model C (V1) (Fully Optimized)**

  **Pure Price Structure** ──> **Statistical Inefficiency** ──> **Dynamic Risk & Momentum**

# The Monotonic Improvement Hypothesis

## Hypothesis: 

If the Variance Ratio (VR) contains genuine predictive information about market regimes, performance metrics will show monotonic improvement (Model A < Model B < Model C) 

**VR < 1 (Mean Reversion Dominant):** Volatility increases slower than a random walk; prices tend to return to the historical average. 

**VR > 1 (Trend Dominant):** Volatility increases faster than a random walk; prices tend to persist in the current direction. 


# 2. Technical Formulations

## 2.1. Variance Ratio (VR) 

Measured at q = 16 (approx. 3days) and q = 30 (approx. 5 days) on 4-hour bars: 

$$VR(q) = \frac{Var[r_q]}{q \cdot Var[r_1]}$$

## 2.2 Bollinger Bands & Width (BBW)

$$\text{Upper, Lower} = MA(20) \pm 2 \cdot \sigma(20)$$

$$BBW(t) = \frac{\text{Upper} - \text{Lower}}{MA(20)}$$


# 3. Step by Step Strategy Evolution

## 3.1 Model A: Simple Mean-Reversion (The Baseline)

- Concept: A naive, pure price-structure model operating under the assumption that the market is always mean-reverting.
  
- Entry Rule: Enter Long when price touches the Lower Bollinger Band ($-2\sigma$); Enter Short when price touches the Upper Bollinger Band($+2\sigma$).
  
- Exit Rule: Touch the opposite band.
  
- Flaw: Completely exposed to "band walking" during strong trending regimes and led to catastrophic losses in persistent markets.

## 3.2 Model B: VR-Filtered Mean-Reversion

- Concept: Introduces the Variance Ratio as a structural regime filter to eliminate false reversion signals during trends.
  
- Entry Rule: Execute the same Bollinger Band breakout entry **ONLY IF** both short-term and long-term VR values indicate a strong mean-reverting regime:
  
  $$VR_{16} < 0.95 \quad \text{AND} \quad VR_{30} < 0.95$$
  
- Exit Rule : Touch the opposite band.
  
- Significance: Filters out toxic trend intervals and convert a losing baseline into a statistically viable strategy.

**Note on Feature Selection: Variance Ratio vs Hurst Exponent**

The Hurst Exponent was intentionally excluded to prevent multicollinearity and information redundancy. Since both metrics mathematically capture identical market dynamics and evaluate the speed of variance diffusion to identify random walks versus structural regimes, then layering them would result in severe signal collision. To maintain a clean and high-signal-to-noise ratio within our rule-based architecture, we exclusively deployed the Variance Ratio, which provided a more numerical and stable regime classification framework. 

## 3.3 Model C (Named Model V1) 

- Concept: A complete asset-allocation framework that not only avoids trends but exploits them and backed by professional risk management.
  
- Entry Rules:
  - a. Reversion: $VR_{16} < 0.95$ AND $VR_{30} < 0.95$ $\rightarrow$ BB $\pm2\sigma$ Breakout.
  - b. Core Momentum: $VR_{16} > 1.05$ AND $VR_{30} > 1.05$ $\rightarrow$ 10-bar Directional Chase.
  - c. Momentum Assist: $BBW_{\text{Expansion}}$ AND $\text{Band Walking}$ $\rightarrow$ Trend Following.

- Exit Rules: **Reversion:** Opposite band touch or Fixed Stop-loss (ATR * 4).
  - **Trend:** Trailing Stop (ATR * 3) or Trend signal breakdown.

 
# 4. Empirical Verification Metrics

The backtest on U.S. Equity Index Futures 4-hour bars (2018-2026) yielded a perfect monotonic validation of our core hypothesis: 

| Strategy | Description | Sharpe Ratio |
|:---------|:------------|------------:|
| Model A  | Simple BB ±2σ Reversion | -0.028 |
| Model B  | Model A + VR Filters (VR < 0.95) | 0.488 |
| Model C  | Model B + Momentum + ATR Stops | 0.965 |


# 5. Statistical Significance

**Hypothesis Framework:**

- H₀ (Null): Per-trade mean return = 0 (no alpha, EMH holds)
- H₁ (Alternative): Mean return > 0 (positive alpha)
- Test: One-sided t-test
- Significance levels: p < 0.10 (marginal), p < 0.05 (significant), p < 0.01 (highly significant)

**Results:**

| Strategy | Sharpe | t-statistic | p-value | Significance |
|:---------|-------:|------------:|--------:|--------------|
| Model A | -0.028 | -0.08 | 0.53 | Not significant |
| Model B | 0.488 | 1.43 | 0.077 | Marginal (p < 0.10) |
| Model C | 0.965 | 2.76 | 0.003 | Highly significant (p < 0.01) |

**Interpretation:**

- Model A: No statistical evidence of alpha — random walk hypothesis holds.
- Model B: First quantitative evidence that VR provided meaningful regime information beyond price structure.
- Model C: Strong statistical rejection of EMH for this strategy on this asset class.

The transition from p = 0.53 → 0.077 → 0.003 mathematically validated the monotonic improvement hypothesis stated in Section 1.

**Test Assumptions and Limitations:**

- Per-trade returns assumed approximately normal. (verified via Q-Q plot)
- Trade independence enforced by single-position constraint.
- p-values are in-sample; OOS robustness verified separately.


# Key Findings

**1. Validation of VR:** The dramatic shift in Sharpe froom Model A (-0.028) to Model B (0.488) mathematically proved that the Variance Ratio successfully diferentiaited between random noise and mean-reverting inefficiencies. 

**2. The Necessity of Symmetry:** Model C achieved its premium alpha (Sharpe 0.965) because it stopped fighting trends and instead turned them into an independent source of return. This model was finalized as Rule-Based-Alpha-Model V1 for low-to-medium volatility index markets. 
