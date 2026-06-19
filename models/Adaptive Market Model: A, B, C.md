# Core Hypothesis & Theoretical Background

This framework is designed to empirically test the Adaptive Market Hypothesis. The core objective is to disprove the strict efficient market hypothesis by demonstrating the statistically significant alpha can be extracted when the market deviates from a random walk. 

We set up three progressive models to isolate the exact source of alpha: 

**Model A (Baseline)** ──> **Model B (Regime Filtered)** ──> **Model C (V1) (Fully Optimized)**

  **Pure Price Structure** ──> **Statistical Inefficiency** ──> **Dynamic Risk & Momentum**

# The Monotonic Improvement Hypothesis

## Hypothesis: 

If the Variance Ratio (VR) contains genuine predictive information about market regimes, performance metrics will show monotonic improvement (Model A < Model B < Model C) 

**VR < 1 (Mean Reversion Dominant):** Volatility increases slower than a random walk; prices tend to return to the historical average. 

**VR > 1 (Trend Dominant):** Volatility increases faster than a random walk; prices tend to persist in the current direction. 


# Technical Formulations

## 2.1. Variance Ratio (VR) 

Measured at q = 16 (approx. 3days) and q = 30 (approx. 5 days) on 4-hour bars: 

$$VR(q) = \frac{Var[r_q]}{q \cdot Var[r_1]}$$

## 2.2 Bollinger Bands & Width (BBW)
