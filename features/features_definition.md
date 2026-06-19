# Feature Definitions

Conceptual definitions of indicators used in V1 and V4 rule-based alpha models.

## Variance Ratio (VR)

A statistical measure of whether returns deviate from a random walk.

**Formula:**

$$VR(q) = \frac{\text{Var}[r_q]}{q \cdot \text{Var}[r_1]}$$

where $r_q$ is the cumulative return over $q$ periods.

**Interpretation:**

| VR Value | Interpretation |
|----------|----------------|
| VR < 1 | Mean-reversion dominant (negative autocorrelation) |
| VR ≈ 1 | Random walk (Efficient Market) |
| VR > 1 | Trend dominant (positive autocorrelation) |

**Origin:** Lo & MacKinlay (1988) — original variance ratio test for random walk hypothesis.

## Bollinger Bands

Volatility bands plotted relative to a moving average.

**Components:**
- Middle Band: Simple Moving Average
- Upper Band: MA + k · σ
- Lower Band: MA − k · σ

where σ is the rolling standard deviation.

**Origin:** Bollinger (2002) — *Bollinger on Bollinger Bands*

## Bollinger Band Width (BBW)

A measure of volatility expansion / contraction.

**Formula:**

$$BBW = \frac{\text{Upper Band} - \text{Lower Band}}{\text{Middle Band}}$$

**Expansion Detection:** BBW above a rolling percentile threshold indicates
significant volatility increase, often associated with directional moves.

## Band Walking

Sustained price movement along an outer band, indicating strong directional momentum.

**Concept:** Multiple consecutive bars close beyond a band (e.g., +1.5σ) within
a rolling window signals trend conviction.

## Average True Range (ATR)

A volatility measure based on the maximum daily range.

**True Range:**

$$TR = \max(H - L, |H - C_{prev}|, |L - C_{prev}|)$$

**ATR (smoothed average over N periods):**

$$ATR(N) = \frac{1}{N} \sum_{i=1}^{N} TR_i$$

**Use Cases:**
- Stop-loss sizing (ATR × multiplier)
- Trailing stop calibration
- Volatility-normalized features

**Origin:** Wilder (1978) — *New Concepts in Technical Trading Systems*

## Feature Usage Summary

| Indicator | Used For |
|-----------|----------|
| Variance Ratio | Regime classification (mean-reversion vs trend) |
| Bollinger Bands | Entry signal (band breakout) |
| Bollinger Band Width | Volatility regime, momentum confirmation |
| Band Walking | Trend continuation confirmation |
| ATR | Stop-loss, trailing exit, volatility normalization |

## References

1. Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *Review of Financial Studies*, 1(1), 41–66.

2. Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.

3. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
