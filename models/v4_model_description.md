# V4 Rule-Based Alpha Model

V4 is a selective alpha model designed for high-volatility markets where false signals are frequent. It uses a 5-state regime classifier based on the Variance Ratio (VR) score quantiles, allowing finer differentiation between strong and weak trend / mean-reversion conditions. V4 enters only on confirmed signals — strong reversion regimes combined with Bollinger Band breakouts, or strong momentum regimes combined with Bollinger Band Width expansion and band-walking confirmation. ATR-based stop loss and trailing exit handle risk management. V4 complements V1 by being more conservative: fewer trades, but with higher signal quality, suitable for markets where signal noise would otherwise erode V1's edge.

## Model Specification

| Item | Value |
|------|-------|
| Regime | 5-State Regime Classifier (Variance Ratio based) |
| Alpha Models | Mean Reversion, Momentum (selective entry on strong signals) |
| Expansion Detection | Bollinger Band Width |
| Risk Management | ATR-based Stop Loss, Trailing Exit |
| Timeframe | 4-hour bars |
| Markets | E-mini S&P 500 (ES), E-mini NASDAQ-100 (NQ), E-mini Dow Jones (YM), E-mini Russell 2000 (RTY) |

## Verified Performance

| Metric | In-Sample | Out-of-Sample |
|--------|-----------|---------------|
| Period | 2018-2023 (5.93 years) | 2024-2026 (2.25 years) |
| Sharpe | 0.943 | 0.989 |

### Robustness

| Test | Result |
|------|--------|
| IS → OOS Sharpe Change | +0.046 (not overfitting) |
| Verified Market | RTY (E-mini Russell 2000) |
| Design Purpose | Selective entry for high-volatility markets |
| Complementarity | Pairs with V1 (general) for full coverage |

## Limitations

V4 shares the asset class constraints of V1, the alpha is specialized for U.S. equity index futures and does not generalize to other asset classes. (e.g., commodities or currencies) The model operates exclusively on 4-hour bars; application to other time-frames would require separate verification and parameter recalibration. V4's lower trade frequency (compared to V1) requires longer verification periods for statistical confidence, and its strict entry conditions may underperform during low-volatility regimes where V1's general approach is more effective.
