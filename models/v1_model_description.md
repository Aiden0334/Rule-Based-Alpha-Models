# V1 Rule-Based Alpha Model

V1 is a general-purpose alpha model that combined mean-reversion and momentum signals on U.S. equity index futures. It employs a binary regime classifier based on the Variance Ratio (VR) to distinguish between mean-revertion and trending market states. Bollinger Band Width expansion and ATR are used as volatility filters to confirm signal validity, while ATR-based stop loss and trailing exit handle risk management. The model was verified across four U.S. equity index futures markets (ES, NQ, YM, RTY) on 4-hour bars.

## Model Specification

| Item | Value |
|------|-------|
| Regime | Binary Regime Classifier (Variance Ratio based) |
| Alpha Logic | Mean Reversion, Momentum |
| Volatility Filter | ATR, Bollinger Band Width |
| Risk Management | ATR-based Stop Loss, Trailing Exit |
| Timeframe | 4-hour bars |
| Markets | E-mini S&P 500 (ES), E-mini NASDAQ-100 (NQ), E-mini Dow Jones (YM), E-mini Russell 2000 (RTY) |

## Verified Performance

| Metric | In-Sample | Out-of-Sample |
|--------|-----------|---------------|
| Period | 2018-2023 (5.93 years) | 2024-2026 (2.25 years) |
| Trades | 700 | 295 |
| Sharpe | 0.963 | 0.973 |
| CAGR | +24.09% | +22.56% |
| MDD | -16.5% | -22.7% |

### Robustness

| Test | Result |
|------|--------|
| IS → OOS Sharpe Change | -0.01 (not overfitting) |
| Rolling Origin (4 splits) | Mean 0.963, Std 0.137, all positive |
| Annual Consistency | 9/9 positive years (2018-2026) |
| Statistical Significance | t = 2.76, p = 0.003 |

## Limitations

The alpha is specialized for U.S. equity index futures and does not generalize to other asset classes such as commodities (crude oil, gold) or currencies, where the underlying market mechanisms differ fundamentally. The model operates exclusively on 4-hour bars; application to other timeframes (daily, hourly) would require separate verification and parameter recalibration. The verified out-of-sample period spans 2.25 years, which, while sufficient for statistical significance, may not capture all possible market regimes.
