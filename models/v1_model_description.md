## V1 Rule-Based Alpha Model

# Regime


Binary Regime Classifier
Variance Ratio based


# Alpha Logic


Mean Reversion
Momentum


# Volatility Filter


ATR
Bollinger Band Width Expansion


# Risk Management


ATR-based Stop Loss
Trailing Exit


# Timeframe


4-hour bars


# Markets


ES (E-mini S&P 500)
NQ (E-mini NASDAQ-100)
YM (E-mini Dow Jones)
RTY (E-mini Russell 2000)


# Verified Performance

MetricIn-Sample (2018-2023)Out-of-Sample (2024-2026)Sharpe0.9630.973CAGR24.09%22.56%MDD-16.5%-22.7%


IS → OOS Sharpe change: -0.01 (no overfitting)
Rolling Origin (4 splits): mean 0.963, std 0.137, all positive
Statistical significance: t = 2.76, p = 0.003


# Limitations


Asset class specialization (U.S. equity index futures only)
Does not generalize to commodities or other asset classes
4-hour timeframe specific
