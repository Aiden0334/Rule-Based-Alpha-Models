"""
final_model.py
═══════════════════════════════════════════════════════════════════
  Phase 1 — Rule-Based Alpha Model (V1 + V4 Integrated)
═══════════════════════════════════════════════════════════════════
[Models]
  V1 (General Type, low-medium volatility markets):
    - Binary VR regime classifier
    - Mean Reversion + Momentum
    - Applied to: ES, NQ, YM

  V4 (Selective Type, high-volatility markets):
    - 5-state VR regime classifier
    - Selective entry on strong signals
    - Applied to: RTY

[Backtest]
  Markets:   ES, NQ, YM (V1), RTY (V4)
  Timeframe: 4-hour bars
  Period:    2018-01-24 ~ 2026-04-03
  IS:        2018-2023
  OOS:       2024-2026
  Cost:      0.04% round-trip
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_DIR = "./futures_data"

MARKET_MODEL = {
    "ES":  "v1",   # Low volatility
    "NQ":  "v1",   # Medium volatility
    "YM":  "v1",   # Low volatility
    "RTY": "v4",   # High volatility
}

FILES = {p: f"{p}_4h_continuous.csv" for p in MARKET_MODEL.keys()}

IS_START  = pd.Timestamp("2018-01-24")
IS_END    = pd.Timestamp("2023-12-31 23:59:59")
OOS_START = pd.Timestamp("2024-01-01")
OOS_END   = pd.Timestamp("2026-04-03 23:59:59")

COST = 0.0004  # 0.04% round-trip

WINDOW       = 100
ATR_LEN      = 14
BB_LENGTH    = 20
MOM_LOOKBACK = 10

V1_VR_LOWER  = 0.95
V1_VR_UPPER  = 1.05
V1_TRAIL_ATR = 3.0

V4_TRAIL_MOM = 3.0
ROLL_Q       = 800

STOP_ATR = 4.0

SHORT_Q = [2, 3, 4, 6, 8]
LONG_Q  = [10, 16, 21, 25, 30]
Q_LIST  = SHORT_Q + LONG_Q

BBW_PERCENTILE_WINDOW = 100
EXPANSION_Q           = 0.80
BAND_WALK_BARS        = 5
BAND_WALK_THRESHOLD   = 3
BAND_WALK_SIGMA       = 1.5


def variance_ratio(prices, q):
    log_p = np.log(prices)
    rets = np.diff(log_p)
    n = len(rets)
    if n < q + 1: return np.nan
    mu = np.mean(rets)
    var_1 = np.sum((rets - mu) ** 2) / n
    if var_1 == 0: return np.nan
    q_rets = log_p[q:] - log_p[:-q]
    return np.sum((q_rets - q * mu) ** 2) / (n * q) / var_1


def prep(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["ma"] = df["close"].rolling(BB_LENGTH).mean()
    df["std"] = df["close"].rolling(BB_LENGTH).std()
    df["upper_2"] = df["ma"] + 2.0 * df["std"]
    df["lower_2"] = df["ma"] - 2.0 * df["std"]
    df["upper_walk"] = df["ma"] + BAND_WALK_SIGMA * df["std"]
    df["lower_walk"] = df["ma"] - BAND_WALK_SIGMA * df["std"]
    df["bbw"] = (df["upper_2"] - df["lower_2"]) / df["ma"]
    df["bbw_high_th"] = df["bbw"].rolling(BBW_PERCENTILE_WINDOW).quantile(EXPANSION_Q)
    df["is_expansion"] = (df["bbw"] > df["bbw_high_th"]).astype(int)
    above = (df["close"] > df["upper_walk"]).astype(int)
    below = (df["close"] < df["lower_walk"]).astype(int)
    df["walk_up"] = (above.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD).astype(int)
    df["walk_down"] = (below.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD).astype(int)

    closes = df["close"].values
    n = len(closes)
    for q in Q_LIST:
        arr = np.full(n, np.nan)
        for i in range(WINDOW, n):
            arr[i] = variance_ratio(closes[i - WINDOW:i], q)
        df[f"vr{q}"] = arr

    short_cols = [f"vr{q}" for q in SHORT_Q]
    long_cols = [f"vr{q}" for q in LONG_Q]
    df["short_vr"] = df[short_cols].mean(axis=1)
    df["long_vr"] = df[long_cols].mean(axis=1)
    df["vr_score"] = 0.8 * (1 - df["long_vr"]) + 0.2 * (1 - df["short_vr"])

    df["q20"] = df["vr_score"].rolling(ROLL_Q).quantile(0.20)
    df["q40"] = df["vr_score"].rolling(ROLL_Q).quantile(0.40)
    df["q60"] = df["vr_score"].rolling(ROLL_Q).quantile(0.60)
    df["q80"] = df["vr_score"].rolling(ROLL_Q).quantile(0.80)

    def classify_regime(row):
        if pd.isna(row["q20"]): return "neutral"
        s = row["vr_score"]
        if s <= row["q20"]: return "strong_mom"
        if s <= row["q40"]: return "mom"
        if s <= row["q60"]: return "neutral"
        if s <= row["q80"]: return "rev"
        return "strong_rev"

    df["regime5"] = df.apply(classify_regime, axis=1)
    df["mom_val"] = df["close"] - df["close"].shift(MOM_LOOKBACK)

    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()

    critical = ["ma", "upper_2", "lower_2", "atr", "vr_score", "q20",
                "q40", "q60", "q80", "regime5", "mom_val", "vr16", "vr30"]
    return df.dropna(subset=critical).reset_index(drop=True)


def model_v1(df, product):
    """V1: Binary VR regime → MR + Momentum"""
    pos, ep, sp, trail, m_pos = 0, 0.0, None, None, None
    trades = []
    entry_dt = None

    for row in df.to_dict("records"):
        c = row["close"]; u = row["upper_2"]; l = row["lower_2"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]

        vr_rev   = row["vr16"] < V1_VR_LOWER and row["vr30"] < V1_VR_LOWER
        vr_trend = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
        bbw_long  = row["is_expansion"] and row["walk_up"]
        bbw_short = row["is_expansion"] and row["walk_down"]

        if pos == 0:
            direction = 0; mode = None
            if vr_rev:
                if c < l:     direction, mode = 1, "rev"
                elif c > u:   direction, mode = -1, "rev"
            elif vr_trend and mom != 0:
                direction = 1 if mom > 0 else -1
                mode = "trend"
            elif bbw_long:  direction, mode = 1, "trend"
            elif bbw_short: direction, mode = -1, "trend"

            if direction != 0:
                pos = direction; ep = c
                sp = c - direction * STOP_ATR * atr
                m_pos = mode
                if mode == "trend": trail = c
                entry_dt = dt
            continue

        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                vr_trend_now = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
                still = vr_trend_now or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V1_TRAIL_ATR * atr or not still:
                    raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                vr_trend_now = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
                still = vr_trend_now or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V1_TRAIL_ATR * atr or not still:
                    raw = (ep - c) / ep

        if raw is not None:
            trades.append({"entry_dt": entry_dt, "exit_dt": dt,
                "direction": pos, "trade_return": raw - COST,
                "product": product, "model": "v1"})
            pos = 0

    return trades


def model_v4(df, product):
    """V4: 5-regime VR → Selective MR + Momentum"""
    pos, ep, sp, trail, m_pos = 0, 0.0, None, None, None
    trades = []
    entry_dt = None

    for row in df.to_dict("records"):
        c = row["close"]; u = row["upper_2"]; l = row["lower_2"]
        atr = row["atr"]; dt = row["datetime"]
        reg = row["regime5"]

        if pos == 0:
            direction = 0; mode = None
            if reg in ("strong_rev", "rev"):
                if c < l:     direction, mode = 1, "rev"
                elif c > u:   direction, mode = -1, "rev"
            elif reg == "strong_mom" and row["is_expansion"]:
                if row["walk_up"]:     direction, mode = 1, "trend"
                elif row["walk_down"]: direction, mode = -1, "trend"
            elif reg not in ("strong_rev", "rev", "strong_mom") and row["is_expansion"]:
                if row["walk_up"]:     direction, mode = 1, "trend"
                elif row["walk_down"]: direction, mode = -1, "trend"

            if direction != 0:
                pos = direction; ep = c
                sp = c - direction * STOP_ATR * atr
                m_pos = mode
                if mode == "trend": trail = c
                entry_dt = dt
            continue

        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                still = (row["regime5"] == "strong_mom") or \
                        (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V4_TRAIL_MOM * atr or not still:
                    raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                still = (row["regime5"] == "strong_mom") or \
                        (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V4_TRAIL_MOM * atr or not still:
                    raw = (ep - c) / ep

        if raw is not None:
            trades.append({"entry_dt": entry_dt, "exit_dt": dt,
                "direction": pos, "trade_return": raw - COST,
                "product": product, "model": "v4"})
            pos = 0

    return trades


def stats(trades):
    if len(trades) == 0:
        return dict(n=0, sharpe=0, mdd=0, total=0, cagr=0, win_rate=0)
    df = pd.DataFrame(trades)
    rets = df["trade_return"].values
    n = len(rets)
    years = (df["exit_dt"].max() - df["entry_dt"].min()).days / 365.25
    years = max(years, 0.01)
    sharpe = (rets.mean() / rets.std() * np.sqrt(n / years)) if rets.std() > 0 else 0
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    total = (np.prod(1 + rets) - 1) * 100
    cagr = ((1 + total / 100) ** (1 / years) - 1) * 100 if total > -100 else -100
    win_rate = (rets > 0).mean() * 100
    return dict(n=n, sharpe=sharpe, mdd=mdd, total=total, cagr=cagr,
                win_rate=win_rate, years=years)


def split_is_oos(trades):
    is_t  = [t for t in trades if IS_START  <= t["entry_dt"] <= IS_END]
    oos_t = [t for t in trades if OOS_START <= t["entry_dt"] <= OOS_END]
    return is_t, oos_t


def simulate_portfolio(trades_list, n_markets):
    if not trades_list:
        return dict(n=0, sharpe=0, mdd=0, total=0, cagr=0)
    cap_per_market = 100 / n_markets
    product_caps = {p: cap_per_market for p in MARKET_MODEL.keys()}
    trades_list = sorted(trades_list, key=lambda x: x["exit_dt"])
    port_rets = []
    port_vals = [sum(product_caps.values())]
    for t in trades_list:
        p = t["product"]
        if p not in product_caps: continue
        pb = sum(product_caps.values())
        product_caps[p] *= (1 + t["trade_return"])
        pa = sum(product_caps.values())
        port_rets.append((pa - pb) / pb)
        port_vals.append(pa)
    rets = np.array(port_rets); n = len(rets)
    years = (trades_list[-1]["exit_dt"] - trades_list[0]["entry_dt"]).days / 365.25
    years = max(years, 0.01)
    sharpe = (rets.mean() / rets.std() * np.sqrt(n / years)) if rets.std() > 0 else 0
    pv = np.array(port_vals); peak = np.maximum.accumulate(pv)
    mdd = ((pv - peak) / peak).min() * 100
    total = (pv[-1] - pv[0]) / pv[0] * 100
    cagr = ((pv[-1] / pv[0]) ** (1 / years) - 1) * 100 if pv[-1] > 0 else -100
    return dict(n=n, sharpe=sharpe, mdd=mdd, total=total, cagr=cagr)


def main():
    print("=" * 90)
    print(" Phase 1 — Rule-Based Alpha Model (V1 + V4 Integrated)")
    print("=" * 90)
    print(f"\n  Data Directory: {DATA_DIR}")
    print(f"  IS Period:      {IS_START.date()} ~ {IS_END.date()}")
    print(f"  OOS Period:     {OOS_START.date()} ~ {OOS_END.date()}")
    print(f"\n  Market -> Model:")
    for market, model in MARKET_MODEL.items():
        print(f"    {market:>4} -> {model}")

    all_trades = []

    for product, model_name in MARKET_MODEL.items():
        path = os.path.join(DATA_DIR, FILES[product])
        if not os.path.exists(path):
            print(f"\n  [{product}] Skipped - {path} not found")
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        data = prep(df)
        if model_name == "v1":
            trades = model_v1(data, product)
        elif model_name == "v4":
            trades = model_v4(data, product)
        else: continue
        all_trades.extend(trades)
        print(f"\n  [{product}] Model {model_name}: {len(trades)} trades")

    print("\n" + "=" * 90)
    print(" Per-Market Performance (IS / OOS)")
    print("=" * 90)
    print(f"\n  {'Market':<7} {'Period':<6} {'Trades':>7} {'Win%':>6} "
          f"{'Sharpe':>8} {'CAGR':>9} {'MDD':>8}")
    print("  " + "-" * 70)

    for product in MARKET_MODEL.keys():
        market_trades = [t for t in all_trades if t["product"] == product]
        if not market_trades: continue
        is_t, oos_t = split_is_oos(market_trades)
        s_is = stats(is_t); s_oos = stats(oos_t)
        print(f"  {product:<7} {'IS':<6} {s_is['n']:>7} {s_is['win_rate']:>5.1f}% "
              f"{s_is['sharpe']:>+7.3f} {s_is['cagr']:>+8.2f}% {s_is['mdd']:>+7.1f}%")
        print(f"  {product:<7} {'OOS':<6} {s_oos['n']:>7} {s_oos['win_rate']:>5.1f}% "
              f"{s_oos['sharpe']:>+7.3f} {s_oos['cagr']:>+8.2f}% {s_oos['mdd']:>+7.1f}%")
        print()

    print("=" * 90)
    print(" Portfolio Performance (Equal Capital Allocation)")
    print("=" * 90)

    is_all, oos_all = split_is_oos(all_trades)
    p_is = simulate_portfolio(is_all, len(MARKET_MODEL))
    p_oos = simulate_portfolio(oos_all, len(MARKET_MODEL))

    print(f"\n  {'Period':<6} {'Events':>7} {'Sharpe':>8} {'CAGR':>9} {'MDD':>8} {'Return':>10}")
    print("  " + "-" * 60)
    print(f"  {'IS':<6} {p_is['n']:>7} {p_is['sharpe']:>+7.3f} "
          f"{p_is['cagr']:>+8.2f}% {p_is['mdd']:>+7.1f}% {p_is['total']:>+9.2f}%")
    print(f"  {'OOS':<6} {p_oos['n']:>7} {p_oos['sharpe']:>+7.3f} "
          f"{p_oos['cagr']:>+8.2f}% {p_oos['mdd']:>+7.1f}% {p_oos['total']:>+9.2f}%")

    print(f"\n  IS -> OOS Sharpe Change: {p_oos['sharpe'] - p_is['sharpe']:+.3f}")


if __name__ == "__main__":
    main()
