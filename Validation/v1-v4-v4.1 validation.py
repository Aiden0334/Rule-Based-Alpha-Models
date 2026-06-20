"""
═══════════════════════════════════════════════════════════════════
  v1, v4, v4.1 검증 
═══════════════════════════════════════════════════════════════════
<v4 → v4.1 차이>
  v4: BBW가 neutral 레짐에서도 단독 진입 가능 → 84거래 Sharpe -0.555 ✗
  v4.1: neutral BBW 단독 진입 완전 제거. strong_mom + BBW 결합만 유지.

<비교 대상>
  v1: 기준 (이전 결과 0.965, 거래 995)
  v4: 5중 분류 + mom 제거 (Sharpe 0.945, 거래 808)
  v4.1: v4에서 neutral BBW 제거 (예상 Sharpe 0.99+, 거래 720)

<기대>
  명확하게 손실 보장이던 84거래 제거
  → 합산 Sharpe 상승, MDD 감소, 평균 효율↑
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
from scipy import stats

DATA_DIR = "./futures_data"

FILES = {
    "ES": "ES_4h_continuous.csv",
    "NQ": "NQ_4h_continuous.csv",
    "YM": "YM_4h_continuous.csv",
}
PRODUCTS = list(FILES.keys())

WINDOW = 100
COST = 0.0004
ATR_LEN = 14
BB_LENGTH = 20
STOP_ATR = 4.0
MOM_LOOKBACK = 10

V1_VR_LOWER = 0.95
V1_VR_UPPER = 1.05
V1_TRAIL_ATR = 3.0
V4_TRAIL_MOM = 3.0

SHORT_Q = [2, 3, 4, 6, 8]
LONG_Q = [10, 16, 21, 25, 30]
Q_LIST = SHORT_Q + LONG_Q
ROLL_Q = 800

BBW_PERCENTILE_WINDOW = 100
EXPANSION_Q = 0.80
BAND_WALK_BARS = 5
BAND_WALK_THRESHOLD = 3
BAND_WALK_SIGMA = 1.5

ALLOC = 1.0 / 3
MAX_POS = 2
BARS_PER_YEAR = 1500


def variance_ratio(prices, q):
    log_p = np.log(prices); rets = np.diff(log_p); n = len(rets)
    if n < q + 1: return np.nan
    mu = np.mean(rets); var_1 = np.sum((rets - mu) ** 2) / n
    if var_1 == 0: return np.nan
    q_rets = log_p[q:] - log_p[:-q]
    return np.sum((q_rets - q * mu) ** 2) / (n * q) / var_1


def prep(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df["ma"] = df["close"].rolling(BB_LENGTH).mean()
    df["std"] = df["close"].rolling(BB_LENGTH).std()
    df["upper"] = df["ma"] + 2.0 * df["std"]
    df["lower"] = df["ma"] - 2.0 * df["std"]
    df["upper_walk"] = df["ma"] + BAND_WALK_SIGMA * df["std"]
    df["lower_walk"] = df["ma"] - BAND_WALK_SIGMA * df["std"]
    
    df["bbw"] = (df["upper"] - df["lower"]) / df["ma"]
    df["bbw_high_th"] = df["bbw"].rolling(BBW_PERCENTILE_WINDOW).quantile(EXPANSION_Q)
    df["is_expansion"] = df["bbw"] > df["bbw_high_th"]
    
    above = (df["close"] > df["upper_walk"]).astype(int)
    below = (df["close"] < df["lower_walk"]).astype(int)
    df["walk_up"] = above.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD
    df["walk_down"] = below.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD
    
    closes = df["close"].values; n = len(closes)
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
    
    def reg5(row):
        if pd.isna(row["q20"]): return "neutral"
        s = row["vr_score"]
        if s <= row["q20"]: return "strong_mom"
        elif s <= row["q40"]: return "mom"
        elif s <= row["q60"]: return "neutral"
        elif s <= row["q80"]: return "rev"
        else: return "strong_rev"
    df["regime5"] = df.apply(reg5, axis=1)
    
    df["mom_val"] = df["close"] - df["close"].shift(MOM_LOOKBACK)
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_LEN).mean()
    
    critical = ["ma", "upper", "lower", "atr", "vr_score", "q20",
                "q40", "q60", "q80", "regime5", "mom_val", "vr16", "vr30"]
    return df.dropna(subset=critical).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# v1 (기준)
# ═══════════════════════════════════════════════════════════════════
def model_v1(df):
    pos, ep, sp, trail, m_pos, src = 0, 0.0, None, None, None, None
    trades = []
    
    for row in df.to_dict("records"):
        c = row["close"]; u = row["upper"]; l = row["lower"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        
        vr_rev = row["vr16"] < V1_VR_LOWER and row["vr30"] < V1_VR_LOWER
        vr_tr = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
        bbw_long = row["is_expansion"] and row["walk_up"]
        bbw_short = row["is_expansion"] and row["walk_down"]
        
        if pos == 0:
            if vr_rev:
                if c < l: pos, ep, sp, m_pos, src = 1, c, c - STOP_ATR * atr, "rev", "vr_rev"
                elif c > u: pos, ep, sp, m_pos, src = -1, c, c + STOP_ATR * atr, "rev", "vr_rev"
            elif vr_tr:
                if mom != 0:
                    d = 1 if mom > 0 else -1
                    pos, ep, m_pos, trail, src = d, c, "trend", c, "vr_trend"
                    sp = c - d * STOP_ATR * atr
            elif bbw_long:
                pos, ep, m_pos, trail, src = 1, c, "trend", c, "bbw_long"
                sp = c - STOP_ATR * atr
            elif bbw_short:
                pos, ep, m_pos, trail, src = -1, c, "trend", c, "bbw_short"
                sp = c + STOP_ATR * atr
            continue
        
        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                vr_tr_now = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
                still = vr_tr_now or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V1_TRAIL_ATR * atr or not still: raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                vr_tr_now = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
                still = vr_tr_now or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V1_TRAIL_ATR * atr or not still: raw = (ep - c) / ep
        
        if raw is not None:
            trades.append({"ret": raw - COST, "src": src, "exit_dt": dt})
            pos = 0
    return trades


# ═══════════════════════════════════════════════════════════════════
# v4 (이전 - neutral BBW 진입 포함)
# ═══════════════════════════════════════════════════════════════════
def model_v4(df):
    pos, ep, sp, trail, m_pos, entry_reg, src = 0, 0.0, None, None, None, None, None
    trades = []
    
    for row in df.to_dict("records"):
        c = row["close"]; u = row["upper"]; l = row["lower"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        reg = row["regime5"]
        
        if pos == 0:
            entered = False
            
            if reg in ("strong_rev", "rev"):
                if c < l:
                    pos, ep, sp, m_pos, entry_reg, src = 1, c, c - STOP_ATR * atr, "rev", reg, f"vr_{reg}"
                    entered = True
                elif c > u:
                    pos, ep, sp, m_pos, entry_reg, src = -1, c, c + STOP_ATR * atr, "rev", reg, f"vr_{reg}"
                    entered = True
            elif reg == "strong_mom" and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg, src = 1, c, "trend", c, reg, "vr_strong_mom_bbw"
                    sp = c - STOP_ATR * atr; entered = True
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg, src = -1, c, "trend", c, reg, "vr_strong_mom_bbw"
                    sp = c + STOP_ATR * atr; entered = True
            
            # BBW 단독 (neutral/mom에서 진입 가능 - 이게 v4의 문제)
            if not entered and reg not in ("strong_rev", "rev", "strong_mom") and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg, src = 1, c, "trend", c, reg, "bbw_long"
                    sp = c - STOP_ATR * atr
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg, src = -1, c, "trend", c, reg, "bbw_short"
                    sp = c + STOP_ATR * atr
            continue
        
        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V4_TRAIL_MOM * atr or not still: raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V4_TRAIL_MOM * atr or not still: raw = (ep - c) / ep
        
        if raw is not None:
            trades.append({"ret": raw - COST, "src": src, "entry_reg": entry_reg, "exit_dt": dt})
            pos = 0; entry_reg = None
    return trades


# ═══════════════════════════════════════════════════════════════════
# v4.1 (수정 - neutral/mom BBW 단독 진입 제거)
# ═══════════════════════════════════════════════════════════════════
def model_v4_1(df):
    pos, ep, sp, trail, m_pos, entry_reg, src = 0, 0.0, None, None, None, None, None
    trades = []
    
    for row in df.to_dict("records"):
        c = row["close"]; u = row["upper"]; l = row["lower"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        reg = row["regime5"]
        
        if pos == 0:
            # 회귀 (strong_rev + rev)
            if reg in ("strong_rev", "rev"):
                if c < l:
                    pos, ep, sp, m_pos, entry_reg, src = 1, c, c - STOP_ATR * atr, "rev", reg, f"vr_{reg}"
                elif c > u:
                    pos, ep, sp, m_pos, entry_reg, src = -1, c, c + STOP_ATR * atr, "rev", reg, f"vr_{reg}"
            # strong_mom + BBW 결합만 (기존 유지)
            elif reg == "strong_mom" and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg, src = 1, c, "trend", c, reg, "vr_strong_mom_bbw"
                    sp = c - STOP_ATR * atr
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg, src = -1, c, "trend", c, reg, "vr_strong_mom_bbw"
                    sp = c + STOP_ATR * atr
            # ★ v4.1 핵심 변경: BBW 단독 진입 제거 (neutral/mom 진입 없음)
            continue
        
        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V4_TRAIL_MOM * atr or not still: raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V4_TRAIL_MOM * atr or not still: raw = (ep - c) / ep
        
        if raw is not None:
            trades.append({"ret": raw - COST, "src": src, "entry_reg": entry_reg, "exit_dt": dt})
            pos = 0; entry_reg = None
    return trades


# ═══════════════════════════════════════════════════════════════════
# 포트폴리오 (간단 버전)
# ═══════════════════════════════════════════════════════════════════
def portfolio_generic(data, model_fn):
    combined = []
    for p in data:
        trs = model_fn(data[p])
        for t in trs:
            combined.append({"product": p, "ret": t["ret"], "exit_dt": t["exit_dt"]})
    if not combined: return pd.Series(dtype=float)
    combined.sort(key=lambda x: x["exit_dt"])
    df_t = pd.DataFrame(combined)
    df_t["ret_alloc"] = df_t["ret"] * ALLOC
    return df_t.groupby("exit_dt")["ret_alloc"].sum()


def stats_of(trades_or_rets, years):
    if isinstance(trades_or_rets, list) and len(trades_or_rets) > 0 and isinstance(trades_or_rets[0], dict):
        rets = np.array([t["ret"] for t in trades_or_rets])
    else:
        rets = np.array(trades_or_rets) if not isinstance(trades_or_rets, np.ndarray) else trades_or_rets
    if len(rets) == 0: return dict(n=0, wr=0, cagr=0, pf=0, sharpe=0, mdd=0, total=0)
    n = len(rets)
    total = np.prod(1 + rets) - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1
    gp = rets[rets > 0].sum(); gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else np.inf
    sharpe = (rets.mean() / rets.std() * np.sqrt(n / years)) if rets.std() > 0 else 0
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    wr = (rets > 0).mean() * 100
    return dict(n=n, wr=wr, cagr=cagr * 100, pf=pf, sharpe=sharpe, mdd=mdd, total=total * 100)


def port_stats(rs, years):
    if len(rs) == 0: return dict(cagr=0, sharpe=0, mdd=0)
    r = rs.values
    total = np.prod(1 + r) - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1
    sharpe = (r.mean() / r.std() * np.sqrt(len(r) / years)) if r.std() > 0 else 0
    eq = np.cumprod(1 + r); peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    return dict(cagr=cagr * 100, sharpe=sharpe, mdd=mdd)


def main():
    print("=" * 92)
    print(" v1 vs v4 vs v4.1 비교 — neutral BBW 제거 효과 검증")
    print("=" * 92)
    
    data = {}
    years_ref = 5.0
    for p, fn in FILES.items():
        path = os.path.join(DATA_DIR, fn)
        df = pd.read_csv(path, parse_dates=["datetime"])
        data[p] = prep(df)
        years_ref = (data[p]["datetime"].max() - data[p]["datetime"].min()).days / 365.25
    
    print("\n[데이터]")
    for p in PRODUCTS:
        d = data[p]
        print(f"  {p}: {len(d):>6}봉, {d['datetime'].iloc[0].strftime('%Y-%m-%d')} ~ {d['datetime'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # 세 모델 실행
    results = {}
    for name, model_fn in [("v1", model_v1), ("v4", model_v4), ("v4.1", model_v4_1)]:
        all_tr = []
        for p in PRODUCTS:
            all_tr.extend(model_fn(data[p]))
        results[name] = all_tr
    
    # [1] 합산
    print("\n[1] 합산 비교")
    print(f"  {'모델':<6} {'거래':>5} {'CAGR':>7} {'PF':>5} {'Sharpe':>7} {'MDD':>7} {'승률':>6} {'t-stat':>7} {'p':>7}")
    print("  " + "─" * 72)
    for name in ["v1", "v4", "v4.1"]:
        s = stats_of(results[name], years_ref)
        rets = np.array([t["ret"] for t in results[name]])
        t_stat, p_val = stats.ttest_1samp(rets, 0)
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        print(f"  {name:<6} {s['n']:>5} {s['cagr']:>6.2f}% {s['pf']:>5.2f} "
              f"{s['sharpe']:>7.3f} {s['mdd']:>6.1f}% {s['wr']:>5.1f}% "
              f"{t_stat:>7.3f} {p_one:>7.4f}")
    
    # [2] 종목별
    print("\n[2] 종목별 Sharpe")
    print(f"  {'모델':<6}", end="")
    for p in PRODUCTS: print(f"{p:>10}", end="")
    print()
    print("  " + "─" * 40)
    for name, fn in [("v1", model_v1), ("v4", model_v4), ("v4.1", model_v4_1)]:
        print(f"  {name:<6}", end="")
        for p in PRODUCTS:
            s = stats_of(fn(data[p]), years_ref)
            print(f"{s['sharpe']:>10.3f}", end="")
        print()
    
    # [3] v4.1 레짐별
    print("\n[3] v4.1 레짐별 거래 성과")
    print(f"  {'레짐':<14} {'거래':>5} {'CAGR':>7} {'Sharpe':>7} {'합산':>8}")
    print("  " + "─" * 48)
    v41 = results["v4.1"]
    for reg in ["strong_rev", "rev", "strong_mom"]:
        sub = [t for t in v41 if t.get("entry_reg") == reg]
        if not sub: continue
        s = stats_of(sub, years_ref)
        print(f"  {reg:<14} {s['n']:>5} {s['cagr']:>6.2f}% {s['sharpe']:>7.3f} {s['total']:>+7.1f}%")
    
    # [4] 연도별
    print("\n[4] 연도별 수익률")
    print(f"  {'연도':>6}", end="")
    for name in ["v1", "v4", "v4.1"]:
        print(f"  {name+'_ret':>10} {name+'_n':>6}", end="")
    print()
    print("  " + "─" * 70)
    yearly = {name: {} for name in ["v1", "v4", "v4.1"]}
    for name in ["v1", "v4", "v4.1"]:
        df_t = pd.DataFrame(results[name])
        df_t["year"] = df_t["exit_dt"].dt.year
        for y in sorted(df_t["year"].unique()):
            yt = df_t[df_t["year"] == y]
            yearly[name][y] = {"ret": (np.prod(1 + yt["ret"]) - 1) * 100, "n": len(yt)}
    all_years = sorted({y for name in ["v1", "v4", "v4.1"] for y in yearly[name]})
    pos_counts = {n: 0 for n in ["v1", "v4", "v4.1"]}
    for y in all_years:
        print(f"  {y:>6}", end="")
        for name in ["v1", "v4", "v4.1"]:
            if y in yearly[name]:
                d = yearly[name][y]
                print(f"  {d['ret']:>+9.2f}% {d['n']:>6}", end="")
                if d['ret'] > 0: pos_counts[name] += 1
            else:
                print(f"  {'-':>10} {'-':>6}", end="")
        print()
    print(f"\n  수익 연도:  v1: {pos_counts['v1']}/{len(all_years)},  "
          f"v4: {pos_counts['v4']}/{len(all_years)},  v4.1: {pos_counts['v4.1']}/{len(all_years)}")
    
    # [5] 포트폴리오
    print("\n[5] 포트폴리오 비교")
    print(f"  {'모델':<6} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7}")
    print("  " + "─" * 36)
    for name, fn in [("v1", model_v1), ("v4", model_v4), ("v4.1", model_v4_1)]:
        rs = portfolio_generic(data, fn)
        ps = port_stats(rs, years_ref)
        print(f"  {name:<6} {ps['cagr']:>6.2f}% {ps['sharpe']:>7.3f} {ps['mdd']:>6.1f}%")
    
    print("\n" + "=" * 92)
    print("[판단 기준]")
    print("=" * 92)
    print("  v4.1 Sharpe > v4 Sharpe → neutral BBW 제거 효과 입증, v4.1 채택")
    print("  v4.1 MDD < v4 MDD → 위험 통제 개선")
    print("  v4.1 수익 연도 >= v4 → 안정성 유지")
    print("  포트폴리오 Sharpe 비교 → 실전 운용 관점에서 어느 게 최강인지")


if __name__ == "__main__":
    main()
