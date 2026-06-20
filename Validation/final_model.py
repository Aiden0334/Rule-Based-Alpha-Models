"""
═══════════════════════════════════════════════════════════════════
  v1 vs v3 vs v4 동시 비교
═══════════════════════════════════════════════════════════════════
<목적>
  같은 데이터에 세 모델 돌려서 직접 비교 -> 두 모델만 채택.
  
<모델>
  v1 (기준):
    - VR(16,30) consensus 이진
    - 회귀: BB ±2σ
    - 모멘텀: VR 추세 OR BBW (결합)
    - 추적 ATR×3, 손절 ATR×4
  
  v3 (차별화 뒤집기):
    - 5중 레짐
    - strong_rev: BB ±2.5σ (더 빡센)
    - rev: BB ±2σ
    - strong_mom: 추적 ATR×2.5 (빨리 익절)
    - mom: 추적 ATR×3
    - BBW: v1 방식 (VR 추세 없을 때만)
    -> withdrew. 
  
  v4 (단순화 + mom 제거):
    - 5중 레짐 (회귀 분류만 활용)
    - strong_rev = rev = BB ±2σ (회귀 통일)
    - mom 진입 제거
    - strong_mom: BBW 결합 필수
    - BBW: VR 추세 없을 때만

<ROLL_Q>
  비교를 위해 800 고정 

<비교 성과 지표>
  합산 Sharpe / CAGR / MDD
  포트폴리오 Sharpe / MDD
  연도별 수익 일관성
  레짐별 거래 분포

<결과>
  모델 v1 & 모델 v4 채택! 
  but, model v1 is for low-volatility market and model v4 fit for high-volatility market. 
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
    "YM": "YM_4h_continuous.csv"
}
PRODUCTS = list(FILES.keys())

# 공통 파라미터
WINDOW = 100
COST = 0.0004
ATR_LEN = 14
BB_LENGTH = 20
STOP_ATR = 4.0
MOM_LOOKBACK = 10

# v1 파라미터
V1_VR_LOWER = 0.95
V1_VR_UPPER = 1.05
V1_BB_MULT = 2.0
V1_TRAIL_ATR = 3.0

# v3 파라미터 (차별화 뒤집기)
V3_BB_STRONG_REV = 2.5      # 더 빡센 진입
V3_BB_REV = 2.0
V3_TRAIL_STRONG_MOM = 2.5   # 빨리 익절
V3_TRAIL_MOM = 3.0

# v4 파라미터 (단순화)
V4_BB_REV = 2.0             # 회귀 통일
V4_TRAIL_MOM = 3.0          # strong_mom용 (mom 제거)

# 5중 레짐 (v3, v4 공통)
SHORT_Q = [2, 3, 4, 6, 8]
LONG_Q = [10, 16, 21, 25, 30]
Q_LIST = SHORT_Q + LONG_Q
ROLL_Q = 800

# BBW (모든 모델 공통)
BBW_PERCENTILE_WINDOW = 100
EXPANSION_Q = 0.80
BAND_WALK_BARS = 5
BAND_WALK_THRESHOLD = 3
BAND_WALK_SIGMA = 1.5

ALLOC = 1.0 / 3
MAX_POS = 2
BARS_PER_YEAR = 1500


# ═══════════════════════════════════════════════════════════════════
# 공통 함수
# ═══════════════════════════════════════════════════════════════════
def variance_ratio(prices, q):
    log_p = np.log(prices); rets = np.diff(log_p); n = len(rets)
    if n < q + 1: return np.nan
    mu = np.mean(rets); var_1 = np.sum((rets - mu) ** 2) / n
    if var_1 == 0: return np.nan
    q_rets = log_p[q:] - log_p[:-q]
    return np.sum((q_rets - q * mu) ** 2) / (n * q) / var_1


def prep(df):
    """모든 모델에서 쓰는 지표 다 계산"""
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    
    df["ma"] = df["close"].rolling(BB_LENGTH).mean()
    df["std"] = df["close"].rolling(BB_LENGTH).std()
    
    # 여러 BB 임계 (모델별)
    df["upper_2"] = df["ma"] + 2.0 * df["std"]
    df["lower_2"] = df["ma"] - 2.0 * df["std"]
    df["upper_2_5"] = df["ma"] + 2.5 * df["std"]
    df["lower_2_5"] = df["ma"] - 2.5 * df["std"]
    
    df["upper_walk"] = df["ma"] + BAND_WALK_SIGMA * df["std"]
    df["lower_walk"] = df["ma"] - BAND_WALK_SIGMA * df["std"]
    
    df["bbw"] = (df["upper_2"] - df["lower_2"]) / df["ma"]
    df["bbw_high_th"] = df["bbw"].rolling(BBW_PERCENTILE_WINDOW).quantile(EXPANSION_Q)
    df["is_expansion"] = df["bbw"] > df["bbw_high_th"]
    
    above = (df["close"] > df["upper_walk"]).astype(int)
    below = (df["close"] < df["lower_walk"]).astype(int)
    df["walk_up"] = above.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD
    df["walk_down"] = below.rolling(BAND_WALK_BARS).sum() >= BAND_WALK_THRESHOLD
    
    # 다중 q VR
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
    
    # 5중 레짐
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
    
    critical = ["ma", "upper_2", "lower_2", "atr", "vr_score", "q20",
                "q40", "q60", "q80", "regime5", "mom_val", "vr16", "vr30"]
    return df.dropna(subset=critical).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 모델 v1 (기준)
# ═══════════════════════════════════════════════════════════════════
def model_v1(df):
    pos, ep, sp, trail, m_pos, src, edt = 0, 0.0, None, None, None, None, None
    trades = []
    
    for row in df.to_dict("records"):
        c = row["close"]
        u = row["upper_2"]; l = row["lower_2"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        
        vr_rev = row["vr16"] < V1_VR_LOWER and row["vr30"] < V1_VR_LOWER
        vr_tr = row["vr16"] > V1_VR_UPPER and row["vr30"] > V1_VR_UPPER
        bbw_long = row["is_expansion"] and row["walk_up"]
        bbw_short = row["is_expansion"] and row["walk_down"]
        
        if pos == 0:
            if vr_rev:
                if c < l: pos, ep, sp, m_pos, src, edt = 1, c, c - STOP_ATR * atr, "rev", "vr_rev", dt
                elif c > u: pos, ep, sp, m_pos, src, edt = -1, c, c + STOP_ATR * atr, "rev", "vr_rev", dt
            elif vr_tr:
                if mom != 0:
                    d = 1 if mom > 0 else -1
                    pos, ep, m_pos, trail, src, edt = d, c, "trend", c, "vr_trend", dt
                    sp = c - d * STOP_ATR * atr
            elif bbw_long:
                pos, ep, m_pos, trail, src, edt = 1, c, "trend", c, "bbw_long", dt
                sp = c - STOP_ATR * atr
            elif bbw_short:
                pos, ep, m_pos, trail, src, edt = -1, c, "trend", c, "bbw_short", dt
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
# 모델 v3 (차별화 뒤집기)
# ═══════════════════════════════════════════════════════════════════
def model_v3(df):
    pos, ep, sp, trail, m_pos, entry_reg, src, edt, tm = 0, 0.0, None, None, None, None, None, None, None
    trades = []
    
    for row in df.to_dict("records"):
        c = row["close"]
        u2 = row["upper_2"]; l2 = row["lower_2"]
        u25 = row["upper_2_5"]; l25 = row["lower_2_5"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        reg = row["regime5"]
        
        if pos == 0:
            entered = False
            
            # strong_rev: BB ±2.5σ (더 빡센)
            if reg == "strong_rev":
                if c < l25:
                    pos, ep, sp, m_pos, entry_reg, src, edt = 1, c, c - STOP_ATR * atr, "rev", reg, "vr_strong_rev", dt
                    entered = True
                elif c > u25:
                    pos, ep, sp, m_pos, entry_reg, src, edt = -1, c, c + STOP_ATR * atr, "rev", reg, "vr_strong_rev", dt
                    entered = True
            # rev: BB ±2σ (기존)
            elif reg == "rev":
                if c < l2:
                    pos, ep, sp, m_pos, entry_reg, src, edt = 1, c, c - STOP_ATR * atr, "rev", reg, "vr_rev", dt
                    entered = True
                elif c > u2:
                    pos, ep, sp, m_pos, entry_reg, src, edt = -1, c, c + STOP_ATR * atr, "rev", reg, "vr_rev", dt
                    entered = True
            # mom / strong_mom: 추적 ATR 차별화
            elif reg == "mom":
                if mom != 0:
                    d = 1 if mom > 0 else -1
                    pos, ep, m_pos, trail, entry_reg, src, edt = d, c, "trend", c, reg, "vr_mom", dt
                    sp = c - d * STOP_ATR * atr
                    tm = V3_TRAIL_MOM
                    entered = True
            elif reg == "strong_mom":
                if mom != 0:
                    d = 1 if mom > 0 else -1
                    pos, ep, m_pos, trail, entry_reg, src, edt = d, c, "trend", c, reg, "vr_strong_mom", dt
                    sp = c - d * STOP_ATR * atr
                    tm = V3_TRAIL_STRONG_MOM  # 빨리 익절
                    entered = True
            
            # BBW: v1 방식 (VR 추세 없을 때만, 즉 진입 안 됐을 때)
            if not entered and reg not in ("mom", "strong_mom") and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg, src, edt = 1, c, "trend", c, reg, "bbw_long", dt
                    sp = c - STOP_ATR * atr
                    tm = V3_TRAIL_MOM
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg, src, edt = -1, c, "trend", c, reg, "bbw_short", dt
                    sp = c + STOP_ATR * atr
                    tm = V3_TRAIL_MOM
            continue
        
        raw = None
        if pos == 1:
            if m_pos == "rev":
                ue = u25 if entry_reg == "strong_rev" else u2
                if c <= sp or c >= ue: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                still = (row["regime5"] in ("mom", "strong_mom")) or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - tm * atr or not still: raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                le = l25 if entry_reg == "strong_rev" else l2
                if c >= sp or c <= le: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                still = (row["regime5"] in ("mom", "strong_mom")) or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + tm * atr or not still: raw = (ep - c) / ep
        
        if raw is not None:
            trades.append({"ret": raw - COST, "src": src, "entry_reg": entry_reg, "exit_dt": dt})
            pos = 0
            entry_reg, tm = None, None
    
    return trades


# ═══════════════════════════════════════════════════════════════════
# 모델 v4 (단순화 + mom 제거)
# ═══════════════════════════════════════════════════════════════════
def model_v4(df):
    pos, ep, sp, trail, m_pos, entry_reg, src, edt = 0, 0.0, None, None, None, None, None, None
    trades = []
    
    for row in df.to_dict("records"):
        c = row["close"]
        u2 = row["upper_2"]; l2 = row["lower_2"]
        atr = row["atr"]; mom = row["mom_val"]; dt = row["datetime"]
        reg = row["regime5"]
        
        if pos == 0:
            entered = False
            
            # 회귀: strong_rev + rev 둘 다 BB ±2σ (통일)
            if reg in ("strong_rev", "rev"):
                if c < l2:
                    pos, ep, sp, m_pos, entry_reg, src, edt = 1, c, c - STOP_ATR * atr, "rev", reg, f"vr_{reg}", dt
                    entered = True
                elif c > u2:
                    pos, ep, sp, m_pos, entry_reg, src, edt = -1, c, c + STOP_ATR * atr, "rev", reg, f"vr_{reg}", dt
                    entered = True
            # strong_mom: BBW 결합 필수 (확신 강한 모멘텀만)
            elif reg == "strong_mom" and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg, src, edt = 1, c, "trend", c, reg, "vr_strong_mom_bbw", dt
                    sp = c - STOP_ATR * atr
                    entered = True
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg, src, edt = -1, c, "trend", c, reg, "vr_strong_mom_bbw", dt
                    sp = c + STOP_ATR * atr
                    entered = True
            # mom 진입 제거 (노이즈)
            
            # BBW 단독 (회귀/strong_mom 외 시점에서만)
            if not entered and reg not in ("strong_rev", "rev", "strong_mom") and row["is_expansion"]:
                if row["walk_up"]:
                    pos, ep, m_pos, trail, entry_reg, src, edt = 1, c, "trend", c, reg, "bbw_long", dt
                    sp = c - STOP_ATR * atr
                elif row["walk_down"]:
                    pos, ep, m_pos, trail, entry_reg, src, edt = -1, c, "trend", c, reg, "bbw_short", dt
                    sp = c + STOP_ATR * atr
            continue
        
        raw = None
        if pos == 1:
            if m_pos == "rev":
                if c <= sp or c >= u2: raw = (c - ep) / ep
            else:
                trail = max(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_up"])
                if c <= sp or c <= trail - V4_TRAIL_MOM * atr or not still: raw = (c - ep) / ep
        elif pos == -1:
            if m_pos == "rev":
                if c >= sp or c <= l2: raw = (ep - c) / ep
            else:
                trail = min(trail, c)
                still = (row["regime5"] == "strong_mom") or (row["is_expansion"] and row["walk_down"])
                if c >= sp or c >= trail + V4_TRAIL_MOM * atr or not still: raw = (ep - c) / ep
        
        if raw is not None:
            trades.append({"ret": raw - COST, "src": src, "entry_reg": entry_reg, "exit_dt": dt})
            pos = 0
            entry_reg = None
    
    return trades


# ═══════════════════════════════════════════════════════════════════
# 포트폴리오 (모델별 함수 받아서)
# ═══════════════════════════════════════════════════════════════════
def portfolio_generic(data, model_fn):
    """간단한 포트폴리오: 각 종목 trade 리스트를 시간순으로 합치고 동시 보유 제한
       정확한 시간축 동시운용은 아니지만 비교용으론 충분"""
    # 각 종목의 trade들 모음
    all_trades_by_p = {}
    for p in data:
        all_trades_by_p[p] = model_fn(data[p])
    
    # 시간순 정렬
    combined = []
    for p, trs in all_trades_by_p.items():
        for t in trs:
            combined.append({"product": p, "ret": t["ret"], "exit_dt": t["exit_dt"]})
    combined.sort(key=lambda x: x["exit_dt"])
    
    # 합산 시계열
    if not combined:
        return pd.Series(dtype=float)
    
    df_t = pd.DataFrame(combined)
    df_t["ret_alloc"] = df_t["ret"] * ALLOC  # 1/3 배분
    return df_t.groupby("exit_dt")["ret_alloc"].sum()


# ═══════════════════════════════════════════════════════════════════
# 통계
# ═══════════════════════════════════════════════════════════════════
def stats_of(trades_or_rets, years):
    if isinstance(trades_or_rets, list) and len(trades_or_rets) > 0 and isinstance(trades_or_rets[0], dict):
        rets = np.array([t["ret"] for t in trades_or_rets])
    else:
        rets = np.array(trades_or_rets) if not isinstance(trades_or_rets, np.ndarray) else trades_or_rets
    
    if len(rets) == 0:
        return dict(n=0, wr=0, cagr=0, pf=0, sharpe=0, mdd=0, total=0)
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


# ═══════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 92)
    print(" v1 vs v3 vs v4 동시 비교 (ROLL_Q=800)")
    print("=" * 92)
    
    data = {}
    years_ref = 5.0
    for p, fn in FILES.items():
        path = os.path.join(DATA_DIR, fn)
        df = pd.read_csv(path, parse_dates=["datetime"])
        data[p] = prep(df)
        years_ref = (data[p]["datetime"].max() - data[p]["datetime"].min()).days / 365.25
    
    # 추가 버전.
    print("\n=== 데이터 영역 확인 ===")
    for p in PRODUCTS:
        d = data[p]
        print(f"{p}: {len(d):>6}봉, "
              f"{d['datetime'].iloc[0].strftime('%Y-%m-%d')} ~ "
              f"{d['datetime'].iloc[-1].strftime('%Y-%m-%d')}"
              )
    print()

    # 세 모델 다 돌림
    results = {}
    for name, model_fn in [("v1", model_v1), ("v3", model_v3), ("v4", model_v4)]:
        all_tr = []
        for p in PRODUCTS:
            all_tr.extend(model_fn(data[p]))
        results[name] = all_tr
    
    # ─── [1] 합산 비교 ───
    print("\n[1] 합산 풀링 비교")
    print(f"  {'모델':<5} {'거래':>5} {'CAGR':>7} {'PF':>5} {'Sharpe':>7} {'MDD':>7} {'승률':>6} {'t-stat':>7} {'p':>7}")
    print("  " + "─" * 70)
    for name in ["v1", "v3", "v4"]:
        s = stats_of(results[name], years_ref)
        rets = np.array([t["ret"] for t in results[name]])
        t_stat, p_val = stats.ttest_1samp(rets, 0)
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        print(f"  {name:<5} {s['n']:>5} {s['cagr']:>6.2f}% {s['pf']:>5.2f} "
              f"{s['sharpe']:>7.3f} {s['mdd']:>6.1f}% {s['wr']:>5.1f}% "
              f"{t_stat:>7.3f} {p_one:>7.4f}")
    
    # ─── [2] 종목별 비교 ───
    print("\n[2] 종목별 Sharpe")
    print(f"  {'모델':<5}", end="")
    # 비교 코드에서 prep() 호출 직후 데이터 크기 확인
    for p in PRODUCTS:
        print(f"{p}: {len(data[p])} rows, "
            f"{data[p]['datetime'].min()} ~ {data[p]['datetime'].max()}")
    print()
    print("  " + "─" * 40)
    for name, model_fn in [("v1", model_v1), ("v3", model_v3), ("v4", model_v4)]:
        print(f"  {name:<5}", end="")
        for p in PRODUCTS:
            tr = model_fn(data[p])
            s = stats_of(tr, years_ref)
            print(f"{s['sharpe']:>10.3f}", end="")
        print()
    
    # ─── [3] 연도별 비교 ───
    print("\n[3] 연도별 수익률 (3 모델 동시)")
    print(f"  {'연도':>6}", end="")
    for name in ["v1", "v3", "v4"]:
        print(f"  {name+'_ret':>10} {name+'_n':>6}", end="")
    print()
    print("  " + "─" * 60)
    
    yearly_data = {name: {} for name in ["v1", "v3", "v4"]}
    for name in ["v1", "v3", "v4"]:
        tr_df = pd.DataFrame(results[name])
        tr_df["year"] = tr_df["exit_dt"].dt.year
        for y in sorted(tr_df["year"].unique()):
            yt = tr_df[tr_df["year"] == y]
            yearly_data[name][y] = {
                "ret": (np.prod(1 + yt["ret"]) - 1) * 100,
                "n": len(yt)
            }
    
    all_years = sorted({y for name in ["v1", "v3", "v4"] for y in yearly_data[name]})
    pos_counts = {"v1": 0, "v3": 0, "v4": 0}
    for y in all_years:
        print(f"  {y:>6}", end="")
        for name in ["v1", "v3", "v4"]:
            if y in yearly_data[name]:
                d = yearly_data[name][y]
                print(f"  {d['ret']:>+9.2f}% {d['n']:>6}", end="")
                if d['ret'] > 0: pos_counts[name] += 1
            else:
                print(f"  {'-':>10} {'-':>6}", end="")
        print()
    print(f"\n  수익 연도:  v1: {pos_counts['v1']}/{len(all_years)},  "
          f"v3: {pos_counts['v3']}/{len(all_years)},  v4: {pos_counts['v4']}/{len(all_years)}")
    
    # ─── [4] v3 레짐별 (차별화 효과 검증) ───
    print("\n[4] v3 레짐별 거래 성과 (strong_rev > rev면 차별화 성공)")
    print(f"  {'레짐':<14} {'거래':>5} {'CAGR':>7} {'Sharpe':>7} {'합산':>8}")
    print("  " + "─" * 48)
    v3_tr = results["v3"]
    for reg in ["strong_rev", "rev", "mom", "strong_mom"]:
        sub = [t for t in v3_tr if t.get("entry_reg") == reg]
        if not sub: continue
        s = stats_of(sub, years_ref)
        print(f"  {reg:<14} {s['n']:>5} {s['cagr']:>6.2f}% {s['sharpe']:>7.3f} {s['total']:>+7.1f}%")
    
    # ─── [5] v4 레짐별 ───
    print("\n[5] v4 레짐별 거래 성과")
    print(f"  {'레짐':<14} {'거래':>5} {'CAGR':>7} {'Sharpe':>7} {'합산':>8}")
    print("  " + "─" * 48)
    v4_tr = results["v4"]
    for reg in ["strong_rev", "rev", "strong_mom", "neutral"]:
        sub = [t for t in v4_tr if t.get("entry_reg") == reg]
        if not sub: continue
        s = stats_of(sub, years_ref)
        print(f"  {reg:<14} {s['n']:>5} {s['cagr']:>6.2f}% {s['sharpe']:>7.3f} {s['total']:>+7.1f}%")
    
    # ─── [6] BBW 효과 ───
    print("\n[6] BBW 거래 성과 (3 모델)")
    print(f"  {'모델':<5} {'BBW거래':>8} {'Sharpe':>7} {'합산':>8}")
    print("  " + "─" * 32)
    for name in ["v1", "v3", "v4"]:
        bbw_tr = [t for t in results[name] if t.get("src", "").startswith("bbw")]
        if not bbw_tr:
            print(f"  {name:<5} {'-':>8} {'-':>7} {'-':>8}")
            continue
        s = stats_of(bbw_tr, years_ref)
        print(f"  {name:<5} {s['n']:>8} {s['sharpe']:>7.3f} {s['total']:>+7.1f}%")
    
    # ─── [7] 포트폴리오 비교 ───
    print("\n[7] 포트폴리오 비교")
    print(f"  {'모델':<5} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7}")
    print("  " + "─" * 36)
    for name, model_fn in [("v1", model_v1), ("v3", model_v3), ("v4", model_v4)]:
        rs = portfolio_generic(data, model_fn)
        ps = port_stats(rs, years_ref)
        print(f"  {name:<5} {ps['cagr']:>6.2f}% {ps['sharpe']:>7.3f} {ps['mdd']:>6.1f}%")
    
    # ─── 요약 ───
    print("\n" + "=" * 92)
    print("요약본")
    print("=" * 92)
    print("  - 합산 Sharpe 가장 높은 모델이 우승")
    print("  - 수익 연도 비율 (안정성) 가장 높은 모델 = 가장 견고함")
    print("  - v3: strong_rev Sharpe > rev Sharpe면 차별화 효과 입증")
    print("  - v4: 단순화가 효과 있으면 채택, 아니면 v1 + 5중 노이즈 = 의미 없음")
    print("  - BBW가 v1에서 양수, v3/v4에서 음수면 BBW는 VR 결합 시에만 효과")


if __name__ == "__main__":
    main()
