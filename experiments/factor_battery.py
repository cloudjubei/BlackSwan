"""Cross-asset factor battery with the FULL rigor cascade, computed WITHIN each asset class so the failure
fingerprint is powered and the waterfall starts from a real IN-SAMPLE headline.

For each (asset class x canonical factor) we sweep the factor's lookback, split the class's common history into a
TRAIN half and a TEST half, and read off:
  headline  = best-lookback GROSS Sharpe on TRAIN            (the naive in-sample optimized backtest a paper reports)
  -selection= headline - selected-lookback GROSS Sharpe on TEST   (the in-sample->OOS overfit gap; McLean-Pontiff)
  -cost     = TEST gross - TEST net (fee x turnover)
  -mult     = Bailey-Lopez de Prado expected-max Sharpe over the N swept lookbacks (Harvey-Liu haircut)
  -capacity = observable-friction (Corwin-Schultz spread + dollar-ADV) square-root-impact haircut
  net       = capacity-adjusted deflated OOS Sharpe;  survives iff net>0 and t>3

Classes: crypto (9 coins, 2022+), equity (10 large-caps, 2018+), commodity (7, 2006+), bond (3, 2006+). Factors:
momentum, value(long-run reversal), short-term reversal, low-vol, trend. All prices are free daily closes on disk."""
import glob
import math
import os

import numpy as np
import pandas as pd

BS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.dirname(os.path.abspath(__file__))
GAMMA = 0.5772156649015329

CLASSES = {
    "crypto": ("binance", ["ADAUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETHUSDT", "LTCUSDT", "SHIBUSDT", "SOLUSDT", "XRPUSDT"], 0.0010),
    "equity": ("stocks", ["AAPL", "AMZN", "AVGO", "GOOGL", "JPM", "META", "MSFT", "NVDA", "TSLA", "WMT"], 0.0005),
    "commodity": ("commodities", ["COPPER", "CORN", "GOLD", "NATGAS", "SILVER", "WHEAT", "WTI"], 0.0005),
    "bond": ("etfs", ["TLT", "IEF", "SHY"], 0.0003),
}
VOL_ANN = {"crypto": 0.60, "equity": 0.30, "commodity": 0.25, "bond": 0.07}


def norm_ppf(p):
    if p <= 0: return -8.0
    if p >= 1: return 8.0
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    pl=0.02425
    if p<pl:
        q=math.sqrt(-2*math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p<=1-pl:
        q=p-0.5; r=q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q=math.sqrt(-2*math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def expected_max_sharpe(v, n):
    if n < 2 or not (v == v) or v <= 0: return 0.0
    sd = math.sqrt(v)
    return sd * ((1 - GAMMA) * norm_ppf(1 - 1.0/n) + GAMMA * norm_ppf(1 - 1.0/(n*math.e)))


def load_class(directory, symbols):
    frames, hl = {}, {}
    for s in symbols:
        paths = sorted(glob.glob(f"{BS}/{directory}/{s}-1d-*.json"))
        if not paths: continue
        df = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
        if "price" not in df or "timestamp_close" not in df: continue
        idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp_close"]).dt.tz_localize(None)).normalize()
        px = pd.Series(df["price"].to_numpy(), index=idx).groupby(level=0).last().sort_index()
        frames[s] = px
        if "price_high" in df and "price_low" in df:
            H = pd.Series(df["price_high"].to_numpy(), index=idx).groupby(level=0).last().sort_index()
            L = pd.Series(df["price_low"].to_numpy(), index=idx).groupby(level=0).last().sort_index()
            q = df.get("asset_volume_quote")
            adv = pd.Series((q.to_numpy() if q is not None else np.nan), index=idx).groupby(level=0).last() if q is not None else None
            hl[s] = (H, L, adv)
    PX = pd.DataFrame(frames).sort_index()
    PX = PX.dropna(how="all")
    return PX, hl


def corwin_schultz(H, L):
    H = np.asarray(H, float); L = np.asarray(L, float)
    ok = (H > 0) & (L > 0) & (H >= L); H, L = H[ok], L[ok]
    if len(H) < 42: return np.nan
    hl = np.log(H/L)**2
    beta = hl[:-1] + hl[1:]
    Hmax = np.maximum(H[:-1], H[1:]); Lmin = np.minimum(L[:-1], L[1:])
    gamma = np.log(Hmax/Lmin)**2
    k = 3 - 2*math.sqrt(2)
    alpha = (np.sqrt(2*beta) - np.sqrt(beta))/k - np.sqrt(gamma/k)
    S = 2*(np.exp(alpha)-1)/(1+np.exp(alpha)); S = S[np.isfinite(S)]
    if len(S) < 21: return np.nan
    blocks = [S[i:i+21] for i in range(0, len(S)-20, 21)]
    return float(np.mean(np.clip([b.mean() for b in blocks], 0, 0.05)))


def class_friction(hl):
    sp, adv = [], []
    for _s, (H, L, a) in hl.items():
        cs = corwin_schultz(H, L)
        if cs == cs: sp.append(cs*1e4)
        if a is not None:
            m = float(pd.Series(a).replace(0, np.nan).median())
            if m == m and m >= 1e6: adv.append(m)
    return (float(np.median(sp)) if sp else np.nan), (float(np.median(adv)) if adv else np.nan)


def weight_book(score, PX, invert, k, reb=21):
    idx = PX.index; cols = list(PX.columns)
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    held = pd.Series(0.0, index=cols); since = reb
    for i in range(len(idx)):
        W.iloc[i] = held.values; since += 1
        if since < reb: continue
        s = score.iloc[i].dropna()
        if len(s) < 2*k: continue
        since = 0
        order = s.sort_values(ascending=invert)
        nxt = pd.Series(0.0, index=cols)
        for sym in order.index[:k]: nxt[sym] = 1.0/k
        for sym in order.index[-k:]: nxt[sym] = -1.0/k
        held = nxt
    return W


def trend_book(PX, lookback, k, reb=21):
    sig = np.sign(PX / PX.shift(lookback) - 1.0)
    idx = PX.index; cols = list(PX.columns); W = pd.DataFrame(0.0, index=idx, columns=cols)
    held = pd.Series(0.0, index=cols); since = reb
    for i in range(len(idx)):
        W.iloc[i] = held.values; since += 1
        if since < reb: continue
        s = sig.iloc[i].dropna()
        if s.abs().sum() == 0: continue
        since = 0
        held = (s/s.abs().sum()).reindex(cols).fillna(0.0)
    return W


def factor_score(PX, factor, lb):
    if factor == "momentum": return PX / PX.shift(lb) - 1.0, False
    if factor == "value": return PX / PX.shift(lb) - 1.0, True
    if factor == "st_reversal": return PX / PX.shift(lb) - 1.0, True
    if factor == "lowvol": return PX.pct_change().rolling(lb).std(), True
    return None, None


FACTORS = {
    "momentum": [63, 126, 252], "value": [504, 756, 1260], "st_reversal": [5, 10, 21],
    "lowvol": [30, 60, 120], "trend": [100, 150, 200],
}


def sharpe(r, ann=True):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0: return np.nan, 0
    return (r.mean()/r.std()) * (math.sqrt(252) if ann else 1.0), len(r)


def run():
    rows = []
    friction = {}
    for cls, (d, syms, fee) in CLASSES.items():
        PX, hl = load_class(d, syms)
        if PX.shape[1] < 3 or len(PX) < 300: continue
        RET = PX.pct_change()
        n = len(PX); split = int(n*0.55)
        tr, te = PX.index[:split], PX.index[split:]
        k = 2 if PX.shape[1] >= 5 else 1
        spread_bps, adv = class_friction(hl); friction[cls] = {"spread_bps": spread_bps, "adv_usd": adv, "n_assets": PX.shape[1]}
        for factor, lbs in FACTORS.items():
            lbs = [lb for lb in lbs if lb < split - 30]
            if not lbs: continue
            per_lb = []
            for lb in lbs:
                if factor == "trend": W = trend_book(PX, lb, k)
                else:
                    sc, inv = factor_score(PX, factor, lb); W = weight_book(sc, PX, inv, k)
                gross = (W.shift(1).fillna(0.0) * RET).sum(axis=1)
                dW = W.diff().abs().sum(axis=1)
                net = gross - fee * dW
                s_tr_g, _ = sharpe(gross.loc[tr])
                s_te_g, nte = sharpe(gross.loc[te])
                s_te_n, _ = sharpe(net.loc[te])
                turn = float(dW.loc[te].mean()) * 252
                per_lb.append(dict(lb=lb, s_tr_g=s_tr_g, s_te_g=s_te_g, s_te_n=s_te_n, turn=turn, nte=nte))
            per_lb = [p for p in per_lb if p["s_tr_g"] == p["s_tr_g"]]
            if not per_lb: continue
            best = max(per_lb, key=lambda p: p["s_tr_g"])          # naive in-sample optimizer picks best-on-train
            s_head = best["s_tr_g"]                                 # in-sample headline (annualized)
            s_oos_g = best["s_te_g"] if best["s_te_g"] == best["s_te_g"] else 0.0
            s_oos_n = best["s_te_n"] if best["s_te_n"] == best["s_te_n"] else 0.0
            tr_sh = np.array([p["s_tr_g"] for p in per_lb], float)
            emax = expected_max_sharpe(float(np.var(tr_sh, ddof=1)) if len(tr_sh) > 1 else 0.0, len(per_lb))
            s_dsr = s_oos_n - emax
            impact_bps = (0.5*spread_bps + 10.0*math.sqrt(10e6/adv)) if (spread_bps == spread_bps and adv == adv and adv > 0) else np.nan
            # annualized-Sharpe cost from impact: (annual return drag = impact_frac x annual turnover) / annual vol
            d_cap = (impact_bps*1e-4*best["turn"]/VOL_ANN[cls]) if impact_bps == impact_bps else 0.0
            s_cap = s_dsr - d_cap
            nte = best["nte"] or 252
            t_cap = (s_cap/math.sqrt(252)) * math.sqrt(nte)         # back to per-bar Sharpe * sqrt(n) = t
            rows.append(dict(cls=cls, factor=factor, k=k, n_lb=len(per_lb), nte=nte,
                s_head=s_head, s_oos_g=s_oos_g, s_oos_n=s_oos_n, s_dsr=s_dsr, s_cap=s_cap,
                d_selection=s_head-s_oos_g, d_cost=s_oos_g-s_oos_n, d_mult=emax, d_cap=d_cap, t_cap=t_cap,
                turnover=best["turn"], spread_bps=spread_bps, adv_usd=adv, friction_index=(best["turn"]*impact_bps if impact_bps==impact_bps else np.nan),
                survives=bool(s_cap > 0 and t_cap > 3.0)))
    return rows, friction


def ols_r2(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; ssr = float(resid@resid); sst = float(((y-y.mean())**2).sum())
    return (1 - ssr/sst if sst > 0 else 0.0), ssr, X.shape[1]


def main():
    rows, friction = run()
    print("per-class friction:", {c: f"spread {v['spread_bps']:.1f}bps ADV ${(v['adv_usd'] or float('nan'))/1e6:.0f}M ({v['n_assets']} assets)" for c, v in friction.items()})
    print("\n" + "="*120)
    print("CROSS-ASSET FACTOR BATTERY — in-sample headline knocked down through the rigor cascade (annualized Sharpe)")
    print("="*120)
    print(f"{'class':10s}{'factor':12s}{'k':>2s}{'IS_head':>8s}{'OOS_g':>7s}{'OOS_n':>7s}{'dsr':>7s}{'CAP':>7s} | {'-select':>8s}{'-cost':>7s}{'-mult':>7s}{'-cap':>7s}{'t':>7s} surv")
    for r in sorted(rows, key=lambda r: (r["cls"], -r["s_cap"])):
        print(f"{r['cls']:10s}{r['factor']:12s}{r['k']:>2d}{r['s_head']:>8.2f}{r['s_oos_g']:>7.2f}{r['s_oos_n']:>7.2f}{r['s_dsr']:>7.2f}{r['s_cap']:>7.2f} | "
              f"{r['d_selection']:>8.2f}{r['d_cost']:>7.2f}{r['d_mult']:>7.2f}{r['d_cap']:>7.2f}{r['t_cap']:>7.2f}  {'YES' if r['survives'] else '.'}")
    surv = [r for r in rows if r["survives"]]
    surv_labels = [r["cls"] + "/" + r["factor"] for r in surv]
    print("\nSURVIVORS at t>3: %d/%d  %s" % (len(surv), len(rows), surv_labels))

    print("\n" + "="*72)
    print("FAILURE FINGERPRINT — mean annualized Sharpe destroyed per rung, by asset class")
    print("="*72)
    print(f"{'class':10s}{'n':>3s}{'IS_head':>9s}{'-select':>9s}{'-cost':>8s}{'-mult':>8s}{'-cap':>8s}   dominant")
    fingerprint = {}
    for cls in CLASSES:
        cr = [r for r in rows if r["cls"] == cls]
        if not cr: continue
        agg = {k: float(np.mean([r[k] for r in cr])) for k in ("s_head", "d_selection", "d_cost", "d_mult", "d_cap")}
        dom = max(("d_selection", "d_cost", "d_mult", "d_cap"), key=lambda k: agg[k])
        fingerprint[cls] = {"n": len(cr), **agg, "dominant": dom}
        print(f"{cls:10s}{len(cr):>3d}{agg['s_head']:>9.2f}{agg['d_selection']:>9.2f}{agg['d_cost']:>8.2f}{agg['d_mult']:>8.2f}{agg['d_cap']:>8.2f}   {dom}")

    print("\n" + "="*72)
    print("MAKE-OR-BREAK — does friction reorder OOS-net survival beyond turnover?")
    print("="*72)
    R = [r for r in rows if all(r[k] == r[k] for k in ("turnover", "spread_bps", "adv_usd")) and r["turnover"] > 0 and r["adv_usd"] and r["adv_usd"] > 0]
    mb = {"n": len(R)}
    if len(R) >= 8:
        y = np.array([r["s_oos_n"] for r in R]); lt = np.log(np.array([r["turnover"] for r in R])+1e-9)
        lsp = np.log(np.array([r["spread_bps"] for r in R])+1e-9); lill = -np.log(np.array([r["adv_usd"] for r in R]))
        r2a, ssra, ka = ols_r2(y, [lt]); r2b, ssrb, kb = ols_r2(y, [lt, lsp, lill])
        nn = len(y); df1, df2 = kb-ka, nn-kb
        F = ((ssra-ssrb)/df1)/(ssrb/df2) if df2 > 0 and ssrb > 0 else float("nan")
        mb.update({"r2_turnover": r2a, "r2_turnover_plus_friction": r2b, "partial_r2": r2b-r2a, "F": F, "df1": df1, "df2": df2})
    print(mb)

    import json
    json.dump({"waterfall": rows, "fingerprint": fingerprint, "friction": friction, "make_or_break": mb, "n_survivors": len(surv)},
              open(f"{SP}/factor_battery_results.json", "w"), indent=2, default=float)
    print(f"\nwrote factor_battery_results.json ({len(rows)} class x factor cells)")


if __name__ == "__main__":
    main()
