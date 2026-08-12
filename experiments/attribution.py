"""The rigor-waterfall + friction-frontier attribution harness over the BlackSwan free-data cross-asset battery.

For each family it decomposes a naive headline Sharpe into four ordered rungs of destruction —
  headline -> (-selection/OOS) -> (-quoted cost) -> (-multiplicity/Deflated-Sharpe) -> (-capacity/friction) -> net
— using ONLY recorded cell metrics (oos_sharpe swept over fee, turnover, skew/kurt, n) plus an OBSERVABLE friction
index built from free daily OHLCV (Corwin-Schultz high-low spread + dollar-ADV). Then:
  (A) per-asset-class failure fingerprint: which rung destroys the most alpha in each market;
  (B) the friction frontier: net survival vs observable friction, with the survivor at the low-friction corner;
  (C) the MAKE-OR-BREAK test: does the friction index reorder survival BEYOND turnover alone (partial-R^2 > 0)?
Reads scratchpad/cells.json (exported from Postgres) + BlackSwan/<dir>/<ASSET>-1d-*.json OHLCV. Writes results.json.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

SP = os.path.dirname(os.path.abspath(__file__))
BS = os.path.dirname(SP)
GAMMA = 0.5772156649015329  # Euler-Mascheroni, for the Bailey-Lopez de Prado expected-max-Sharpe benchmark

# ---- normal inverse CDF (Acklam) so we need no scipy -------------------------------------------------------
def norm_ppf(p):
    if p <= 0: return -np.inf
    if p >= 1: return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

# ---- asset universe: routing + classification -------------------------------------------------------------
ASSET_CLASS = {**{a: "crypto" for a in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]},
               **{a: "equity" for a in ["AAPL", "NVDA", "SPY"]},
               **{a: "commodity" for a in ["GOLD", "SILVER", "COPPER", "WTI"]},
               **{a: "bond" for a in ["TLT", "IEF", "SHY"]}, "UUP": "fx"}
DIRS = ["binance", "stocks", "etfs", "commodities", "fx"]
DIVERSIFIED = ["GOLD", "SILVER", "COPPER", "WTI", "SPY", "TLT", "IEF", "UUP"]
# family -> (primary asset class for the fingerprint, representative asset set for friction)
FAMILY_MAP = {
    "tsmom": ("multi", DIVERSIFIED), "xsmom": ("multi", DIVERSIFIED), "xsection": ("multi", DIVERSIFIED),
    "lowvol": ("multi", DIVERSIFIED), "seasonal": ("multi", DIVERSIFIED), "pairs": ("multi", DIVERSIFIED),
    "globalfactors": ("multi", DIVERSIFIED), "riskparity": ("multi", DIVERSIFIED),
    "roll": ("commodity", ["WTI"]), "roll-basket": ("commodity", ["WTI", "GOLD", "COPPER"]),
    "overnight": ("equity", ["SPY"]), "fomc": ("equity", ["SPY", "BTCUSDT"]),
    "cot": ("commodity", ["WTI", "GOLD", "COPPER"]),
}
CRYPTO_FAMILIES = ["run", "worldmodel", "voltarget", "attention", "funding", "liquidation", "orderflow",
                   "regime", "intraday", "events", "cross-cot"]
for f in CRYPTO_FAMILIES:
    FAMILY_MAP[f] = ("crypto", ["BTCUSDT"])
REALISTIC_FEE = {"crypto": 0.001, "equity": 0.0005, "commodity": 0.0005, "bond": 0.0003, "multi": 0.0005, "fx": 0.0003}
VOL_ANN = {"crypto": 0.60, "equity": 0.18, "commodity": 0.28, "bond": 0.08, "multi": 0.15, "fx": 0.08}


# ---- observable friction index from free OHLCV (Corwin-Schultz spread + dollar-ADV) ------------------------
def load_ohlc(asset):
    for d in DIRS:
        paths = sorted(glob.glob(f"{BS}/{d}/{asset}-1d-*.json"))
        if paths:
            df = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
            for c in ["price_high", "price_low", "price"]:
                if c not in df.columns:
                    return None
            df = df.dropna(subset=["price_high", "price_low", "price"]).reset_index(drop=True)
            return df
    return None


def corwin_schultz(H, L):
    """Corwin-Schultz (2012) high-low bid-ask spread (fraction of price). Per-pair RAW estimate, averaged in
    ~21-day blocks with negatives zeroed at the BLOCK level (the paper's monthly convention) so liquid names get
    a small positive spread instead of the median-of-daily-clipped 0 pathology. Winsorized to tame outliers."""
    H = np.asarray(H, float); L = np.asarray(L, float)
    ok = (H > 0) & (L > 0) & (H >= L)
    H, L = H[ok], L[ok]
    if len(H) < 42:
        return np.nan
    hl = np.log(H / L) ** 2
    beta = hl[:-1] + hl[1:]
    Hmax = np.maximum(H[:-1], H[1:]); Lmin = np.minimum(L[:-1], L[1:])
    gamma = np.log(Hmax / Lmin) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))          # RAW per-pair spread (may be negative from noise)
    S = S[np.isfinite(S)]
    if len(S) < 21:
        return np.nan
    blocks = [S[i:i + 21] for i in range(0, len(S) - 20, 21)]
    bm = np.clip(np.array([b.mean() for b in blocks]), 0, 0.05)  # monthly mean, negatives->0, winsorize at 500bps
    return float(np.mean(bm)) if len(bm) else np.nan


def asset_friction():
    """Per-asset observable friction primitives: CS spread (bps) + median daily dollar-ADV ($)."""
    out = {}
    for asset, cls in ASSET_CLASS.items():
        df = load_ohlc(asset)
        if df is None:
            continue
        spread = corwin_schultz(df["price_high"], df["price_low"])
        if "asset_volume_quote" in df.columns and df["asset_volume_quote"].notna().any():
            adv = float(df["asset_volume_quote"].replace(0, np.nan).median())
        else:
            adv = float((df["price"] * df.get("volume", pd.Series(np.nan, index=df.index))).replace(0, np.nan).median())
        # front-continuous commodity/fx series carry no real futures volume on free data -> ADV < $1M is unreliable
        reliable = adv == adv and adv >= 1e6
        out[asset] = {"class": cls, "spread_bps": (spread * 1e4) if spread == spread else np.nan,
                      "adv_usd": adv if reliable else np.nan, "adv_reliable": reliable, "n": len(df)}
    return out


# ---- waterfall per family ----------------------------------------------------------------------------------
def expected_max_sharpe(v_sharpe, n_trials):
    """Bailey-Lopez de Prado expected maximum Sharpe of n independent zero-skill trials with cross-trial
    Sharpe std sqrt(v). The multiplicity benchmark a real signal must beat."""
    if n_trials < 2 or not (v_sharpe == v_sharpe) or v_sharpe <= 0:
        return 0.0
    sd = math.sqrt(v_sharpe)
    return sd * ((1 - GAMMA) * norm_ppf(1 - 1.0 / n_trials) + GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def annual_turnover(cells):
    tpd = np.array([c["metrics"].get("trades_per_day", np.nan) for c in cells], float)
    tpd = tpd[np.isfinite(tpd)]
    if len(tpd):
        return float(np.median(tpd)) * 252.0
    tn = np.array([c["metrics"].get("turnover", np.nan) for c in cells], float)
    tn = tn[np.isfinite(tn)]
    return float(np.median(tn)) if len(tn) else np.nan


def family_friction(fam, afriction):
    _cls, assets = FAMILY_MAP.get(fam, ("multi", DIVERSIFIED))
    sp = [afriction[a]["spread_bps"] for a in assets if a in afriction and afriction[a]["spread_bps"] == afriction[a]["spread_bps"]]
    adv = [afriction[a]["adv_usd"] for a in assets if a in afriction and afriction[a]["adv_usd"] == afriction[a]["adv_usd"]]
    spread = float(np.mean(sp)) if sp else np.nan
    dv = float(np.median(adv)) if adv else np.nan
    return spread, dv


def sharpe_at_fee(cells, lo=True):
    fees = sorted({c["config"].get("transaction_fee") for c in cells if c["config"].get("transaction_fee") is not None})
    if not fees:
        s = np.array([c["metrics"].get("oos_sharpe", np.nan) for c in cells], float)
        s = s[np.isfinite(s)]
        return (float(np.median(s)) if len(s) else np.nan), np.array([c["metrics"].get("oos_sharpe", np.nan) for c in cells], float)
    fee = fees[0] if lo else fees[-1]
    sel = [c for c in cells if c["config"].get("transaction_fee") == fee]
    s = np.array([c["metrics"].get("oos_sharpe", np.nan) for c in sel], float)
    s = s[np.isfinite(s)]
    return (float(np.median(s)) if len(s) else np.nan), s


def config_key(cfg):
    return tuple(sorted((k, str(v)) for k, v in cfg.items()
                        if k not in ("walk_forward_window", "seed", "transaction_fee", "asset")))


def run_waterfall(cells_by_fam, afriction):
    """DSR-correct waterfall: the multiplicity rung deflates the WINNING strategy config by the expected max of
    N INDEPENDENT trials, where N = number of distinct strategy configs (NOT cells — windows/seeds/fee are not
    independent trials). Rungs: headline(best config) -> -selection(vs median config) -> -cost(where fee swept)
    -> -multiplicity(DSR expected-max over N configs) -> -capacity(observable-friction impact haircut)."""
    rows = []
    ref_aum = 10e6
    for fam, cells in sorted(cells_by_fam.items()):
        cls = FAMILY_MAP.get(fam, ("multi", DIVERSIFIED))[0]
        by_cfg = {}
        for c in cells:
            s = c["metrics"].get("oos_sharpe")
            if s is None or s != s:
                continue
            by_cfg.setdefault(config_key(c["config"]), []).append(c)
        cfg_sharpe, cfg_n = [], []
        for _k, cs in by_cfg.items():
            ss = np.array([c["metrics"]["oos_sharpe"] for c in cs], float)
            cfg_sharpe.append(float(np.mean(ss)))                          # config's mean OOS Sharpe across windows
            no = np.array([c["metrics"].get("oos_n_obs", np.nan) for c in cs], float)
            cfg_n.append(float(np.nansum(no)) if np.isfinite(no).any() else 252.0)
        cfg_sharpe = np.array(cfg_sharpe)
        n_trials = len(cfg_sharpe)
        if n_trials < 2:
            continue
        n_obs = float(np.median(cfg_n))
        s_head = float(np.max(cfg_sharpe))                                 # reported winner
        s_oos = float(np.median(cfg_sharpe))                               # honest spec-curve center
        # cost rung: only where fee was actually swept
        fees = sorted({c["config"].get("transaction_fee") for c in cells if c["config"].get("transaction_fee") is not None})
        def sh_at(fee):
            vals = [c["metrics"].get("oos_sharpe") for c in cells if c["config"].get("transaction_fee") == fee]
            vals = [x for x in vals if x is not None and x == x]
            return float(np.median(vals)) if vals else np.nan
        if len(fees) >= 2 and sh_at(fees[0]) == sh_at(fees[0]) and sh_at(fees[-1]) == sh_at(fees[-1]):
            d_cost = max(0.0, sh_at(fees[0]) - sh_at(fees[-1])); cost_measured = True
        else:
            d_cost = 0.0; cost_measured = False
        # multiplicity: deflate the WINNER by expected-max under the null over N independent configs
        v = float(np.var(cfg_sharpe, ddof=1)) if n_trials > 1 else 0.0
        emax = expected_max_sharpe(v, n_trials)
        s_net = s_head - d_cost
        s_dsr = s_net - emax
        # capacity: square-root impact haircut at ref AUM using observable friction
        spread_bps, adv = family_friction(fam, afriction)
        turn = annual_turnover(cells)
        if spread_bps == spread_bps and adv and adv > 0 and turn == turn:
            impact_bps = 0.5 * spread_bps + 10.0 * math.sqrt(ref_aum / adv)
            drag_ann = impact_bps * 1e-4 * turn
            d_cap = drag_ann / VOL_ANN.get(cls, 0.15)
            friction_index = turn * impact_bps
        else:
            d_cap = np.nan; friction_index = np.nan; impact_bps = np.nan
        s_cap = s_dsr - (d_cap if d_cap == d_cap else 0.0)
        t_cap = s_cap * math.sqrt(max(n_obs, 1))
        rows.append(dict(
            family=fam, cls=cls, n_cells=len(cells), n_trials=n_trials, n_obs=int(n_obs),
            s_head=s_head, s_oos=s_oos, s_net=s_net, s_dsr=s_dsr, s_cap=s_cap,
            d_selection=s_head - s_oos, d_cost=d_cost, cost_measured=cost_measured, d_mult=emax,
            d_cap=(d_cap if d_cap == d_cap else 0.0), t_cap=t_cap,
            survives=bool(s_cap > 0 and t_cap > 3.0),
            turnover=turn, spread_bps=spread_bps, adv_usd=adv, impact_bps=impact_bps, friction_index=friction_index,
        ))
    return rows


# ---- make-or-break: does friction reorder survival BEYOND turnover? ---------------------------------------
def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2, ss_res, X.shape[1]


def make_or_break(rows):
    R = [r for r in rows if all(r[k] == r[k] for k in ("turnover", "spread_bps", "adv_usd", "friction_index")) and r["turnover"] > 0 and r["adv_usd"] > 0]
    if len(R) < 8:
        return {"note": f"too few families with full friction ({len(R)})"}
    y = np.array([r["s_net"] for r in R], float)
    lt = np.log(np.array([r["turnover"] for r in R]) + 1e-9)
    lsp = np.log(np.array([r["spread_bps"] for r in R]) + 1e-9)
    lill = -np.log(np.array([r["adv_usd"] for r in R]))          # illiquidity = 1/ADV
    _, r2_turn, ssr_a, k_a = ols(y, [lt])
    _, r2_full, ssr_b, k_b = ols(y, [lt, lsp, lill])
    n = len(y)
    df1, df2 = k_b - k_a, n - k_b
    F = ((ssr_a - ssr_b) / df1) / (ssr_b / df2) if df2 > 0 and ssr_b > 0 else float("nan")
    return {"n_families": n, "r2_turnover_only": r2_turn, "r2_turnover_plus_friction": r2_full,
            "partial_r2": r2_full - r2_turn, "F_added": F, "df1": df1, "df2": df2,
            "friction_beats_turnover": bool(r2_full - r2_turn > 0.05)}


def main():
    cells = json.load(open(f"{SP}/cells.json"))
    by_fam = {}
    for c in cells:
        by_fam.setdefault(c["family"], []).append(c)
    print(f"loaded {len(cells)} cells across {len(by_fam)} families")
    print("computing observable friction (Corwin-Schultz spread + dollar-ADV) per asset ...")
    afriction = asset_friction()
    for a, v in sorted(afriction.items(), key=lambda kv: kv[1]["class"]):
        print(f"  {a:8s} {v['class']:9s} spread {v['spread_bps']:7.1f}bps  ADV ${v['adv_usd']/1e6:10.1f}M  n={v['n']}")

    rows = run_waterfall(by_fam, afriction)

    print("\n" + "=" * 118)
    print("RIGOR WATERFALL  (Sharpe destroyed at each ordered rung; net = capacity-adjusted deflated Sharpe)")
    print("=" * 118)
    print(f"{'family':15s}{'cls':10s}{'n':>4s}{'head':>7s}{'oos':>7s}{'net':>7s}{'dsr':>7s}{'CAP':>7s} | "
          f"{'-sel':>7s}{'-cost':>7s}{'-mult':>7s}{'-cap':>7s}{'t':>7s}  surv")
    for r in sorted(rows, key=lambda r: -r["s_cap"]):
        print(f"{r['family']:15s}{r['cls']:10s}{r['n_cells']:>4d}{r['s_head']:>7.2f}{r['s_oos']:>7.2f}{r['s_net']:>7.2f}"
              f"{r['s_dsr']:>7.2f}{r['s_cap']:>7.2f} | {r['d_selection']:>7.2f}{r['d_cost']:>7.2f}{r['d_mult']:>7.2f}"
              f"{r['d_cap']:>7.2f}{r['t_cap']:>7.2f}  {'YES' if r['survives'] else '.'}")

    # ---- (A) asset-class failure fingerprint ----
    print("\n" + "=" * 70)
    print("(A) FAILURE FINGERPRINT — mean Sharpe destroyed per rung, by asset class")
    print("=" * 70)
    print(f"{'class':10s}{'n_fam':>6s}{'-sel':>8s}{'-cost':>8s}{'-mult':>8s}{'-cap':>8s}   dominant rung")
    fingerprint = {}
    for cls in ["crypto", "equity", "commodity", "bond", "multi", "fx"]:
        cr = [r for r in rows if r["cls"] == cls]
        if not cr:
            continue
        ds = {k: float(np.nanmean([r[k] for r in cr])) for k in ("d_selection", "d_cost", "d_mult", "d_cap")}
        dom = max(ds, key=ds.get)
        fingerprint[cls] = {"n_fam": len(cr), **ds, "dominant": dom}
        print(f"{cls:10s}{len(cr):>6d}{ds['d_selection']:>8.2f}{ds['d_cost']:>8.2f}{ds['d_mult']:>8.2f}{ds['d_cap']:>8.2f}   {dom}")

    # ---- (B) friction frontier ----
    print("\n" + "=" * 70)
    print("(B) FRICTION FRONTIER — net Sharpe vs observable friction index (log), low->high")
    print("=" * 70)
    fr = sorted([r for r in rows if r["friction_index"] == r["friction_index"]], key=lambda r: r["friction_index"])
    for r in fr:
        bar = "#" * max(0, int(r["s_net"] * 20)) if r["s_net"] > 0 else ""
        print(f"  frict {r['friction_index']:12.0f}  net {r['s_net']:+6.2f}  {r['family']:15s} {bar}")

    # ---- (C) make-or-break ----
    print("\n" + "=" * 70)
    print("(C) MAKE-OR-BREAK — does friction reorder survival BEYOND turnover?")
    print("=" * 70)
    mb = make_or_break(rows)
    print(json.dumps(mb, indent=2))

    json.dump({"waterfall": rows, "fingerprint": fingerprint, "friction_asset": afriction, "make_or_break": mb},
              open(f"{SP}/results.json", "w"), indent=2, default=float)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
