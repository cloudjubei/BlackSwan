# The paper's novel contribution + source papers to replicate

Output of a web-grounded deep-research pass (4 researchers + synthesizer) on: given we've disproved 63/65
free-data edges, what can the paper offer that is genuinely NEW (not a review)?

## The novel thesis (the spine)

**Holding research discipline fixed across the entire anomaly zoo turns the replication debate into a
CONTROLLED NATURAL EXPERIMENT whose outcome is predicted ex-ante by limits-to-arbitrage theory.** We fix
pre-registration + mutation-proven point-in-time leakage guards + realistic per-trade cost + Deflated-Sharpe/
best-of-N multiplicity + deep-history OOS power identically across 18 families and 4 asset classes (crypto,
commodity, equity, rates), so **capacity is the only free variable**. Limits-to-arbitrage (Shleifer-Vishny;
Novy-Marx-Velikov) makes the *falsifiable prediction* that the last edge standing must sit in the
lowest-capacity, hardest-to-arbitrage corner — and it does: 63/65 collapse and the sole survivor is
small-breadth commodity-complex mean-reversion, INCONCLUSIVE precisely because low capacity = few independent
bets = low power. The engine is the instrument; **the falsifiable limits-to-arbitrage horse-race is the
contribution.** This simultaneously (a) TESTS rather than invokes LTA theory, (b) arbitrates the Hou-Xue-Zhang
("most anomalies false") vs Jensen-Kelly-Pedersen ("no replication crisis") standoff by showing "no crisis" is
an artifact of low-turnover, index-level, cost-free, pre-decay construction, and (c) shows that even
Baltussen-Swinkels-van Vliet (2021) "Global Factor Premiums" — the one prior multi-asset robustness claim —
FAILS TO REPLICATE on freely-obtainable data, collapsing to the single low-capacity commodity residual.

> **HONESTY CORRECTION (adversarial panel, verified — RESULT IS IN).** The Baltussen result is **SCOPE-LIMITED,
> not a flat contradiction.** We implement 4 of his 6 factors (commodity-heavy + rates, equal-weight, no equity
> single-stock cross-section, no FX); his diversified premium fails to replicate (t=−1.20, 4/13, no construction
> rescues it) — BUT the collapse is **panel composition, NOT cost** (momentum inverts even at zero fee; the
> financials momentum cross-section is a *degenerate empty book* — 4 ETFs cannot form a k=3 long/short), so the
> free panel structurally cannot reproduce his vol-scaled within-asset-class engine. The one positive component
> (value = commodity 5yr reversal, the residual) is itself sub-significant, so the LTA "last thing standing"
> reading is directional. **Do NOT claim to refute his full multi-asset result** (needs equity single-stock /
> FX / term-structure data). Lead the paper with the LTA horse-race + engine; Baltussen is a scoped supporting
> piece. That the adversarial panel *caught* this overclaim is itself a demonstration of the methods contribution.

## Ranked contributions (lead with the empirical spine; engine is the instrument)

1. **[STRONG] Uniform-discipline sieve as a falsifiable TEST of limits-to-arbitrage** — capacity is the sorting
   variable that predicts which corner survives; the survivor lands exactly where LTA says it must.
2. **[STRONG] First implemented adversarial reproduce-AND-refute engine for PUBLISHED finance claims** — oracle
   = leakage-controlled, cost-realistic, multiplicity-corrected OOS *survival* (not code/rubric fidelity, not
   in-sample IC); an LLM panel must FAIL to overturn each verdict and name the exact hole; PIT guards proven by
   mutation testing. Deliverable is *earned negative results*.
3. **[STRONG→MEDIUM] Cross-asset arbitration of HXZ-vs-JKP + scope-limited non-replication of Baltussen (2021)** — his diversified premium fails to replicate on the free tradeable slice (panel-composition, not cost), collapsing to the commodity residual; NOT a refutation of his full vol-scaled multi-asset result (equity single-stock/FX/term-structure data-gated). Downgraded from STRONG by the adversarial panel — lead with the LTA horse-race, not this.
4. **[MEDIUM] Deep-history OOS power-extension** as a reusable pre-registered primitive that flips
   INCONCLUSIVE→DISPROVED (statistical power as a first-class design lever).
5. **[MEDIUM] An apples-to-apples, adversarially-verified cross-asset NULL CATALOGUE** (65 probes / 1,818
   cells on a common panel) as a reusable artifact.

## Do NOT frame the paper as (already-occupied / weak)
A review/meta-analysis; "we reproduced N anomalies" (Chen-Zimmermann own volume); "anomalies decay"
(McLean-Pontiff own it); "most anomalies are p-hacked" (HXZ own it, and Chen-Zimmermann dispute it — our
disproval is COST+DECAY+LEAKAGE, not a p-hacking claim); "commodity mean-reversion exists" (AMP/Zaremba own
existence — ours is the *capacity-sorting test*); "LTA explains persistence" as narrative (must TEST it);
"AI agents can reproduce papers" (PaperBench et al. own it); "we found new alpha" (false to our result); "a
backtesting protocol" as prose (Arnott-Harvey-Markowitz 2019 wrote it — we OPERATIONALIZE + enforce it).

## Source papers to replicate / engage — prioritized

### HIGH — the spine, all feasible on free data
- **Baltussen, Swinkels & van Vliet (2021), "Global Factor Premiums", JFE 142(3):1128-1154** — THE foil.
  Reproduce its 6 style premiums (trend, momentum, value, carry, seasonality, low-beta) on our panel to stage
  the direct, explained contradiction. *This is the top replication target.*
- **Jensen, Kelly & Pedersen (2023), "Is There a Replication Crisis in Finance?", JF 78(5):2465-2518** — the
  strongest "no crisis" counter; factor defs openly released; show why fixed cost/multiplicity/deep-OOS flips
  their optimism cross-asset.
- **Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere", JF 68:929-985** — our commodity
  mean-reversion residual IS their ~5yr commodity-value signal; must state it in their vocabulary.
- **Boons & Prado (2019), "Basis-Momentum", JF 74(1):239-279** — strongest competing explanation for the
  residual; replicate to rule in/out basis-momentum.
- **Bakshi, Gao & Rossi (2019), "Understanding the Sources of Risk...Commodity Returns", Mgmt Sci 65:619-641**
  — the 3-factor (average/basis/momentum) spanning benchmark the residual must show alpha against.
- **Novy-Marx & Velikov (2016), "A Taxonomy of Anomalies and Their Trading Costs", RFS 29(1):104-147** — the
  capacity/cost-survival methodology we extend cross-asset; quantifies how small the survivor's capacity is.

### MEDIUM — commodity residual framing (free data)
- **Gorton & Rouwenhorst (2006), "Facts and Fantasies about Commodity Futures", FAJ 62(2):47-68** — field-
  standard collateralized index/roll construction.
- **Miffre & Rallis (2007), "Momentum Strategies in Commodity Futures Markets", JBF** — short-horizon
  contrarian does NOT work; constrains the residual to long-horizon/value framing.
- **Fuertes, Miffre & Rallis (2010), "Tactical Allocation in Commodity Futures Markets", JBF** — reversal
  edges entangle with term structure; the basis control.
- **Han (2023), "Commodity momentum and reversal: Do they exist, and if so, why?", J. Futures Markets**
  (+ Bianchi et al. 2016) — the immediate prior art the residual must be differentiated from.

### MEDIUM — frame/cite (need paid CRSP/Compustat or non-free data)
- **McLean & Pontiff (2016), JF 71(1):5-32** — decay-as-arbitrage mechanism; frame, don't fully re-replicate.
- **Hou, Xue & Zhang (2020), "Replicating Anomalies", RFS 33(5):2019-2133** — the "most false" pole; cite.
- **Harvey, Liu & Zhu (2016), RFS 29(1):5-68** — multiplicity foundation our DSR operationalizes.
- **Zaremba, Bianchi & Mikutowski (2021), "Long-run reversal...seven centuries", JBF** — establishes
  commodity long-run reversal EXISTS (so existence is not our contribution); deep-history comparator.

### LOW — candidate NEW battery additions (free data; most will also die — that is the point)
- **Lucca & Moench (2015), "The Pre-FOMC Announcement Drift", JF 70(1):329-371** (+ Kurov et al. 2021, "The
  Disappearing Pre-FOMC Announcement Drift", FRL 40) — equity version already decayed; the CRYPTO 24/7
  pre-FOMC window (BTC + published FOMC dates) has never been cost+DSR-tested. A clean apples-to-apples null.
- **Lou, Polk & Skouras (2019), "A Tug of War: Overnight vs Intraday Expected Returns", JFE 134(1):192-213**
  (+ Da-Engelberg-Gao attention via Google Trends) — index/crypto overnight premium + crypto attention under
  cost+multiplicity are under-tested.

## Concurrent-work threats (AI-agent finance — READ IN FULL before writing; all verified real)
- **QRAFTI (arXiv:2604.18500, Lim & Muthuraman 2026)** — closest prior art: multi-agent EQUITY factor research
  (replicate + test signals + report). Differentiator = adversarial refutation panel + cost-survival gate +
  mutation-proven PIT guards + deep-history power extension + cross-asset + negative-results deliverable.
- Adjacent (verify specifics before citing): "From Hypotheses to Factors: Constrained LLM Agents in Crypto"
  (2604.26747); "Profit Mirage: Information Leakage in LLM-based Financial Agents" (2510.07920); PaperBench /
  CORE-Bench (reproduction-fidelity benchmarks — different oracle).
