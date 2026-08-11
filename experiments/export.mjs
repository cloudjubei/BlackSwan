#!/usr/bin/env node
// Publication-grade evidence export (plan §D): pull the entire pre-registered probe trail from the
// blackswan-experiments DB and emit (1) experiments/battery.json — the machine-readable evidence (diffable),
// and (2) experiments/BATTERY.html — a single self-contained, shareable static HTML page (the paper's spine).
// The generic report capability lives in modeltrainer (buildBattery / renderBatteryHtml); this file supplies
// the BlackSwan-specific family labels + narrative.
//
//   node experiments/export.mjs
//
// Env: BLACKSWAN_DB_URL / THEFACTORY_DB_DIR / MODELTRAINER_DIR (sibling defaults).

import { createRequire } from 'module'
import { fileURLToPath } from 'url'
import { writeFileSync } from 'fs'
import path from 'path'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const DB_DIR = process.env.THEFACTORY_DB_DIR || path.resolve(HERE, '../../thefactory-db')
const MT_DIR = process.env.MODELTRAINER_DIR || path.resolve(HERE, '../../thefactory-modeltrainer')
const DB_URL = process.env.BLACKSWAN_DB_URL || 'postgresql://thefactory:thefactory@localhost:5435/thefactorydb'
const SCOPE = 'blackswan-experiments'

// type -> human family label, in report order.
const FAMILY = {
  'blackswan-intraday-hypothesis': 'Price — intraday decision-frequency (crypto)',
  'blackswan-riskparity-hypothesis': 'Price — cross-class risk-parity / vol-target (crypto)',
  'blackswan-funding-hypothesis': 'Positioning — perp funding rate (crypto)',
  'blackswan-orderflow-hypothesis': 'Microstructure — taker order-flow imbalance (crypto)',
  'blackswan-events-hypothesis': 'Scheduled events — macro releases (crypto)',
  'blackswan-liquidation-hypothesis': 'Mechanical — liquidation cascades (crypto)',
  'blackswan-regime-hypothesis': 'Macro regime market-timing overlay (crypto)',
  'blackswan-attention-hypothesis': 'Attention — Wikipedia pageviews (crypto)',
  'blackswan-roll-hypothesis': 'Commodity index roll — WTI (Mou 2011 reproduce-and-refute)',
  'blackswan-roll-basket-hypothesis': 'Commodity index roll — pooled energy basket',
  'blackswan-worldmodel-hypothesis': 'Macro world model — gold/silver/copper drivers',
  'blackswan-cot-hypothesis': 'Positioning/flow — CFTC COT (level + Williams index + flow)',
  'blackswan-cross-cot-hypothesis': 'Positioning — cross-sectional COT (market-neutral)',
  'blackswan-tsmom-hypothesis': 'Time-series momentum / trend-following (Moskowitz 2012)',
  'blackswan-xsmom-hypothesis': 'Cross-sectional momentum / reversal (Jegadeesh-Titman 1993)',
  'blackswan-lowvol-hypothesis': 'Low-volatility / Betting-Against-Beta (Frazzini-Pedersen 2014)',
  'blackswan-seasonal-hypothesis': 'Calendar / seasonal (turn-of-month, sell-in-May, Monday)',
  'blackswan-pairs-hypothesis': 'Pairs / statistical arbitrage (Gatev et al. 2006)',
}
const FAMILY_ORDER = Object.values(FAMILY)

const { buildBattery, renderBatteryHtml } = await import(path.join(MT_DIR, 'dist/index.js'))
const { openDatabase } = await import(path.join(DB_DIR, 'dist/index.js'))
const pg = createRequire(path.join(DB_DIR, '/'))('pg')

const c = new pg.Client({ connectionString: DB_URL, connectionTimeoutMillis: 8000 })
await c.connect()
const rows = (await c.query(
  "SELECT external_key, type, content FROM entities WHERE project_id=$1 AND type LIKE 'blackswan-%-hypothesis' AND external_key LIKE 'probe-%' ORDER BY type, external_key",
  [SCOPE],
)).rows
await c.end()

const hypotheses = rows.map((r) => ({
  id: r.external_key,
  type: r.type,
  title: r.content.title,
  status: r.content.status,
  gate: r.content.gate,
  spec: r.content.spec,
  claim: r.content.claim,
  rationale: r.content.rationale,
}))

const battery = buildBattery(hypotheses, { familyOf: (t) => FAMILY[t] ?? t, familyOrder: FAMILY_ORDER })
writeFileSync(path.join(HERE, 'battery.json'), JSON.stringify(battery, null, 2))

// --- BlackSwan-specific narrative (author-controlled HTML sections) ----------------------------------
const sections = [
  {
    heading: 'Thesis (honest, defensible scope)',
    html:
      '<p>Across a broad, <strong>pre-registered</strong> battery of signal families, <strong>no strategy shows a ' +
      'cost-surviving, out-of-sample, multiplicity-corrected trading edge in the freely-obtainable data</strong> ' +
      'over the tested markets (crypto BTC/ETH/SOL; commodity futures GOLD/SILVER/COPPER/WTI + the energy complex; ' +
      'and a survivorship-free diversified daily panel of 7 commodities + SPY/TLT/IEF/UUP) and period (crypto ' +
      '2018–2026; the diversified panel 2006–2026 with 17 out-of-sample windows 2008–2024). Every family was ' +
      'tested with leakage-controlled point-in-time joins (mutation-proven guards), walk-forward out-of-sample ' +
      'windows, per-trade transaction cost, Deflated-Sharpe / best-of-N multiplicity correction, and ' +
      '<strong>independent adversarial verification</strong> of every verdict.</p>' +
      '<p><strong>The honest exceptions</strong> keep this from being an all-null sweep (and make it more ' +
      'credible). Two threads are recorded <strong>inconclusive</strong>, and they are the SAME weak ' +
      'relative-value / <em>mean-reversion</em> effect seen two ways on the commodity-heavy panel: ' +
      'cross-sectional <em>reversal</em> (long past losers / short past winners; annualised Sharpe ≈ +0.43, ' +
      'time-unstable — all in 2016–2024, a coinflip 2008–2015) and <em>pairs mean-reversion</em> (fade ' +
      'divergences; annualised ≈ +0.35, persistent in both sub-periods, cost-surviving to 40 bps). Both are ' +
      'positive and cost-surviving but sub-significant (per-config t &lt; 2, within best-of-N multiplicity), so ' +
      'neither is a bankable edge — they are flagged as open threads for a power-and-replication follow-up. A ' +
      'third, sell-in-May, is a weaker still <em>disproved-marginal</em> tilt (pooled daily t 0.85, carried by ' +
      '~2 of 17 years).</p>',
  },
  {
    heading: 'Published-anomaly battery (the academic canon)',
    html:
      '<p>The paper must confront the classic academic anomalies the literature claims <em>do</em> survive, not ' +
      'only the signals already known to fail. Tested on the free, survivorship-free diversified panel, 17 OOS ' +
      'windows 2008–2024, realistic cost:</p>' +
      '<ul>' +
      '<li><strong>Time-series momentum / trend-following</strong> (Moskowitz-Ooi-Pedersen 2012 — the flagship, the ' +
      'CTA industry): <strong>DISPROVED</strong>, adversarially verified. A textbook <em>post-publication decay</em> ' +
      '(McLean-Pontiff 2016) — pre-2012 annualised Sharpe +0.51 (the edge was real in-sample, and the backtest ' +
      'reproduces the canonical 2008/2010 trend years, a positive control), post-2012 −0.18; full-sample t≈0. ' +
      'Gross ≈ net, so it is a gross null, not a cost kill.</li>' +
      '<li><strong>Cross-sectional momentum</strong> (Jegadeesh-Titman 1993): <strong>DISPROVED — and it inverts</strong>. ' +
      'The published long-winners/short-losers book is significantly <em>negative</em> (−12%/yr) on the free ' +
      'commodity-heavy universe; its mirror (reversal) is the inconclusive open thread above.</li>' +
      '<li><strong>Low-volatility / Betting-Against-Beta</strong> (Frazzini-Pedersen 2014): <strong>DISPROVED</strong> — ' +
      'long low-beta / short high-beta is a coinflip (t≈0, negative net return) across all formation windows ' +
      '(both beta- and volatility-ranked).</li>' +
      '<li><strong>Calendar / seasonal</strong> (turn-of-month, sell-in-May, Monday effect): <strong>DISPROVED</strong> — ' +
      'the exposure-balanced spreads are null-to-negative; the Monday effect is fully decayed (gross t≈0); ' +
      'sell-in-May is a weak disproved-marginal tilt (pooled daily t 0.85).</li>' +
      '<li><strong>Pairs / statistical arbitrage</strong> (Gatev-Goetzmann-Rouwenhorst 2006): <strong>INCONCLUSIVE</strong> — ' +
      'distance-pairs mean-reversion is a weak, persistent, cost-surviving tilt (≈+0.35 Sharpe, 12/17 windows) ' +
      'that does not clear multiplicity-corrected significance; the second mean-reversion open thread.</li>' +
      '</ul>',
  },
  {
    heading: 'What this claim is — and is not',
    html:
      '<p>It is <em>not</em> "no edge exists anywhere." It <em>is</em>: (a) the pre-registered families below ' +
      'fail their declared gate net-of-cost out-of-sample — every one disproved except two honestly-recorded ' +
      'inconclusives (cross-sectional reversal and pairs mean-reversion — the same weak relative-value effect, ' +
      'above), with the search space disclosed; and (b) for the reproduced published claims (Mou 2011 ' +
      'front-running the Goldman roll; and the factor canon — trend-following, cross-sectional momentum, ' +
      'Betting-Against-Beta, the calendar anomalies, and distance pairs-trading), a faithful re-implementation ' +
      'under this discipline does not survive, with the exact methodological hole named (post-publication decay, ' +
      'sign inversion, multiplicity/mirror artifact, or sub-significant relative-value tilt).</p>' +
      '<p><strong>Power caveat.</strong> The Deflated-Sharpe gates at ~252 observations per window reject only ' +
      '<em>large</em> single-window edges — a <em>modest, persistent</em> edge (annualised Sharpe ≈ 0.3–0.8) ' +
      'cannot be excluded. Verdicts therefore rest on directional refutation, hold-underperformance, and ' +
      'beta/multiplicity attribution — not on tight zero-edge confidence intervals.</p>',
  },
  {
    heading: 'Reproduce-and-refute',
    html:
      '<p><strong>Mou (2011), "Front-Running the Goldman Roll"</strong> — reproduced as the WTI M1–M2 ' +
      'calendar-spread probe (<code>probe-roll-frontrun</code>) and the pooled energy basket ' +
      '(<code>probe-roll-basket-frontrun</code>). Under leakage control + cost + multiplicity + OOS it does ' +
      '<strong>not</strong> survive: the per-commodity effect is sign-reproduced but never DSR-significant and is ' +
      'tail-driven; the pooled energy front-run is robustly negative (energy demand-seasonality opposes the roll). ' +
      'Named holes: single-asset underpowering vs the pooled portfolio, and crash-contango tail contamination.</p>',
  },
  {
    heading: 'Recurring failure signature',
    html:
      '<p>Every tempting single-window cell across the battery resolves to one of: <strong>beta</strong> (long a ' +
      'rising asset — self-identifies via ~zero alpha-over-hold), <strong>best-of-N multiplicity</strong> (its ' +
      't-statistic sits at the expected maximum under the null), or a <strong>mechanically-coupled mirror</strong> ' +
      '(the inverse arm carries no independent power). The winning <em>arm</em> tends to track the year’s price ' +
      'regime — i.e. it is price-timing (already null), not a new signal.</p>',
  },
  {
    heading: 'Reproducibility',
    html:
      '<p>Each probe is a <code>trainer/&lt;name&gt;.py</code> CLI + a <code>.factory/trainer-&lt;name&gt;.json</code> ' +
      'manifest, pre-registered and run via <code>experiments/</code> (see <code>registry.mjs</code>); its ' +
      'hypothesis + declared gate + swept search space + adversarial verdict live in the ' +
      '<code>blackswan-experiments</code> trail. Full machine-readable evidence: <code>experiments/battery.json</code>. ' +
      'Regenerate this page with <code>node experiments/export.mjs</code>.</p>',
  },
]

const html = renderBatteryHtml(battery, {
  title: 'BlackSwan — the no-edge battery',
  subtitle: 'A pre-registered, adversarially-verified map of where trading edge is NOT, in freely-obtainable data.',
  generatedNote: 'Generated from the blackswan-experiments pre-registered probe trail — do not hand-edit.',
  sections,
  footerHtml:
    'BlackSwan / thefactory-modeltrainer &middot; evidence battery &middot; ' +
    `${battery.stats.probes} probes · ${battery.stats.families} families · ${battery.stats.cellsRun.toLocaleString()} cells.`,
})
writeFileSync(path.join(HERE, 'BATTERY.html'), html)

console.log('✔ wrote experiments/battery.json + experiments/BATTERY.html')
console.log(`  ${battery.stats.probes} probes, ${battery.stats.families} families, ${battery.stats.cellsRun} cells | ${JSON.stringify(battery.stats.byStatus)}`)
