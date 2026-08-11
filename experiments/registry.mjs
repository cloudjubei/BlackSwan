// The recorded catalogue of BlackSwan side-experiment PROBES — each a pre-registered hypothesis (or two: a
// thesis + a mirror control) with a declared gate and a sweep matrix, run against a .factory trainer manifest.
// This is the committed, reproducible source of truth for the probes; experiments/cli.mjs pre-registers, runs,
// and analyses them via the modeltrainer engine. A probe's `type` yields its DB entity types:
// `${type}-hypothesis` (the pre-registered thesis) and `${type}-experiment` (the campaign's run records).

export const SCOPE = 'blackswan-experiments'

// Standard Deflated-Sharpe gate for a market-neutral / directional strategy judged on its own risk-adjusted
// return, multiplicity-corrected across the per-arm sweep and required in a majority of windows.
const dsrGate = (trials, minWindows = 2) => ({
  kind: 'deflated-sharpe', metric: 'oos_sharpe', threshold: 0, direction: 'max',
  minWindows, windowLever: 'walk_forward_window', trials,
})

// The deep-history OOS window set (train from 2006, one accounted year each) — 17 regime-diverse years
// 2008-2024 (the GFC, 2011, 2015-16, 2018, 2020). A published anomaly must clear the DSR gate in a MAJORITY
// of these, not one lucky period; this is the power the tighter DSR confidence interval needs to reject a
// MODEST persistent edge, not just a large one.
const DEEP_WINDOWS = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']

export const PROBES = {
  // ==================================================================================================
  // PUBLISHED-ANOMALY BATTERY — the academic/practitioner canon the "no-edge" paper must confront (the
  // strategies the literature actively claims DO work), tested on the free survivorship-free daily panel.
  // ==================================================================================================

  // --- TIME-SERIES MOMENTUM / trend-following — the flagship (Moskowitz-Ooi-Pedersen 2012) ------------
  tsmom: {
    type: 'blackswan-tsmom',
    manifest: '.factory/trainer-tsmom.json',
    proposedBy: 'tsmom-probe',
    thesis: 'Time-series momentum / trend-following — diversified basket',
    gate: dsrGate(6, 9),
    fixed: { universe: 'diversified', weight_scheme: 'invvol', rebalance_days: 21, vol_span: 63, transaction_fee: 0.0005 },
    sweep: { lookback: [63, 126, 252], walk_forward_window: DEEP_WINDOWS },
    arms: [
      {
        id: 'probe-tsmom-trend', fixed: { signal: 'trend' },
        title: 'Time-series momentum (long up-trenders / short down-trenders) times a diversified futures basket net of cost',
        claim:
          'Taking each asset in a survivorship-free diversified basket (7 commodities + SPY/TLT/IEF/UUP) LONG when ' +
          'its trailing 3-12m return is positive and SHORT when negative, risk-parity-sized and monthly-rebalanced, ' +
          'produces positive DSR-deflated oos_sharpe in a MAJORITY of the 17 deep-history windows 2008-2024, net of ' +
          'a realistic per-side cost.',
        rationale:
          'The single most-defended free anomaly — the entire managed-futures / CTA industry and Moskowitz-Ooi-' +
          'Pedersen (2012), who report a ~0.8 Sharpe across ~58 futures. This is the flagship the "no-edge" paper ' +
          'must confront. PRE-REGISTERED EXPECTATION: refutation net of cost OOS — trend following endured a long ' +
          'flat-to-negative stretch post-2011, public rules decay (McLean-Pontiff 2016), and a small 11-asset unit-' +
          'gross book at realistic cost is marginal. A SURVIVOR (positive DSR-Sharpe in a majority of 17 regime-' +
          'diverse years, thesis not inverse) would be the program\'s first cost-surviving edge. Sign from past-only ' +
          'trailing return, vol from past-only window, position applied next bar, price-blind, turnover-costed.',
      },
      {
        id: 'probe-tsmom-trend-inverse', fixed: { signal: 'trend_inverse' },
        title: 'The INVERSE trend book (short up-trenders / long down-trenders) times the basket (control)',
        claim: 'The exact sign-negation of the trend book has positive DSR-deflated oos_sharpe in a majority of the 17 windows 2008-2024.',
        rationale:
          'Algebraic-mirror control. If trend is null but the inverse wins across windows, trend REVERSES at this ' +
          'horizon; if both are null, the trend sign carries no cost-surviving directional edge. Pre-registering ' +
          'both prevents post-hoc direction cherry-picking.',
      },
    ],
  },

  // --- CROSS-SECTIONAL MOMENTUM / REVERSAL — paper-grade (Jegadeesh-Titman 1993) ----------------------
  // Supersedes the B1-era xsection nulls (hashed keys 6fff…/f23b…), which ranked SURVIVORSHIP-BIASED
  // megacaps over two windows; this re-tests the same rule on the survivorship-free diversified panel with a
  // realistic cost and the full 17-window DSR gate, so the cross-sectional refutation is paper-grade.
  xsmom: {
    type: 'blackswan-xsmom',
    manifest: '.factory/trainer-xsmom.json',
    proposedBy: 'xsmom-probe',
    thesis: 'Cross-sectional momentum / reversal — survivorship-free diversified basket',
    gate: dsrGate(6, 9),
    fixed: { universe: 'diversified', k: 3, rebalance_days: 21, long_only: false, transaction_fee: 0.0005 },
    sweep: { lookback: [63, 126, 252], walk_forward_window: DEEP_WINDOWS },
    arms: [
      {
        id: 'probe-xsmom-momentum', fixed: { signal: 'momentum' },
        title: 'Cross-sectional momentum (long past winners / short past losers) times a survivorship-free basket net of cost',
        claim:
          'Ranking a survivorship-free diversified basket by trailing 3-12m return and holding a dollar-neutral ' +
          'LONG-top-3 / SHORT-bottom-3 book (monthly rebalance) has positive DSR-deflated oos_sharpe in a MAJORITY ' +
          'of the 17 deep-history windows 2008-2024, net of a realistic per-side cost.',
        rationale:
          'The founding cross-sectional anomaly (Jegadeesh-Titman 1993), the other half of the momentum literature. ' +
          'PRE-REGISTERED EXPECTATION: refutation — cross-sectional momentum is beta-neutral so it removed the ' +
          'single-asset beta artifact, but on a small commodity-heavy basket it becomes a momentum-vs-reversal ' +
          'regime bet whose winning arm tracks the year, and public momentum decayed post-2000 (crashes 2009/2020). ' +
          'Upgrades the survivorship-biased B1 xsection null to a survivorship-free, deep-history, DSR-gated test. ' +
          'Rank from past-only trailing return, book applied next bar, holes never forward-filled, turnover-costed.',
      },
      {
        id: 'probe-xsmom-reversal', fixed: { signal: 'reversal' },
        title: 'Cross-sectional reversal (long past losers / short past winners) times the basket (mirror)',
        claim: 'The exact rank-inversion (long the bottom-3 / short the top-3) has positive DSR-deflated oos_sharpe in a majority of the 17 windows 2008-2024.',
        rationale: 'Algebraic-mirror control. Pre-registering both prevents post-hoc direction cherry-picking, and separates a genuine reversal edge from a momentum one.',
      },
    ],
  },

  // --- LOW-VOLATILITY / BETTING-AGAINST-BETA (Baker-Haugen; Frazzini-Pedersen 2014) -------------------
  lowvol: {
    type: 'blackswan-lowvol',
    manifest: '.factory/trainer-lowvol.json',
    proposedBy: 'lowvol-probe',
    thesis: 'Low-volatility / Betting-Against-Beta — survivorship-free diversified basket',
    gate: dsrGate(12, 9),
    fixed: { universe: 'diversified', k: 3, rebalance_days: 21, transaction_fee: 0.0005 },
    // Both rank keys are swept so the recorded evidence matches the claim's title: rank_by='beta' is
    // Betting-Against-Beta, rank_by='vol' is the low-VOLATILITY anomaly (Baker-Haugen). Persisting both closes
    // the evidence-trail gap the adversarial verification flagged (the vol leg was previously only live-checked).
    sweep: { rank_by: ['beta', 'vol'], span: [63, 126, 252], walk_forward_window: DEEP_WINDOWS },
    arms: [
      {
        id: 'probe-lowvol-lowrisk', fixed: { signal: 'lowrisk' },
        title: 'Betting-Against-Beta (long low-beta / short high-beta) times a survivorship-free basket net of cost',
        claim:
          'Ranking a survivorship-free diversified basket by trailing beta to the equal-weight basket and holding a ' +
          'dollar-neutral LONG-low-beta / SHORT-high-beta book (monthly rebalance) has positive DSR-deflated ' +
          'oos_sharpe in a MAJORITY of the 17 deep-history windows 2008-2024, net of a realistic per-side cost.',
        rationale:
          'The low-risk anomaly (Frazzini-Pedersen 2014 BAB; Baker-Haugen) — a heavily-cited "it survives" claim. ' +
          'PRE-REGISTERED EXPECTATION: refutation — on a small commodity+rates basket the low-beta/high-beta split ' +
          'collapses toward a rates-vs-commodities bet whose sign tracks the regime, and public low-vol crowded/' +
          'decayed. A survivor (positive DSR-Sharpe in a majority of windows, thesis not inverse) would be a first ' +
          'cost-surviving edge. Score past-only, book applied next bar, holes never forward-filled, turnover-costed.',
      },
      {
        id: 'probe-lowvol-lowrisk-inverse', fixed: { signal: 'lowrisk_inverse' },
        title: 'The INVERSE (long high-beta / short low-beta) book times the basket (mirror)',
        claim: 'The exact negation (long high-beta / short low-beta) has positive DSR-deflated oos_sharpe in a majority of the 17 windows 2008-2024.',
        rationale: 'Algebraic-mirror control. Pre-registering both prevents post-hoc direction cherry-picking.',
      },
    ],
  },

  // --- CALENDAR / SEASONAL anomalies (turn-of-month, sell-in-May, Monday effect) ----------------------
  seasonal: {
    type: 'blackswan-seasonal',
    manifest: '.factory/trainer-seasonal.json',
    proposedBy: 'seasonal-probe',
    thesis: 'Calendar / seasonal anomalies — SPY exposure-balanced spread',
    gate: dsrGate(6, 9),
    fixed: { universe: 'spy', tom_days: 3, transaction_fee: 0.0005 },
    sweep: { rule: ['turn_of_month', 'sell_in_may', 'day_of_week'], walk_forward_window: DEEP_WINDOWS },
    arms: [
      {
        id: 'probe-seasonal-seasonal', fixed: { signal: 'seasonal' },
        title: 'The classic calendar effects (turn-of-month / sell-in-May / Monday) time SPY net of cost',
        claim:
          'Trading the exposure-balanced calendar spread on SPY — long the claimed-good window (turn-of-month, ' +
          'Nov-Apr, or Tue-Fri), short its complement, time-neutral — has positive DSR-deflated oos_sharpe in a ' +
          'MAJORITY of the 17 deep-history windows 2008-2024, net of cost.',
        rationale:
          'The famous free calendar anomalies (Lakonishok-Smidt 1988; Bouman-Jacobsen 2002; French 1980). ' +
          'PRE-REGISTERED EXPECTATION: refutation — these are decades-old, heavily publicised, and largely ' +
          'decayed/arbitraged (McLean-Pontiff 2016); the exposure-balanced spread removes the market-drift ' +
          'confound, so a win must be a genuine seasonal differential. A survivor across a majority of windows ' +
          '(thesis not inverse) would be a first cost-surviving edge. The book is a pure function of the calendar ' +
          '(never price; mutation-proven), turnover-costed.',
      },
      {
        id: 'probe-seasonal-inverse', fixed: { signal: 'seasonal_inverse' },
        title: 'The INVERSE calendar spread (long the bad window / short the good) times SPY (mirror)',
        claim: 'The exact negation of the calendar spread has positive DSR-deflated oos_sharpe in a majority of the 17 windows 2008-2024.',
        rationale: 'Algebraic-mirror control. Pre-registering both prevents post-hoc direction cherry-picking.',
      },
    ],
  },

  // --- PAIRS / STATISTICAL ARBITRAGE (distance pairs — Gatev-Goetzmann-Rouwenhorst 2006) --------------
  pairs: {
    type: 'blackswan-pairs',
    manifest: '.factory/trainer-pairs.json',
    proposedBy: 'pairs-probe',
    thesis: 'Pairs / statistical arbitrage — diversified basket distance pairs',
    gate: dsrGate(6, 9),
    fixed: { universe: 'diversified', formation_days: 252, k: 5, transaction_fee: 0.0005 },
    sweep: { entry: [1.5, 2.0, 2.5], walk_forward_window: DEEP_WINDOWS },
    arms: [
      {
        id: 'probe-pairs-meanrev', fixed: { signal: 'meanrev' },
        title: 'Distance pairs trading (fade divergence, close on convergence) times the basket net of cost',
        claim:
          'Selecting the 5 closest-moving pairs of a survivorship-free diversified basket on a 12-month formation ' +
          'window and fading their divergences (short the rich leg / long the cheap leg beyond `entry` formation ' +
          'std-devs, close on convergence) has positive DSR-deflated oos_sharpe in a MAJORITY of the 17 deep-history ' +
          'windows 2008-2024, net of cost.',
        rationale:
          'The classic distance stat-arb (Gatev-Goetzmann-Rouwenhorst 2006) — a distinct MECHANISM (spread ' +
          'mean-reversion, market-neutral), not price direction. PRE-REGISTERED EXPECTATION: refutation — GGR\'s ' +
          'edge decayed sharply post-2002 as it was arbitraged, and a small daily free basket has few genuinely-' +
          'cointegrated pairs (cross-asset "pairs" like gold-vs-bonds are economically unrelated and diverge ' +
          'permanently). A survivor across a majority of windows (thesis not inverse) would be a first cost-surviving ' +
          'edge. Pair selection + spread mean/std past-only, book one bar behind, dollar-neutral, turnover-costed.',
      },
      {
        id: 'probe-pairs-meanrev-inverse', fixed: { signal: 'meanrev_inverse' },
        title: 'The INVERSE (chase divergence) pairs book times the basket (mirror)',
        claim: 'The exact negation (chase divergence instead of fading it) has positive DSR-deflated oos_sharpe in a majority of the 17 windows 2008-2024.',
        rationale: 'Algebraic-mirror control. Pre-registering both prevents post-hoc direction cherry-picking.',
      },
    ],
  },

  // --- BALTUSSEN 2021 "Global Factor Premiums" replication — THE FOIL (the paper's spine) --------------
  // cf-* windows train from 2006 so the 5yr VALUE reversal always has its formation. Reuses tsmom/xsection/
  // lowvol build_weights; the diversified equal-risk average is the headline foil (Baltussen's central claim).
  globalfactors: {
    type: 'blackswan-globalfactors',
    manifest: '.factory/trainer-globalfactors.json',
    proposedBy: 'globalfactors-probe',
    thesis: 'Global factor premiums (Baltussen 2021) — the multi-asset foil',
    gate: dsrGate(10, 7),
    fixed: { universe: 'diversified', lookback: 252, value_lookback: 1260, span: 252, k: 3, rebalance_days: 21, vol_span: 63, transaction_fee: 0.0005 },
    sweep: { factor: ['trend', 'momentum', 'value', 'lowbeta', 'diversified'], walk_forward_window: ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', 'cf-2020', 'cf-2021', 'cf-2022', 'cf-2023', 'cf-2024'] },
    arms: [
      {
        id: 'probe-globalfactors-published', fixed: { signal: 'published' },
        title: 'The Baltussen (2021) global style premiums (and their diversified combination) survive our discipline net of cost',
        claim:
          'Baltussen-Swinkels-van Vliet (2021) report trend / momentum / value / low-beta (and their diversified ' +
          'equal-risk combination) as ROBUST multi-asset premiums. Reproduced on the free survivorship-free panel ' +
          'and subjected to realistic per-trade cost + DSR/best-of-N multiplicity + point-in-time guards + ' +
          'deep-history OOS, at least the DIVERSIFIED combination has positive DSR-deflated oos_sharpe in a MAJORITY ' +
          'of the 13 windows.',
        rationale:
          'THE FOIL — the one prior human multi-asset unified test, and the direct contradiction that is the paper\'s ' +
          'spine. PRE-REGISTERED EXPECTATION: the factors and their diversified combination FAIL net of cost OOS, ' +
          'because Baltussen\'s robustness rests on low-turnover, index-level, cost-free construction dominated by ' +
          'pre-1980 history, whereas our 2012-2024 window sits inside the post-publication crowded-decay regime with ' +
          'realistic frictions. If the diversified combination survives, Baltussen holds on free data and the paper\'s ' +
          'contradiction fails — an honest, pre-registered decider. Reuses mutation-proven build_weights; value = 5yr ' +
          'reversal; diversified = unit-gross equal-risk average, applied next bar, turnover-costed.',
      },
      {
        id: 'probe-globalfactors-inverse', fixed: { signal: 'inverse' },
        title: 'The INVERSE of the Baltussen factors (and their combination) times the panel (mirror)',
        claim: 'The exact negation of each factor has positive DSR-deflated oos_sharpe in a majority of the 13 windows.',
        rationale: 'Algebraic-mirror control. Pre-registering both prevents post-hoc direction cherry-picking.',
      },
    ],
  },

  // --- commodity index-roll ("Goldman roll"), single-asset WTI (DISPROVED — recorded) ----------------
  roll: {
    type: 'blackswan-roll',
    manifest: '.factory/trainer-roll.json',
    proposedBy: 'commodity-roll-probe',
    thesis: 'Commodity index-roll (Goldman roll) front-running on WTI',
    gate: dsrGate(9),
    fixed: { transaction_cost: 0.0003, roll_schedule: 'gsci' },
    sweep: { asset: ['WTI'], entry_bday: [2, 3, 4], exit_bday: [9, 10, 11], walk_forward_window: ['roll-2006-2011', 'roll-2016-2025'] },
    arms: [
      { id: 'probe-roll-frontrun', fixed: { signal: 'roll_frontrun' }, title: 'Front-running the GSCI roll (short the M1-M2 spread) clears cost on WTI',
        claim: 'Shorting the WTI M1-M2 spread into the roll has positive DSR-deflated Sharpe net of cost in both windows.',
        rationale: 'Reproduce-and-refute of Mou (2011). Recorded probe; see the DB trail for the full verdict (DISPROVED).' },
      { id: 'probe-roll-fade', fixed: { signal: 'roll_fade' }, title: 'Fading the GSCI roll (long the M1-M2 spread) clears cost on WTI',
        claim: 'Longing the WTI M1-M2 spread into the roll has positive DSR-deflated Sharpe net of cost in both windows.',
        rationale: 'Algebraic-mirror control of the front-run arm. Recorded probe (DISPROVED).' },
    ],
  },

  // --- pooled energy-basket index-roll (DISPROVED — recorded) -----------------------------------------
  'roll-basket': {
    type: 'blackswan-roll-basket',
    manifest: '.factory/trainer-roll-basket.json',
    proposedBy: 'commodity-roll-basket-probe',
    thesis: 'Pooled energy-basket index-roll',
    gate: dsrGate(9),
    fixed: { transaction_cost: 0.0003, roll_schedule: 'gsci', basket: 'energy4' },
    sweep: { entry_bday: [2, 3, 4], exit_bday: [9, 10, 11], walk_forward_window: ['roll-2006-2011', 'roll-2016-2025'] },
    arms: [
      { id: 'probe-roll-basket-frontrun', fixed: { signal: 'roll_frontrun' }, title: 'Front-running the GSCI roll across the pooled energy basket clears DSR-deflated Sharpe',
        claim: 'An equal-weight energy-basket short-spread portfolio has positive DSR-deflated Sharpe net of cost in both windows.',
        rationale: 'Power-recoverable follow-up to the WTI roll null. Recorded probe (DISPROVED — energy front-run only).' },
      { id: 'probe-roll-basket-fade', fixed: { signal: 'roll_fade' }, title: 'Fading the GSCI roll across the pooled energy basket clears DSR-deflated Sharpe',
        claim: 'An equal-weight energy-basket long-spread portfolio has positive DSR-deflated Sharpe net of cost in both windows.',
        rationale: 'Algebraic-mirror control. Recorded probe (DISPROVED).' },
    ],
  },

  // --- GOLD macro world model (the driver-conditioned timing probe) ------------------------------------
  'gold-worldmodel': {
    type: 'blackswan-worldmodel',
    manifest: '.factory/trainer-worldmodel.json',
    proposedBy: 'gold-worldmodel-probe',
    thesis: 'Gold macro world model — driver-conditioned timing',
    gate: dsrGate(9),
    fixed: { asset: 'GOLD', driver_set: 'gold_macro3', transaction_fee: 0.0005 },
    sweep: { lookback: [21, 63, 126], walk_forward_window: ['2022', '2023', '2024'] },
    arms: [
      {
        id: 'probe-gold-worldmodel', fixed: { signal: 'worldmodel' },
        title: 'The canonical gold macro world model TIMES gold (positive DSR-deflated Sharpe across regimes)',
        claim:
          'Conditioning a long/short/flat GOLD position on its three canonical macro drivers — 10y real rate ' +
          '(DFII10) FALLING, broad dollar (DTWEXBGS) FALLING, 10y inflation breakeven (DGS10-DFII10) RISING, each ' +
          'a past-only trend — produces POSITIVE DSR-deflated risk-adjusted return net of cost, robustly across ' +
          'the 2022 (rate-hike headwind) and 2024 (gold-bull) regimes.',
        rationale:
          'First WORLD-MODEL probe (the B4 line, generalised beyond crypto) — the honest, disciplined version of ' +
          '"model what moves gold and predict its price". PRE-REGISTERED EXPECTATION: FAIL. Gold\'s real-rate/USD ' +
          'sensitivity is textbook and heavily arbitraged, and 2024 is a vivid regime where gold DECOUPLED from its ' +
          'macro drivers (it rallied ~+27% into rising real rates / a strong dollar, driven by central-bank + ' +
          'geopolitical demand the macro world model cannot see). A survivor (positive DSR-Sharpe in BOTH regimes, ' +
          'thesis not inverse) would be the program\'s first cost-surviving edge. Drivers joined point-in-time ' +
          '(pit_fusion), past-only trend, position applied next bar, price-blind. Gold\'s secular uptrend makes ' +
          'return_vs_hold exposure-biased, so the decider is the model\'s own DSR-deflated oos_sharpe.',
      },
      {
        id: 'probe-gold-worldmodel-inverse', fixed: { signal: 'worldmodel_inverse' },
        title: 'The INVERSE gold macro world model times gold (control)',
        claim:
          'Taking the OPPOSITE of the driver-implied side produces positive DSR-deflated Sharpe net of cost across ' +
          'both regimes — i.e. the drivers carry real directional content, just inverted.',
        rationale:
          'Mirror control. If the thesis is null but the inverse wins across regimes, the drivers lead gold the ' +
          'opposite way; if both are null, the macro world model carries no cost-surviving directional edge for gold. ' +
          'Pre-registering both prevents post-hoc direction cherry-picking.',
      },
    ],
  },

  // --- GOLD world model, FAIR TEST (de-collinearised drivers + the excluded 2020-21 regime) ------------
  'gold-worldmodel-fair': {
    type: 'blackswan-worldmodel',
    manifest: '.factory/trainer-worldmodel.json',
    proposedBy: 'gold-worldmodel-fair-probe',
    thesis: 'Gold macro world model — FAIR test (de-collinearised + 2020-24)',
    gate: dsrGate(15, 3),
    fixed: { asset: 'GOLD', driver_set: 'gold_macro3b', transaction_fee: 0.0005 },
    sweep: { lookback: [21, 63, 126], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: 'probe-gold-worldmodel-fair', fixed: { signal: 'worldmodel' },
        title: 'The FAIR (de-collinearised) gold macro world model times gold across 2020-24',
        claim:
          'The de-collinearised gold world model (real rate DFII10↓, USD DTWEXBGS↓, breakeven from the STANDALONE ' +
          'T10YIE↑ — DFII10 no longer double-counted) produces positive DSR-deflated Sharpe in a MAJORITY of the ' +
          'five windows 2020-2024, INCLUDING the 2020-21 real-rate-collapse regime the first probe excluded.',
        rationale:
          'Fair-test follow-up to probe-gold-worldmodel (INCONCLUSIVE), addressing both defects the adversarial ' +
          'verification named: (1) DE-COLLINEARISE — breakeven now the standalone FRED T10YIE series, so DFII10 ' +
          'appears once (the composite is no longer a double-counted real-rate trend follower); (2) ADD the ' +
          'FAVORABLE 2020-21 regime (real rates crashed, gold rose — where the signal SHOULD go long and win). ' +
          'PRE-REGISTERED EXPECTATION: if the world model is a FAIR-WEATHER signal it wins 2020-21 and loses the ' +
          '2023-24 decoupling regime — regime-dependent, NOT a robust edge; a real edge must clear DSR-Sharpe in a ' +
          'majority of the five windows. Point-in-time (pit_fusion, DTWEXBGS lag now corrected), past-only, ' +
          'next-bar, price-blind.',
      },
      {
        id: 'probe-gold-worldmodel-fair-inverse', fixed: { signal: 'worldmodel_inverse' },
        title: 'The INVERSE fair gold world model times gold (control)',
        claim: 'The opposite of the fair driver-implied side clears DSR-Sharpe in a majority of 2020-24 windows.',
        rationale: 'Mechanically-coupled control (carries no independent power); pre-registered to prevent direction cherry-picking.',
      },
    ],
  },

  // --- GOLD real-rate-ALONE timer (closes the "composite masked a single driver" escape hatch) ---------
  'gold-realrate': {
    type: 'blackswan-worldmodel',
    manifest: '.factory/trainer-worldmodel.json',
    proposedBy: 'gold-realrate-probe',
    thesis: 'Gold real-rate-alone timer',
    gate: dsrGate(15, 3),
    fixed: { asset: 'GOLD', driver_set: 'gold_realrate1', transaction_fee: 0.0005 },
    sweep: { lookback: [21, 63, 126], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: 'probe-gold-realrate', fixed: { signal: 'worldmodel' },
        title: 'The real-rate-ALONE gold timer (long when DFII10 falls) times gold across 2020-24',
        claim:
          'Timing GOLD on the single canonical driver — long when the 10y real rate (DFII10) is trending down, short ' +
          'when up — produces positive DSR-deflated Sharpe in a majority of the five windows 2020-2024, INCLUDING the ' +
          '2020-21 real-rate-collapse regime where a real-rate timer should most clearly win.',
        rationale:
          'The cheap missing arm the fair-test verification demanded: it isolates the real rate so the equal-weight ' +
          'majority vote cannot mask a single predictive driver. If real-rate-alone ALSO nulls in 2020-21, the ' +
          'composite null generalises to the strongest gold channel and the gold "no macro timing" claim is earned; ' +
          'if it survives, the composite was diluting a real signal. PRE-REGISTERED EXPECTATION: null (gold decoupled ' +
          'from real rates 2023-24, and even the 2020-21 collapse was too noisy/lagged for a trend timer).',
      },
      {
        id: 'probe-gold-realrate-inverse', fixed: { signal: 'worldmodel_inverse' },
        title: 'The INVERSE real-rate gold timer (control)',
        claim: 'The opposite of the real-rate-implied side clears DSR-Sharpe in a majority of 2020-24 windows.',
        rationale: 'Mechanically-coupled control; pre-registered to prevent direction cherry-picking.',
      },
    ],
  },

  // --- SILVER world model (precious-metal monetary drivers, same as gold-fair) -------------------------
  'silver-worldmodel': {
    type: 'blackswan-worldmodel',
    manifest: '.factory/trainer-worldmodel.json',
    proposedBy: 'silver-worldmodel-probe',
    thesis: 'Silver macro world model',
    gate: dsrGate(15, 3),
    fixed: { asset: 'SILVER', driver_set: 'silver_macro3', transaction_fee: 0.0005 },
    sweep: { lookback: [21, 63, 126], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: 'probe-silver-worldmodel', fixed: { signal: 'worldmodel' },
        title: 'The precious-metal macro world model times SILVER across 2020-24',
        claim:
          'Conditioning a long/short/flat SILVER position on the precious-metal monetary drivers (real rate DFII10↓, ' +
          'USD DTWEXBGS↓, breakeven T10YIE↑) produces positive DSR-deflated Sharpe in a majority of the 2020-24 windows.',
        rationale:
          'Extension to silver — a precious metal driven by the same monetary channel as gold, plus an industrial ' +
          'component the macro set does not capture. PRE-REGISTERED EXPECTATION: FAIL / regime-dependent like gold, ' +
          'and noisier (silver is higher-beta). Honest recording of whether the same drivers time a second metal.',
      },
      {
        id: 'probe-silver-worldmodel-inverse', fixed: { signal: 'worldmodel_inverse' },
        title: 'The INVERSE silver world model (control)',
        claim: 'The opposite of the driver-implied side clears DSR-Sharpe in a majority of 2020-24 windows.',
        rationale: 'Mechanically-coupled control; pre-registered to prevent direction cherry-picking.',
      },
    ],
  },

  // --- CFTC COT positioning (the flow/positioning class) — one probe per metal -------------------------
  ...Object.fromEntries(['GOLD', 'SILVER', 'COPPER'].map((asset) => [`cot-${asset.toLowerCase()}`, {
    type: 'blackswan-cot',
    manifest: '.factory/trainer-cot.json',
    proposedBy: 'cot-probe',
    thesis: `CFTC COT managed-money positioning — ${asset}`,
    gate: dsrGate(10, 3),
    fixed: { asset, cot_pct: 0.90, release_lag_days: 4, transaction_fee: 0.0005 },
    sweep: { hold_bars: [10, 21], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: `probe-cot-${asset.toLowerCase()}-contrarian`, fixed: { signal: 'cot_contrarian' },
        title: `Fading a crowded COT managed-money extreme times ${asset}`,
        claim:
          `Shorting ${asset} when speculative net positioning is at a top-tail (crowded-long) extreme of its ` +
          `past-only distribution — and long on a crowded-short extreme — has positive DSR-deflated Sharpe (and ` +
          `positive per-trade signal_expectancy) in a majority of the 2020-24 windows, i.e. crowded specs unwind.`,
        rationale:
          `POSITIONING/FLOW probe — the one signal class the program had not tested, and the "flows that actually ` +
          `moved gold" the metals-worldmodel verification flagged. Real structural story: crowded speculators are ` +
          `FORCED unwinders on a reversal. PRE-REGISTERED EXPECTATION: FAIL — COT is public + weekly-lagged + a ` +
          `well-known retail contrarian screen (so arbitraged), and fading a metal in a secular bull fights the ` +
          `trend. A survivor across a majority of five regime-diverse years would be the first cost-surviving ` +
          `edge. Release-lagged (report Tue usable Mon after Fri), past-only extreme, next-bar, price-blind.`,
      },
      {
        id: `probe-cot-${asset.toLowerCase()}-momentum`, fixed: { signal: 'cot_momentum' },
        title: `Riding a crowded COT managed-money extreme times ${asset}`,
        claim:
          `Going long ${asset} on a crowded-long positioning extreme (and short on crowded-short) has positive ` +
          `DSR-deflated Sharpe in a majority of the 2020-24 windows — spec positioning leads price.`,
        rationale:
          `The opposite reading (positioning as trend confirmation). Pre-registering both arms prevents post-hoc ` +
          `direction cherry-picking. Same probe-first discipline. PRE-REGISTERED EXPECTATION: FAIL.`,
      },
    ],
  }])),

  // --- CFTC COT INDEX (trailing Williams 3yr min-max — fixes the expanding-quantile flaw) --------------
  ...Object.fromEntries(['GOLD', 'SILVER', 'COPPER'].map((asset) => [`cotidx-${asset.toLowerCase()}`, {
    type: 'blackswan-cot',
    manifest: '.factory/trainer-cot.json',
    proposedBy: 'cot-index-probe',
    thesis: `CFTC COT INDEX (trailing 3yr) — ${asset}`,
    gate: dsrGate(10, 3),
    fixed: { asset, cot_pct: 0.90, release_lag_days: 4, cot_index_window: 756, min_history: 126, transaction_fee: 0.0005 },
    sweep: { hold_bars: [10, 21], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: `probe-cotidx-${asset.toLowerCase()}-contrarian`, fixed: { signal: 'cot_contrarian' },
        title: `Fading a trailing-3yr COT-INDEX extreme times ${asset}`,
        claim:
          `The canonical Williams COT INDEX (managed-money net min-max normalised over a trailing ~3yr window, ` +
          `regime-adaptive) at a top extreme -> SHORT ${asset} (crowded longs unwind) has positive DSR-deflated ` +
          `Sharpe in a majority of the 2020-24 windows.`,
        rationale:
          `The trailing-window fix the COT (expanding-quantile) verification demanded: it adapts to regime and ` +
          `FIRES even when positioning is nowhere near an all-time extreme — closing the gold 2020/2021 no-trade ` +
          `hole where the expanding quantile (anchored to the 2010-11 mania) was structurally unreachable. This is ` +
          `the honest re-test of the positioning-extreme sub-class. PRE-REGISTERED EXPECTATION: FAIL (it is THE ` +
          `most published retail positioning screen, so the most arbitraged). Release-lagged, past-only, next-bar.`,
      },
      {
        id: `probe-cotidx-${asset.toLowerCase()}-momentum`, fixed: { signal: 'cot_momentum' },
        title: `Riding a trailing-3yr COT-INDEX extreme times ${asset}`,
        claim: `A trailing COT-INDEX top extreme -> LONG ${asset} has positive DSR-deflated Sharpe in a majority of the 2020-24 windows.`,
        rationale: `The opposite reading. Pre-registering both arms prevents post-hoc direction cherry-picking. PRE-REGISTERED EXPECTATION: FAIL.`,
      },
    ],
  }])),

  // --- CFTC COT FLOW/CHANGE (weekly delta — the last untested positioning variant) ---------------------
  ...Object.fromEntries(['GOLD', 'SILVER', 'COPPER'].map((asset) => [`cotflow-${asset.toLowerCase()}`, {
    type: 'blackswan-cot',
    manifest: '.factory/trainer-cot.json',
    proposedBy: 'cot-flow-probe',
    thesis: `CFTC COT FLOW (weekly change) — ${asset}`,
    gate: dsrGate(10, 3),
    fixed: { asset, cot_pct: 0.90, release_lag_days: 4, cot_flow_lag: 5, transaction_fee: 0.0005 },
    sweep: { hold_bars: [10, 21], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: `probe-cotflow-${asset.toLowerCase()}-momentum`, fixed: { signal: 'cot_momentum' },
        title: `Riding the COT FLOW (managed money buying hard) times ${asset}`,
        claim:
          `A top-tail extreme in the WEEKLY CHANGE of managed-money net (specs buying hard) -> LONG ${asset} ` +
          `(riding the flow) has positive DSR-deflated Sharpe in a majority of the 2020-24 windows — positioning ` +
          `FLOW leads price where the level does not.`,
        rationale:
          `The last untested variant of the positioning class (the "flow" half): FLOW answers which way smart ` +
          `money is MOVING, distinct from how extreme its level is, and much of the academic COT ` +
          `return-predictability literature lives in CHANGES not levels. PRE-REGISTERED EXPECTATION: FAIL — ` +
          `managed money is a CTA/trend-following cohort, so weekly delta-in-net is highly COLLINEAR with price ` +
          `momentum (already comprehensively null), so a "win" concentrated in trend years is beta, not a COT ` +
          `edge. Running it to formally CLOSE the positioning class. Release-lagged, past-only, next-bar, price-blind.`,
      },
      {
        id: `probe-cotflow-${asset.toLowerCase()}-contrarian`, fixed: { signal: 'cot_contrarian' },
        title: `Fading the COT FLOW (late buyers) times ${asset}`,
        claim: `A top-tail flow extreme -> SHORT ${asset} (fade the late buyers) has positive DSR-deflated Sharpe in a majority of the 2020-24 windows.`,
        rationale: `The opposite reading. Pre-registering both arms prevents post-hoc direction cherry-picking. PRE-REGISTERED EXPECTATION: FAIL.`,
      },
    ],
  }])),

  // --- CROSS-ASSET-CLASS COT (equity / rates — the new universe, weaker CTA-collinearity) --------------
  ...Object.fromEntries([['SPY', 'equity (E-mini S&P 500)'], ['IEF', '10y Treasury notes'], ['TLT', 'long Treasury bonds']].map(([a, label]) => [`cotfin-${a.toLowerCase()}`, {
    type: 'blackswan-cot',
    manifest: '.factory/trainer-cot.json',
    proposedBy: 'cot-financial-probe',
    thesis: `Cross-asset COT (TFF leveraged funds) — ${a} ${label}`,
    gate: dsrGate(12, 7),
    fixed: { asset: a, cot_pct: 0.90, cot_index_window: 756, min_history: 126, release_lag_days: 4, transaction_fee: 0.0005, hold_bars: 10 },
    sweep: { walk_forward_window: ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'] },
    arms: [
      {
        id: `probe-cotfin-${a.toLowerCase()}-contrarian`, fixed: { signal: 'cot_contrarian' },
        title: `Fading a crowded leveraged-funds COT-index extreme times ${a} (${label})`,
        claim:
          `Fading a top-tail leveraged-funds positioning extreme (trailing COT index) on ${label} has positive ` +
          `DSR-deflated Sharpe in a majority of the 12 windows 2010-2021 — the cross-asset universe where ` +
          `speculators are less purely trend-following, so positioning may carry an edge the commodity class did not.`,
        rationale:
          `The cross-ASSET-CLASS positioning thread (the one genuinely-different free universe after the commodity ` +
          `managed-money class closed). Equity/rate leveraged-funds include real hedgers / risk-parity / vol-control, ` +
          `so the CTA-collinearity-with-price defeater is WEAKER than for commodity managed money. PRE-REGISTERED ` +
          `EXPECTATION: FAIL — public, lagged, widely-watched positioning is arbitraged. TFF leveraged funds, ` +
          `release-lagged, past-only, next-bar, price-blind. COT coverage 2006-2022.`,
      },
      {
        id: `probe-cotfin-${a.toLowerCase()}-momentum`, fixed: { signal: 'cot_momentum' },
        title: `Riding a crowded leveraged-funds COT-index extreme times ${a} (${label})`,
        claim: `Riding a top-tail leveraged-funds extreme on ${label} has positive DSR-deflated Sharpe in a majority of the 12 windows 2010-2021.`,
        rationale: `The mirror arm. Pre-registering both prevents post-hoc direction cherry-picking. PRE-REGISTERED EXPECTATION: FAIL.`,
      },
    ],
  }])),

  // --- CROSS-SECTIONAL COT (relative value, market-neutral — the structurally-different thread) ---------
  'cross-cot': {
    type: 'blackswan-cross-cot',
    manifest: '.factory/trainer-cross-cot.json',
    proposedBy: 'cross-cot-probe',
    thesis: 'Cross-sectional COT long/short — complex6',
    gate: dsrGate(10, 3),
    fixed: { basket: 'complex6', cot_index_window: 756, min_history: 126, release_lag_days: 4, transaction_fee: 0.0005 },
    sweep: { cross_k: [2, 3], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: 'probe-cross-cot-contrarian', fixed: { signal: 'cross_contrarian' },
        title: 'A market-neutral cross-sectional COT long/short (long least-crowded, short most-crowded) times the commodity complex',
        claim:
          'Ranking GOLD/SILVER/COPPER/WTI/CORN/WHEAT by managed-money positioning extremity (trailing COT index) ' +
          'and holding a dollar-neutral LONG-least-crowded / SHORT-most-crowded book has positive DSR-deflated ' +
          'Sharpe in a majority of the 2020-24 windows — a genuine RELATIVE-positioning edge that survives where ' +
          'the single-asset (beta-laden) positioning probes could not.',
        rationale:
          'The structurally-different positioning thread the COT verification named: a CROSS-SECTIONAL (relative ' +
          'value) construction, market-neutral across metals+energy+ags, so the common commodity beta that made ' +
          'every single-asset tempting cell a beta artifact CANCELS. If any real positioning edge exists, it must ' +
          'show here. PRE-REGISTERED EXPECTATION: FAIL — the cross-sectional signal is likely a momentum-vs-value ' +
          'regime bet (cross-sectional momentum rides trending commodities, contrarian mean-reverts), so the ' +
          'winning ARM tracks the year\'s regime and neither survives a majority of windows DSR-corrected. Release-' +
          'lagged, past-only, next-bar, price-blind.',
      },
      {
        id: 'probe-cross-cot-momentum', fixed: { signal: 'cross_momentum' },
        title: 'A market-neutral cross-sectional COT long/short (long most-crowded, short least-crowded) times the commodity complex',
        claim:
          'The reverse ranking (long the most-crowded / short the least-crowded — cross-sectional positioning ' +
          'momentum) has positive DSR-deflated Sharpe in a majority of the 2020-24 windows.',
        rationale: 'The mirror arm. Pre-registering both prevents post-hoc direction cherry-picking. PRE-REGISTERED EXPECTATION: FAIL.',
      },
    ],
  },

  // --- COPPER world model (financial-conditions only — deliberately incomplete) ------------------------
  'copper-worldmodel': {
    type: 'blackswan-worldmodel',
    manifest: '.factory/trainer-worldmodel.json',
    proposedBy: 'copper-worldmodel-probe',
    thesis: 'Copper financial-conditions world model',
    gate: dsrGate(15, 3),
    fixed: { asset: 'COPPER', driver_set: 'copper_macro3', transaction_fee: 0.0005 },
    sweep: { lookback: [21, 63, 126], walk_forward_window: ['2020', '2021', '2022', '2023', '2024'] },
    arms: [
      {
        id: 'probe-copper-worldmodel', fixed: { signal: 'worldmodel' },
        title: 'A financial-conditions macro model times COPPER across 2020-24',
        claim:
          'Conditioning a long/short/flat COPPER position on financial-conditions drivers (USD DTWEXBGS↓, real rate ' +
          'DFII10↓, growth curve T10Y2Y↑) produces positive DSR-deflated Sharpe in a majority of the 2020-24 windows.',
        rationale:
          'Extension to copper — but DELIBERATELY INCOMPLETE: copper ("Dr. Copper") is INDUSTRIAL, and its real ' +
          'drivers (China demand, LME inventories, global manufacturing) are NOT in the mined macro set; this tests ' +
          'only what FINANCIAL macro (dollar, rates, curve) sees. PRE-REGISTERED EXPECTATION: FAIL — the model is ' +
          'missing copper\'s actual drivers; an honest recording of the LIMIT of macro-only copper timing, and the ' +
          'motivation to mine industrial data if it is ever worth pursuing.',
      },
      {
        id: 'probe-copper-worldmodel-inverse', fixed: { signal: 'worldmodel_inverse' },
        title: 'The INVERSE copper financial-conditions model (control)',
        claim: 'The opposite of the driver-implied side clears DSR-Sharpe in a majority of 2020-24 windows.',
        rationale: 'Mechanically-coupled control; pre-registered to prevent direction cherry-picking.',
      },
    ],
  },
}
