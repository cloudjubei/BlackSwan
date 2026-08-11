// Shared machinery for the BlackSwan experiment runner: resolve the modeltrainer engine + the DB, and
// pre-register / run / analyse a probe from experiments/registry.mjs. Paths resolve to sibling repos by
// default and are overridable by env (MODELTRAINER_DIR, THEFACTORY_DB_DIR, BLACKSWAN_DB_URL) so the tooling is
// portable and CI-friendly. The modeltrainer engine (node/TS) is BlackSwan's side-experiment substrate; this
// file is the thin, committed BlackSwan-side glue that records + runs the probes.

import { createRequire } from 'module'
import { fileURLToPath } from 'url'
import path from 'path'
import { SCOPE } from './registry.mjs'

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const MT_DIR = process.env.MODELTRAINER_DIR || path.resolve(REPO_ROOT, '../thefactory-modeltrainer')
const DB_DIR = process.env.THEFACTORY_DB_DIR || path.resolve(REPO_ROOT, '../thefactory-db')
const DB_URL = process.env.BLACKSWAN_DB_URL || 'postgresql://thefactory:thefactory@localhost:5435/thefactorydb'

const hypType = (probe) => `${probe.type}-hypothesis`
const expType = (probe) => `${probe.type}-experiment`
const NOW = () => new Date().toISOString()

async function _imports() {
  const mt = await import(path.join(MT_DIR, 'dist/index.js'))
  const tools = await import(path.join(MT_DIR, 'node_modules/thefactory-tools/dist/index.js'))
  const db = await import(path.join(DB_DIR, 'dist/index.js'))
  const require = createRequire(path.join(DB_DIR, '/'))
  return { mt, tools, db, pg: require('pg') }
}

export async function openEngine() {
  const { mt, tools, db, pg } = await _imports()
  const database = await db.openDatabase({ connectionString: DB_URL, migrations: 'skip', logLevel: 'error' })
  const engine = mt.createModelTrainerTools({
    computeRunner: new tools.LocalComputeRunner(),
    storage: new tools.DbDataStorage(database),
    logger: { info: () => {}, warn: (m, x) => console.warn('  [warn]', m, x ? JSON.stringify(x) : '') },
    availableParallelism: () => 8,
  })
  const client = new pg.Client({ connectionString: DB_URL, connectionTimeoutMillis: 8000 })
  await client.connect()
  return { engine, database, client, close: async () => { await client.end(); await database.close() } }
}

function armMatrix(probe, arm) {
  return { fixed: { ...probe.fixed, ...arm.fixed }, sweep: probe.sweep }
}

function cellCount(probe) {
  return Object.values(probe.sweep).reduce((n, vals) => n * vals.length, 1)
}

// Pre-register a probe's hypotheses (idempotent-safe: an already-recorded hypothesis is SKIPPED, never
// overwritten, so a materialised verdict is protected — pass force:true to re-seed to untested).
export async function preregister(probe, { force = false, apply = false } = {}) {
  const { database, client, close } = await openEngine()
  const at = NOW()
  try {
    for (const arm of probe.arms) {
      const existing = await client.query('SELECT content FROM entities WHERE project_id=$1 AND external_key=$2', [SCOPE, arm.id])
      const status = existing.rows[0]?.content?.status
      if (existing.rows.length && !force) {
        console.log(`  ${arm.id.padEnd(34)} SKIP (exists: ${status})`)
        continue
      }
      const record = {
        id: arm.id, type: hypType(probe), title: arm.title, claim: arm.claim, rationale: arm.rationale,
        source: 'human', status: 'untested', proposedBy: probe.proposedBy, gate: probe.gate,
        spec: armMatrix(probe, arm), createdAt: at, updatedAt: at, transitions: [], verdictSource: 'auto',
      }
      console.log(`  ${arm.id.padEnd(34)} ${apply ? (existing.rows.length ? 'RE-SEED' : 'CREATE') : 'would create'}  cells=${cellCount(probe)}`)
      if (apply) await database.upsertEntity({ projectId: SCOPE, type: hypType(probe), externalKey: arm.id, content: record, shouldEmbed: false })
    }
  } finally { await close() }
}

export async function runCampaign(probe) {
  const { engine, close } = await openEngine()
  try {
    for (const arm of probe.arms) {
      const started = Date.now(); let last = -1
      const res = await engine.runSideExperimentCampaign({
        scope: SCOPE, projectRoot: REPO_ROOT, manifestRelPath: probe.manifest,
        thesis: `${probe.thesis} — ${arm.id}`, thesisTarget: 'signal', hypothesisId: arm.id,
        matrix: armMatrix(probe, arm), concurrency: 4, source: 'human', proposedBy: probe.proposedBy,
        onProgress: (p) => { if (p.done !== last) { last = p.done; process.stdout.write(`\r  ${arm.id.padEnd(34)} ${p.done}/${p.total} (${((Date.now() - started) / 1000).toFixed(0)}s)   `) } },
      })
      console.log(`\n  ${arm.id.padEnd(34)} -> ${res.completed}/${res.planned} completed, ${res.failed} failed`)
    }
  } finally { await close() }
}

const num = (s, k) => Number((s?.metrics || {})[k] ?? NaN)
const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN)

// Read-only per-window breakdown of a probe's recorded runs — mean + best oos_sharpe (and t ~ sharpe*sqrt(n),
// plus sharpe_vs_hold) grouped by window. The rigorous DSR verdict + adversarial verification are separate.
export async function analyze(probe) {
  const { client, close } = await openEngine()
  try {
    const rows = (await client.query('SELECT content FROM entities WHERE project_id=$1 AND type=$2', [SCOPE, expType(probe)])).rows.map((r) => r.content)
    for (const arm of probe.arms) {
      const rec = rows.find((r) => r.hypothesisId === arm.id)
      const cells = rec?.cells || []
      console.log(`\n=== ${arm.id} (${cells.length} cells) ===`)
      const byWin = {}
      for (const c of cells) { const w = (c.config || {}).walk_forward_window; (byWin[w] ||= []).push(c) }
      for (const [w, cs] of Object.entries(byWin).sort()) {
        const sh = cs.map((c) => num(c, 'oos_sharpe')).filter(Number.isFinite)
        const best = cs.slice().sort((a, b) => num(b, 'oos_sharpe') - num(a, 'oos_sharpe'))[0]
        const nobs = num(best, 'oos_n_obs')
        const t = num(best, 'oos_sharpe') * Math.sqrt(Number.isFinite(nobs) ? nobs : 0)
        console.log(`  ${String(w).padEnd(10)} n=${cs.length}  mean sharpe ${mean(sh).toFixed(3)}  best ${num(best, 'oos_sharpe').toFixed(3)} (t~${t.toFixed(2)})  vs_hold ${num(best, 'sharpe_vs_hold').toFixed(3)}  ret ${num(best, 'total_return_pct').toFixed(1)}% hold ${num(best, 'hold_return_pct').toFixed(1)}%`)
      }
    }
  } finally { await close() }
}
