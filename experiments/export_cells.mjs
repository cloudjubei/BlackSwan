// Export every recorded experiment cell (family, hypothesisId, config, projected metrics) to JSON so the Python
// attribution harness can read the battery without a psycopg dependency. Also exports the battery-theses.
import { createRequire } from 'module'
import path from 'path'
import fs from 'fs'
const require = createRequire(path.join('/Users/cloud/Documents/Work/thefactory-db', '/'))
const pg = require('pg')
const client = new pg.Client({ connectionString: 'postgresql://thefactory:thefactory@localhost:5435/thefactorydb', connectionTimeoutMillis: 8000 })
await client.connect()
const SCOPE = 'blackswan-experiments'
const OUT = new URL('.', import.meta.url).pathname

const MK = ['oos_sharpe', 'oos_n_obs', 'sharpe_vs_hold', 'turnover', 'trades_per_day', 'n_trades', 'gross_return_pct',
  'total_return_pct', 'hold_return_pct', 'realized_cost_bps', 'oos_ret_skew', 'oos_ret_kurt', 'psr',
  'min_track_record_length', 'universe_size', 'time_in_market_pct', 'mean_legs', 'time_in_market', 'max_drawdown_pct']

const rows = (await client.query(
  "SELECT type, content FROM entities WHERE project_id=$1 AND type LIKE 'blackswan-%-experiment'", [SCOPE])).rows
const out = []
for (const r of rows) {
  const c = r.content
  const family = r.type.replace('blackswan-', '').replace('-experiment', '')
  for (const cell of (c.cells || [])) {
    const m = cell.metrics || {}, cfg = cell.config || {}
    const met = {}
    for (const k of MK) if (m[k] !== undefined) met[k] = m[k]
    out.push({ family, hypothesisId: c.hypothesisId, key: cell.key, config: cfg, metrics: met })
  }
}
fs.writeFileSync(`${OUT}/cells.json`, JSON.stringify(out))
console.log(`wrote ${out.length} cells -> cells.json`)

const th = (await client.query("SELECT external_key, content FROM entities WHERE project_id=$1 AND type='blackswan-battery-thesis'", [SCOPE])).rows.map(r => r.content)
fs.writeFileSync(`${OUT}/battery_theses.json`, JSON.stringify(th, null, 2))
console.log(`wrote ${th.length} battery-theses -> battery_theses.json`)
await client.end()
