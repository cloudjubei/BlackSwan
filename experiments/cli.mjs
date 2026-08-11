#!/usr/bin/env node
// BlackSwan side-experiment runner. Records + runs the pre-registered probes in registry.mjs via the
// modeltrainer engine.
//
//   node experiments/cli.mjs list
//   node experiments/cli.mjs <probe> preregister [--apply] [--force]
//   node experiments/cli.mjs <probe> run
//   node experiments/cli.mjs <probe> analyze
//
// preregister is a dry-run by default (prints what it would do); pass --apply to write. An already-recorded
// hypothesis is SKIPPED (a materialised verdict is never clobbered) unless --force re-seeds it to untested.

import { PROBES } from './registry.mjs'
import { preregister, runCampaign, analyze } from './lib.mjs'

const [probeName, action, ...flags] = process.argv.slice(2)
const force = flags.includes('--force')
const apply = flags.includes('--apply')

function usage() {
  console.log('usage: node experiments/cli.mjs <probe> <preregister|run|analyze> [--apply] [--force]')
  console.log('       node experiments/cli.mjs list')
  console.log('probes:', Object.keys(PROBES).join(', '))
}

if (probeName === 'list' || !probeName) {
  console.log('Recorded BlackSwan probes:')
  for (const [name, p] of Object.entries(PROBES)) {
    console.log(`  ${name.padEnd(18)} ${p.manifest}  arms=${p.arms.map((a) => a.id).join(', ')}`)
  }
  process.exit(0)
}

const probe = PROBES[probeName]
if (!probe) { console.error(`unknown probe: ${probeName}`); usage(); process.exit(1) }

const run = async () => {
  if (action === 'preregister') await preregister(probe, { force, apply })
  else if (action === 'run') await runCampaign(probe)
  else if (action === 'analyze') await analyze(probe)
  else { usage(); process.exit(1) }
}
run().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1) })
