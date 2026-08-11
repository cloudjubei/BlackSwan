# BlackSwan experiments

The recorded, reproducible catalogue of BlackSwan **side-experiment probes** — each a pre-registered
hypothesis (usually a thesis + a mirror control), a declared **gate**, and a **sweep matrix**, run against a
`.factory/trainer-*.json` manifest. This replaces one-off scripts: the probes live in the repo, so any of them
can be re-registered, re-run, and re-analysed deterministically.

Each probe conforms to the modeltrainer trainer contract via a `trainer/<name>.py` CLI module + a
`.factory/trainer-<name>.json` manifest. The engine (pre-register → run campaign → materialise verdict) is
provided by the sibling `thefactory-modeltrainer` repo; this directory is the thin, committed BlackSwan-side glue.

## Usage

```bash
node experiments/cli.mjs list                          # the recorded probes
node experiments/cli.mjs <probe> preregister           # dry-run: show what would be recorded
node experiments/cli.mjs <probe> preregister --apply   # write the untested hypotheses to the trail
node experiments/cli.mjs <probe> run                   # run the campaign (sweep) for each arm
node experiments/cli.mjs <probe> analyze               # read-only per-window oos_sharpe breakdown
```

`preregister` is **idempotent-safe**: a hypothesis that already exists is SKIPPED (a materialised verdict is
never clobbered); pass `--force` to re-seed it to `untested`. Verdicts are materialised (with adversarial
verification) separately — the analyse step is a quick read, not the gate decision.

## Environment

| Var | Default | Purpose |
| --- | --- | --- |
| `MODELTRAINER_DIR` | `../thefactory-modeltrainer` | the modeltrainer engine (node/TS) |
| `THEFACTORY_DB_DIR` | `../thefactory-db` | the pgvector DB client |
| `BLACKSWAN_DB_URL` | `postgresql://thefactory:thefactory@localhost:5435/thefactorydb` | experiments DB |

Data miners that some probes depend on need their own keys (e.g. `EIA_API_KEY` for the roll probes'
`scripts/fetch_energy.py`, `FRED_API_KEY` for the world model's macro drivers).

## Recorded probes

| Probe | Manifest | Status |
| --- | --- | --- |
| `roll` | `trainer-roll.json` | DISPROVED — single-asset WTI Goldman-roll front-run |
| `roll-basket` | `trainer-roll-basket.json` | DISPROVED — pooled energy-basket roll front-run |
| `gold-worldmodel` | `trainer-worldmodel.json` | driver-conditioned gold timing (real rate + USD + breakeven) |

The full pre-registered claims + gates live in [`registry.mjs`](./registry.mjs); the verdicts live in the DB
hypothesis trail (`blackswan-experiments` scope).
