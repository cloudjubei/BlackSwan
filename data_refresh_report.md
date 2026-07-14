# Data refresh report — 2026-07

Backfill of binance/ through the latest complete month (2026-06), altcoin 1h/1d derivation, and a
new stocks/ dataset for the top-10 US stocks. All data validated per month (strictly monotonic
unique timestamps, positive numeric prices, bars inside the month, expected bar counts) and across
month boundaries (previous close + 1 == next open). Everything round-trips through the existing
readers (`pd.read_json` in the providers) unchanged — verified end-to-end by building configs via
`trainer.config_builder` and stepping real providers over the new files.

## Crypto download (data.binance.vision monthly archives)

| Symbol | TF | Downloaded | Coverage now | Gaps |
|---|---|---|---|---|
| BTCUSDT | 1m | 2024-9 .. 2026-6 (22 months) | 2017-8 .. 2026-6 (107 months) | none |
| BTCUSDT | 1h | 2024-9 .. 2026-6 (22 months) | 2017-8 .. 2026-6 (107 months) | none |
| BTCUSDT | 1d | 2024-7 .. 2026-6 (24 months) | 2017-8 .. 2026-6 (107 months) | none |
| 8 altcoins* | 1m | 2024-5 .. 2026-6 (26 months each) | 2022-1 .. 2026-6 (54 months each) | none |

*ADAUSDT, DOGEUSDT, DOTUSDT, ETHUSDT, LTCUSDT, SHIBUSDT, SOLUSDT, XRPUSDT.

Every downloaded month was a complete month (full expected bar count); zero intra-month or
boundary gaps were found. Altcoins were not extended backward before 2022-01 (several only list
from 2020-2021 on Binance and the trainer's windows start at 2020; not trivially cheap, skipped).

Source quirks handled:
- Archives from 2025-01 onward carry open/close times in MICROSECONDS (Binance change); the
  converter normalizes to epoch ms, matching the on-disk convention.
- 1m rows carry `tokenPair`/`interval` keys, 1h/1d rows don't — matching the existing files.
- New 1h/1d months carry no `indicators` dict. Indicators are computed at runtime from OHLCV
  (`_add_curated_indicators` / feature cache); the embedded dicts in pre-2024-07 files are a legacy
  data-mine artifact. Only the legacy `indicator: indicatorsN` / `type: standard` config paths read
  them, and those already can't span derived-cache files either.

## Derived altcoin 1h/1d (from the 1m source of truth)

- 1d: 54 monthly files per altcoin written to `binance/{SYMBOL}-1d-{Y}-{M}.json` (432 files) — the
  config_builder 1d path reads these raw. Derivation = `trainer.derive_cache.derive_bars`
  (open=first, high=max, low=min, close=last, volume/quote/trades/taker summed). Spot-verified
  equal to direct 1m aggregation.
- 1h: 54 months per altcoin materialised into the canonical derived cache
  `binance/derived/{SYMBOL}-1h-{Y}-{M}.json` via `ensure_derived` (432 files). BTCUSDT's 1h cache
  for the new months stays lazy, per the repo design (derived on first use).

## Stocks (new stocks/ dataset)

Top-10 US-listed stocks by market cap, plain common-stock tickers: NVDA, MSFT, AAPL, GOOGL, AMZN,
META, AVGO, TSLA, JPM, WMT. BRK-B was excluded deliberately: the hyphenated ticker doesn't fit the
`SYMBOL-TF-Y-M.json` filename grammar (`[A-Z0-9]+`) and the task prefers plain common stock.

- Coverage: 2018-1 .. 2026-6, 102 monthly files per ticker (1020 files), daily bars only.
- Format: identical kline shape + naming as binance/ (`{TICKER}-1d-{YYYY}-{M}.json`, month not
  zero-padded). Verified to load through `SingleDataProvider` untouched.
- Sources: stooq.com (primary) is currently unreachable programmatically — it serves a JavaScript
  proof-of-work bot-challenge page instead of CSV (verified via curl; affects all request forms).
  The script detects that and falls back to yfinance (installed into .venv), which supplied all 10
  tickers. Prices are split/dividend-adjusted so splits (e.g. NVDA 2024) don't appear as fake
  crashes.
- Neutral fills for fields the source lacks, matched to what the reader consumes:
  `asset_volume_quote` = close x volume, `asset_volume_taker_base/quote` = half of volume/quote
  (constant `taker_buy_ratio` 0.5), `trades_number` = 0 (its pct_change feature becomes a neutral
  zero column).
- Validation: weekend/holiday holes are treated as normal; months are checked structurally plus a
  sane trading-day count (15-23). No findings.
- The readers don't hardcode binance/ — `train_data_paths` are explicit file lists, so stocks/
  works with existing providers; only `trainer/config_builder.py` path helpers are binance-specific.

## Code changes (working tree, not committed)

- `scripts/backfill_klines.py` — idempotent Binance backfill + altcoin derivation; streams one
  month at a time (small peak memory); per-month validation; summary table. Pure logic covered by
  `scripts/test_backfill_klines.py`.
- `scripts/backfill_stocks.py` — stooq-with-yfinance-fallback stocks backfill; same month-file
  writer; stock-aware validation. Covered by `scripts/test_backfill_stocks.py`.
- `trainer/config_builder.py` — the hardcoded "BTCUSDT-only" gates on the 1m and 1h paths are now
  data-driven (fail fast when the asset's raw 1m window files are missing), since the altcoin
  klines they were waiting on now exist. 1d needed no change.
- `trainer/data_inventory.py` — `scan_stocks_inventory()` (thin `scan_inventory` over stocks/);
  docstring refreshed.
- Tests updated/added in `trainer/test_config_builder.py`, `trainer/test_data_inventory.py`.

## Re-running the backfill later

```bash
# crypto through the latest complete month + altcoin 1h/1d derivation (idempotent)
.venv/bin/python -m scripts.backfill_klines
# stocks through the latest complete month (idempotent)
.venv/bin/python -m scripts.backfill_stocks
# plan preview / verification
.venv/bin/python -m scripts.backfill_klines --dry-run
.venv/bin/python -c "from trainer.data_inventory import scan_inventory, scan_stocks_inventory; print(scan_inventory()); print(scan_stocks_inventory())"
```

## Open questions

- `trainer/walk_forward.py` windows still end at test-year 2024; add "2025" (and later "2026")
  to `_WINDOW_TEST_YEARS` when the trainer should evaluate on the new data.
- If stooq drops its bot-challenge, the script automatically prefers it again (no key, full
  history); nothing to change.
