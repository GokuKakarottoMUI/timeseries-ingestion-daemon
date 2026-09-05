# timeseries-ingestion-daemon

[![tests](https://github.com/dinhphucdien/timeseries-ingestion-daemon/actions/workflows/tests.yml/badge.svg)](https://github.com/dinhphucdien/timeseries-ingestion-daemon/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An async daemon that continuously pulls OHLCV candles from multiple crypto exchanges and stores
them in **TileDB** sparse arrays, laid out so a training job can read a whole symbol back as one
contiguous `float64` block with no copies and no dataframe layer in between.

The design constraint driving everything here: **no per-candle work in interpreted Python**. Every
hot path is a vectorized NumPy operation or a call into a C/C++/Rust library. There is no pandas
anywhere in the pipeline — not even as a temporary intermediate — because a DataFrame's row/label
model, implicit copies and index alignment overhead would dominate the cost of moving millions of
candles.

---

## Architecture

### Ingestion and read path

```
   exchange REST APIs
           │  aiohttp, per-exchange rate limiter + 429 backoff
           ▼
   ExchangeFormatter          parse JSON → (N, 6) float64, driven purely by config
           │                  [timestamp, open, high, low, close, volume]
           ▼
   TensorBuffer               in-memory accumulator, np.concatenate (C level)
           │
           ▼
   DatabaseManager            dedup → one F-ordered allocation → columnar write
           │                  writes serialized per array, parallel across symbols
           ▼
   ┌─────────────────────────────────────────────┐
   │  TileDB sparse 2D array, one per symbol     │
   │  dims  : (timeframe_minutes int32,          │
   │           timestamp int64 ms-epoch)         │
   │  attrs : open, high, low, close, volume     │
   │          (float64, Zstd-7)                  │
   └─────────────────────────────────────────────┘
           │  flock(2) reader/writer coordination
           ▼
   GetDataFromDatabase        one thread + one array handle per symbol
           │                  → (N, 5) float64 Fortran-order block
           ▼
   feature_pipeline           indicator columns appended beside the stored ones
           │                  → (N, 18) float64, one allocation, warmup left NaN
           ▼
   model training
```

Storing every timeframe of a symbol inside **one** array (rather than one array per timeframe) is
what lets the read path open a single handle per symbol and slice each timeframe off it, instead of
paying an open + fragment-list fetch per timeframe.

### Building custom timeframes (phase 2)

Phase 1 stores the timeframes the exchange actually serves. Phase 2 derives the non-standard ones
(11m, 45m, 90m, 2d, 8d …) from what is already on disk, so the exchange is hit once per base candle
and never again for a derived one.

```
   for each (symbol, custom timeframe)
           │
           ▼
   CacheManager                which slots are genuinely missing
           │                   nothing missing → skip the source query entirely
           ▼
   dependency layers           regular TF = level 0, custom TF = level(source) + 1
           │                   a chain like 1h → 3h → 6h therefore needs two layers
           ▼
   ┌────────────────── layer 0 ──────────────────┐   sources inside a layer run in
   │   source 1d ──┬──► 2d                       │   parallel; layers run strictly in
   │               └──► 3d                       │   order, because layer N+1 reads
   └─────────────────────────────────────────────┘   what layer N has just written
           │   … layer 1, layer 2, same shape        (buffer flushed after each layer)
           ▼
   query_candles               read the base OHLCV back → (N, 6) float64
           │                   range clamped to [slot_min, slot_max + target]
           ▼
   _aggregate_candles_batch    searchsorted → slot id per candle, then
           │                   maximum/minimum/add.reduceat per group
           │                   open = first of slot, close = last of slot
           │                   no Python loop over candles at any point
           ▼
   TensorBuffer → TileDB       written back into the SAME array,
                               under a different timeframe_minutes coordinate
```

### Indicators

`indicators/` ports four TradingView scripts to NumPy — EMA, MACD, RSI and
Bollinger Bands — and appends them to the stored OHLCV as extra columns.

The work is in matching Pine exactly, because an indicator that is *almost* right
disagrees with the chart the strategy was designed on, silently. Two details carry
most of that risk:

- **`ta.ema` is seeded with `SMA(length)`, not with the first sample.** The common
  `adjust=False` shortcut produces a curve that converges only slowly and never
  quite matches.
- **Pine counts warmup from the first non-`na` value, not from bar 0.** The MACD
  signal line is an EMA *of the MACD*, which is itself `na` for 25 bars, so the
  signal must start at bar 33 — not 8.

Those warmup boundaries are asserted per column in the test suite, since an
off-by-one there stays plausible-looking and wrong.

The port is also checked against the chart itself, which is the only thing that
can prove the Pine reference was read correctly: `python -m examples.tradingview_check`
prints the values beside readable dates. On BTCUSD daily, six consecutive closed
candles matched TradingView across all fourteen columns — EMA 9/26/50/100/200,
MACD line/signal/histogram, RSI and its smoothing MA, and the three Bollinger
bands.

Checking the tail is not enough on its own. An EMA forgets its seed
exponentially, so seeding from the wrong value still agrees after a few thousand
bars — the mistake only shows near the start of the series. So the head was
checked too: at bar 231 of 5,497 the SMA seed gives 5.07 where seeding from the
first sample would give 5.55, and the chart reads the former.

Warmup is left as `NaN`. There is no forward-fill and no sentinel value, so the
consumer decides where its usable history begins rather than inheriting a choice
made here — and a filled-in value can never be mistaken for a real one.

`ta.stdev` is the population form (`ddof=0`); using the sample form would widen
every Bollinger band by a constant factor without ever raising.

The EMA/RMA recursions are IIR filters, so they run through
`scipy.signal.lfilter` (C/Fortran core) with the Pine seed expressed as the
filter's initial condition — not a Python loop over candles.

### Hand-off to torch

`feature_pipeline.to_torch()` exposes a block to PyTorch without copying it —
`torch.from_numpy` aliases the same buffer, so a write through either side is
visible from the other.

The Fortran layout, chosen so TileDB could take contiguous columns on **write**,
pays off again here on **read**:

| | shares memory | C-contiguous |
|---|---|---|
| per-column tensor | yes | **yes** — usable directly |
| whole `(N, F)` block | yes | no (strides `(1, N)`) |
| `torch.tensor(block)` | no — always copies | yes |

So the columns are the useful handle. Handing over the whole block is still
zero-copy, but the first op that needs C-contiguity calls `.contiguous()`
internally and copies it anyway — the copy is deferred, not removed. The tests
pin this by pointer identity rather than by inspection, because a stray
`.contiguous()` doubles the footprint and nothing fails.

torch is an **optional** dependency (`pip install -e ".[torch]"`); everything
else in the repo runs on NumPy and SciPy alone.

### Daemon cycle

`continuous_fetch` is meant to stay up for weeks, so the loop is built around returning memory and
never carrying state across a cycle boundary.

```
   _tune_malloc()          mallopt: pin mmap/trim thresholds, cap arenas at 4
           │               must run BEFORE any thread exists
           ▼
   cold start (once)       precompute → prime executors → prewarm DNS
           │               gc.collect() + gc.freeze(): config, TileDB ctx, session
           │               and executors are immutable for the run, so the GC
           │               never rescans them again
           ▼
   ┌───►  fetch pass (phase 1)         symbols × timeframes, one shared event loop
   │           │
   │           ▼
   │      aggregate custom TF          skipped when no new slot needs building
   │           │
   │           ▼
   │      consolidate + vacuum         only if this cycle actually wrote candles;
   │           │                       holds the flock EXCLUSIVE for its duration
   │           ▼
   │      drain TensorBuffer           leftovers never cross into the next cycle
   │           │
   │           ▼
   │      gc.collect() + malloc_trim   hand the heap back to the OS while idle
   │           │
   │           ▼
   │      RSS over ceiling? ──yes──►   flush logs, then os.execv(self)
   │           │                       libtiledb leaks a little per sparse read and
   │           no                      it cannot be reclaimed in-process, so the
   │           ▼                       only honest fix is a clean restart at the
   └──── sleep(interval)               cycle boundary
```

---

## What is interesting in here

**Exchange adapters are pure configuration.** Adding an exchange means adding a JSON block, not
writing code. `ingestion/config/exchange_configs.json` describes the URL template, the query
parameters, where the candles live in the response (`data_path`), and how each OHLCV field maps
onto the payload — by array index for exchanges that return arrays (Binance), by key for ones that
return objects (Bitstamp). `ExchangeFormatter` reads that description and produces the same
`(N, 6)` float64 array either way.

**Rate limiting is a leaky bucket, not a semaphore.** `SlidingWindowRateLimiter` hands each request
an evenly spaced slot rather than letting 80 coroutines fire at once. On HTTP 429 it halves the
rate and recovers gradually, so the fetcher self-governs against the exchange's real limit instead
of oscillating between bans and idling.

**Cross-process read/write coordination.** TileDB `vacuum` physically deletes fragment files. A
reader that opened the array just before a vacuum has a stale fragment list and hits `ENOENT` when
it lazily opens those files — losing one (symbol, timeframe) *silently*. `ingestion/rwlock.py`
closes that window with POSIX `flock(2)`: readers hold SHARED for a whole symbol read, the writer
holds EXCLUSIVE around consolidate+vacuum. The lock lives on the file descriptor, so a crashed
process cannot leave a stale lock behind.

**RSS ratchet control for a long-lived daemon.** A process that runs for weeks grows its RSS even
with bounded Python state, for two glibc reasons: the dynamic mmap threshold ratchets upward as
NumPy/TileDB free multi-MB blocks (so later large allocations come from `sbrk` heap and are never
returned to the OS), and arena count grows with thread count. `continuous_fetch.py` pins both
thresholds via `mallopt`, caps `M_ARENA_MAX`, and calls `malloc_trim(0)` at the idle point of each
cycle. It also re-execs itself if RSS crosses a configured ceiling, because libtiledb 0.36.1 leaks
a small amount per sparse read that cannot be reclaimed in-process.

**Zero-copy is enforced at the boundaries, not just claimed.**
- Write: TileDB consumes buffers *per column* and needs each one contiguous. Candle arrays arrive
  C-ordered, so `_write_single` makes exactly one `asfortranarray` of the five written columns —
  a single allocation instead of five separate copies made inside TileDB.
- Read: `_assemble_block` makes one Fortran-ordered `(N, len(BASE_COLS))` allocation and fills it
  column by column. The by-name entries in the returned dict are views into that same buffer, so
  handing them to torch stays copy-free.

**`float64` end to end, for a measured reason.** With a 0.01 exchange tick, `float32`'s ulp exceeds
one cent above ~131 072 USD — two genuinely different prices collapse onto the same value. Storage
and the read path therefore stay `float64`; any down-cast belongs once at the end of a feature
pipeline, not at the storage boundary.

**Custom timeframes are aggregated, not re-fetched.** Non-standard intervals (11m, 45m, 90m, 2d,
8d …) are built from an already-stored base timeframe with a vectorized bucket aggregation
(`np.add.reduceat`-style slot maths), so the exchange is hit once per base candle.

---

## Requirements

- **Linux.** Not portable by design: the daemon reads `/proc/self/statm` and `/proc/meminfo` for
  its memory guards and uses `fcntl.flock` for cross-process locking.
- **Python 3.12+**
- Local filesystem for the data directory (`flock` is advisory POSIX locking and is not valid
  over NFS).
- For the optional GUI: a running X11/Wayland session and `libxcb-cursor0`
  (`sudo apt install libxcb-cursor0` on Debian/Ubuntu — Qt 6.5+ requires it).

## Quickstart

```bash
git clone <repo-url> && cd timeseries-ingestion-daemon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",gui" for the config GUI, ",torch" for the tensor hand-off

pytest                           # no network or existing database needed

python -m ingestion.api_fetch    # one fetch pass; creates ./data/market_data/...
python -m ingestion.query_data   # inspect what landed: counts, range, gaps
python -m examples.query_demo    # read it back the way a training job would
python -m examples.indicators_demo  # compute indicators and show the feature block
```

Data goes to `./data/` inside the repo. To put it elsewhere, set `TSD_DATA_ROOT`:

```bash
TSD_DATA_ROOT=/mnt/fast-ssd/candles python -m ingestion.api_fetch
```

### Entry points

| Command | What it does |
|---|---|
| `python -m ingestion.api_fetch` | One fetch pass: all symbols × timeframes, then aggregate custom timeframes, then consolidate |
| `python -m ingestion.continuous_fetch` | The daemon — same pass on a loop, with the memory guards and self-restart |
| `python -m ingestion.query_data` | Inspect one symbol/timeframe: candle count, min/max, duplicates, gap report |
| `python -m examples.query_demo` | Read every active symbol back and print block shape / dtype / zero-copy status |
| `python -m examples.indicators_demo` | Compute the indicator features and print each column's shape and warmup boundary |
| `python -m ingestion.unified_config_manager` | PySide6 GUI to edit the JSON configs and run the daemon with live logs |

## Configuration

Everything is JSON/YAML; no code change is needed to add a symbol, a timeframe or an exchange.

| File | Controls |
|---|---|
| `ingestion/config/exchange_configs.json` | Exchange adapters: URL, params, response mapping, rate limit, concurrency, 429 backoff |
| `ingestion/config/symbols_config.json` | Which markets / categories / symbols are active, and each symbol's per-exchange ticker variants |
| `ingestion/config/all_timeframes.json` | Standard timeframes, plus custom timeframes and which base timeframe each aggregates from |
| `ingestion/config/historical_data_config.json` | Start date, `fetch_all` mode, missing-timestamp scan, fetch/write concurrency, write rate cap |
| `ingestion/config/continuous_fetch_mode.json` | Daemon loop intervals and the RSS ceiling that triggers a restart |
| `get_data/config/query_config.yaml` | Read side: query the full history or from a date, and reader thread count |
| `indicators/config/indicators_config.yaml` | Indicator parameters: EMA lengths, MACD periods, RSI length, Bollinger length and multiplier |
| `indicators/config/compute_config.yaml` | Feature block output dtype |

The `active` flags are hierarchical — a symbol is only fetched when its market, its category and
the symbol itself are all active.

## Layout

```
ingestion/                  fetch + storage
  api_fetch.py              async fetch orchestration, TensorBuffer, rate limiting
  continuous_fetch.py       daemon loop, glibc allocator tuning, self-restart
  database.py               TileDB schema, batch writes, consolidate/vacuum
  exchange_utils.py         config-driven URL building and response parsing
  calculate_tf_and_custom_tf.py   timeframe maths + vectorized custom-TF aggregation
  cache_timestamp.py        which intervals still need fetching, cached
  timestamp_scanner.py      scans stored dimensions to find missing candles
  rwlock.py                 flock(2) reader/writer lock
  query_data.py             CLI to inspect a stored array
  unified_config_manager.py PySide6 config GUI
  config/*.json
get_data/                   read path for training
  get_data_from_database.py parallel per-symbol read → contiguous float64 blocks
  config/query_config.yaml
indicators/                 feature layer
  primitives.py             Pine-faithful sma / ema / rma / stdev
  engines/                  one class per TradingView script
  feature_pipeline.py       reads through get_data, appends indicator columns,
                            hands off to torch without copying
  config/*.yaml
examples/query_demo.py      end-to-end read example
examples/indicators_demo.py end-to-end feature example
tests/                      pytest suite (no network, no pre-existing database)
```

## Testing

```bash
pytest              # whole suite
pytest -v -k tiledb # just the storage round-trip
```

The tests build TileDB arrays under pytest's `tmp_path` and drive the parsers with recorded
exchange payloads, so they need neither network access nor an existing database.

CI runs the suite on every push against a **base install only** — no torch, no PySide6 — which
keeps the optional dependencies honestly optional: the torch hand-off test skips rather than
fails. A clean checkout gives 118 passed, 1 skipped; with `[torch]` installed, 130 passed.

## License

MIT — see [LICENSE](LICENSE).
