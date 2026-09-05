# timeseries-ingestion-daemon

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
   training / feature pipeline
```

Storing every timeframe of a symbol inside **one** array (rather than one array per timeframe) is
what lets the read path open a single handle per symbol and slice each timeframe off it, instead of
paying an open + fragment-list fetch per timeframe.

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
pip install -e ".[dev]"          # add ",gui" for the config GUI

pytest                           # no network or existing database needed

python -m ingestion.api_fetch    # one fetch pass; creates ./data/market_data/...
python -m ingestion.query_data   # inspect what landed: counts, range, gaps
python -m examples.query_demo    # read it back the way a training job would
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
examples/query_demo.py      end-to-end read example
tests/                      pytest suite (no network, no pre-existing database)
```

## Testing

```bash
pytest              # whole suite
pytest -v -k tiledb # just the storage round-trip
```

The tests build TileDB arrays under pytest's `tmp_path` and drive the parsers with recorded
exchange payloads, so they need neither network access nor an existing database.

## License

MIT — see [LICENSE](LICENSE).
