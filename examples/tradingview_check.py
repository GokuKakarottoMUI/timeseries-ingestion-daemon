"""Print indicator values next to readable dates, to check against a chart.

    python -m examples.tradingview_check
    python -m examples.tradingview_check 1d 10        # timeframe, rows

The test suite proves the port matches *one reading* of the Pine reference; it
cannot prove the reading was right. Only a chart can close that loop, so this
prints the numbers in a form you can actually line up against one.

To make the comparison meaningful, the chart has to be configured identically:

  * **Symbol** — use the same exchange the data came from, e.g. `BITSTAMP:BTCUSD`.
    A different exchange has different candles and will never agree.
  * **Timezone** — set the chart to **UTC** (right-click the time axis). Stored
    timestamps are UTC; a local-time chart shifts every row.
  * **Indicator settings** — the values printed at the top of the run are the
    ones this pipeline used. Enter the same ones on the chart.
  * **Compare a closed candle**, not the newest one: the live bar keeps moving.

Small differences in the last decimal are expected and fine. A constant offset,
or a value that is right except near the start of the series, is not — that
points at seeding or warmup, which is exactly what is easy to get wrong.
"""
import sys
from datetime import datetime, timezone

from indicators.config.config_indicators import INDICATORS_CONFIG
from indicators.feature_pipeline import run

COLUMNS = ("close", "ema_9", "ema_26", "ema_50", "ema_100", "ema_200",
           "macd", "macd_signal", "macd_hist", "rsi_14", "rsi_ma_14",
           "bb_upper", "bb_basis", "bb_lower")

DATE_FMT = {1: "%Y-%m-%d %H:%M", 60: "%Y-%m-%d %H:%M"}      # by timeframe minutes


def print_settings():
    cfg = INDICATORS_CONFIG
    print("Set the chart to these, or the numbers cannot agree:")
    print(f"  EMA        lengths {cfg['ema']['lengths']} on {cfg['ema']['source']}")
    print(f"  MACD       {cfg['macd']['fast']}/{cfg['macd']['slow']}/{cfg['macd']['signal']}, "
          f"oscillator and signal both EMA, on {cfg['macd']['source']}")
    print(f"  RSI        length {cfg['rsi']['length']}, "
          f"smoothing SMA length {cfg['rsi']['ma_length']}")
    print(f"  Bollinger  length {cfg['bbands']['length']}, basis SMA, "
          f"StdDev {cfg['bbands']['mult']}")
    print("  Timezone   UTC")


def main():
    want_tf = sys.argv[1] if len(sys.argv) > 1 else None
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    result = run()
    if not result:
        print("\nNo data. Fetch first, and check that the timeframes you want are "
              "marked both `active` and `active_featured` in all_timeframes.json.")
        return

    keys = sorted(result)
    if want_tf:
        keys = [k for k in keys if k[3] == want_tf]
        if not keys:
            available = sorted({k[3] for k in result})
            print(f"\nTimeframe '{want_tf}' is not in the result. Available: "
                  f"{', '.join(available)}")
            return

    print()
    print_settings()

    for key in keys:
        entry = result[key]
        feat, names, ts = entry["feat"], entry["names"], entry["timestamp"]
        present = [c for c in COLUMNS if c in names]
        idx = [names.index(c) for c in present]

        # Sub-daily timeframes need the time of day to identify a bar.
        span = int(ts[1] - ts[0]) if len(ts) > 1 else 86_400_000
        fmt = "%Y-%m-%d" if span >= 86_400_000 else "%Y-%m-%d %H:%M"
        width = 12 if span >= 86_400_000 else 18

        n = min(rows, feat.shape[0])
        print(f"\n{key[0]}/{key[1]}/{key[2]} @ {key[3]} — "
              f"{feat.shape[0]:,} candles, last {n} shown")
        print(f"{'date (UTC)':<{width}}" + "".join(f"{c:>12}" for c in present))
        print("-" * (width + 12 * len(present)))
        for i in range(feat.shape[0] - n, feat.shape[0]):
            when = datetime.fromtimestamp(int(ts[i]) / 1000, tz=timezone.utc)
            print(f"{when.strftime(fmt):<{width}}"
                  + "".join(f"{feat[i, j]:>12,.2f}" for j in idx))
    print()


if __name__ == "__main__":
    main()
