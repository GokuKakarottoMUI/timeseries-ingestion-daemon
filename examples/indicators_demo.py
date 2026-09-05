"""Compute indicators over the stored candles and show what comes out.

    python -m examples.indicators_demo

Prints, per symbol and timeframe, the feature block's shape and layout and —
for each indicator column — the first bar that carries a value. Those warmup
indices are the quickest way to confirm the port still matches TradingView:
EMA-200 must start at bar 199, the MACD signal at 33, RSI at 14, and so on.

Also prints the last few rows so the values can be checked against a chart.
"""
import numpy as np

from get_data.get_data_from_database import BASE_COLS
from indicators.feature_pipeline import run

TAIL_ROWS = 3


def main():
    print("\n" + "=" * 96)
    print("Indicator features (parameters from indicators/config/indicators_config.yaml)")
    print("=" * 96)

    result = run()
    if not result:
        print("\nNo data. Run `python -m ingestion.api_fetch` first.")
        return

    for (market, category, symbol, tf), entry in result.items():
        feat, names, ts = entry["feat"], entry["names"], entry["timestamp"]
        n_base = len(BASE_COLS)

        print(f"\n{market}/{category}/{symbol} @ {tf}")
        print(f"  rows          : {feat.shape[0]:,}")
        print(f"  columns       : {feat.shape[1]} "
              f"({n_base} stored + {feat.shape[1] - n_base} indicators)")
        print(f"  dtype / layout: {feat.dtype}, F-contiguous={feat.flags['F_CONTIGUOUS']}")
        print(f"  memory        : {feat.nbytes / 1024 / 1024:,.2f} MB")

        print(f"\n  {'column':<14}{'first bar':>10}{'  last value':>18}")
        print("  " + "-" * 42)
        for j, name in enumerate(names[n_base:], start=n_base):
            col = feat[:, j]
            finite = np.flatnonzero(np.isfinite(col))
            if finite.size:
                print(f"  {name:<14}{finite[0]:>10}{col[-1]:>18,.4f}")
            else:
                print(f"  {name:<14}{'never':>10}{'--':>18}   (history too short)")

        rows = min(TAIL_ROWS, feat.shape[0])
        print(f"\n  last {rows} rows — compare these against the chart:")
        shown = ("close", "ema_9", "ema_200", "macd", "rsi_14", "bb_upper", "bb_lower")
        idx = [names.index(c) for c in shown]
        print("    " + "timestamp".ljust(16) + "".join(f"{c:>14}" for c in shown))
        for i in range(feat.shape[0] - rows, feat.shape[0]):
            values = "".join(f"{feat[i, j]:>14,.2f}" for j in idx)
            print(f"    {int(ts[i]):<16}{values}")

    print("\n" + "=" * 96)
    print("Done")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
