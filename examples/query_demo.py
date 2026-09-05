"""Read the stored candles back the way a training job would.

Run it after a fetch:

    python -m examples.query_demo

It prints, per symbol and timeframe, the candle count, the timestamp range and the
memory footprint of each zero-copy column view, so you can confirm the read path
returns one contiguous float64 block rather than a pile of copies.

What to change is in `get_data/config/query_config.yaml`, not in this file.
"""
import logging

from get_data.get_data_from_database import GetDataFromDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    print("\n" + "=" * 80)
    print("Query data from database (driven by query_config.yaml)")
    print("=" * 80 + "\n")

    db = GetDataFromDatabase()
    data = db.query_data_for_training()

    if not data:
        print("No data returned. Fetch first, or check the `active` flags in the config.")
        return

    for market_cat, market_data in data.items():
        for symbol_cat, symbol_data in market_data.items():
            for symbol, timeframes in symbol_data.items():
                for tf_name, tf_data in timeframes.items():
                    ts    = tf_data['timestamp']
                    block = tf_data['block']

                    print(f"\n{market_cat}/{symbol_cat}/{symbol} @ {tf_name}")
                    print(f"  candles     : {len(ts):,}")
                    print(f"  first / last: {ts[0]} → {ts[-1]}" if len(ts) else "  (empty)")
                    print(f"  block       : shape={block.shape} dtype={block.dtype} "
                          f"F-contiguous={block.flags['F_CONTIGUOUS']}")
                    print(f"  columns     : {', '.join(tf_data['columns'])}")

                    total_kb = block.nbytes / 1024
                    print(f"  memory      : {total_kb:,.2f} KB block "
                          f"+ {ts.nbytes / 1024:,.2f} KB timestamp")
                    for col in tf_data['columns']:
                        view = tf_data[col]
                        # base is not None ⇒ view vào chính `block`, không phải bản copy
                        print(f"    {col:<8} {view.nbytes / 1024:>10,.2f} KB  "
                              f"zero-copy view={view.base is not None}")

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
