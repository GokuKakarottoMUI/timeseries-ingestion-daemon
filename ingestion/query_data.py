"""Inspect what is actually stored in a TileDB symbol array.

Run it after a fetch to check the data really landed:

    python -m ingestion.query_data

Two sections:
  1. OVERVIEW — integrity stats: candle count, min/max timestamp, duplicates, gaps.
  2. VIEW     — print the OHLCV rows inside the [FROM, TO] date range.

Edit the configuration block below to pick symbol / timeframe / range.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import tiledb

from ingestion.config_fetch_data import (
    SYMBOLS_CONFIG, TIMEFRAMES, CUSTOM_TIMEFRAMES, build_array_path,
)

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH — chỉnh ở đây
# ══════════════════════════════════════════════════════════════════════════════
SYMBOL     = "BTCUSD"        # tên array
TIMEFRAME  = "1d"            # khớp TIMEFRAMES / CUSTOM_TIMEFRAMES
FROM       = "2025-01-01"    # 'YYYY-MM-DD' (UTC). None → bỏ qua phần xem nến
TO         = None            # 'YYYY-MM-DD' (UTC). None → tới hết
MAX_ROWS   = 20              # số dòng in ở mỗi đầu/cuối khi xem nến
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_ts(ms: int) -> str:
    """timestamp ms → chuỗi UTC người đọc."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _date_to_ms(date_str: str) -> int:
    """'YYYY-MM-DD' (UTC 00:00) → timestamp ms."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _resolve_array_path(symbol: str) -> str | None:
    """Quét SYMBOLS_CONFIG tìm market/symbol_category chứa symbol → build path."""
    for market, md in SYMBOLS_CONFIG["market"].items():
        for sc_name, sc in md["symbols_config"].items():
            if symbol in sc["symbols"]:
                return build_array_path(market, sc_name, symbol)
    return None


def _resolve_tf_minutes(timeframe: str) -> int | None:
    """timeframe_minutes từ TIMEFRAMES, fallback CUSTOM_TIMEFRAMES."""
    if timeframe in TIMEFRAMES:
        return TIMEFRAMES[timeframe].get("minutes", 0) or None
    custom = CUSTOM_TIMEFRAMES.get("custom_intervals", {})
    if timeframe in custom:
        m = custom[timeframe].get("minutes", 0)
        h = custom[timeframe].get("hours", 0)
        return (m or h * 60) or None
    return None


def overview(array_path: str, tf_minutes: int) -> None:
    """Thống kê toàn vẹn — query chỉ dimension timestamp (bỏ I/O attribute)."""
    print(f"\n{'='*70}\n== OVERVIEW {SYMBOL} tf{tf_minutes}m ==\n{'='*70}")

    with tiledb.open(array_path, "r") as A:
        res = A.query(attrs=[], dims=["timestamp"])[tf_minutes, :]
    ts = np.asarray(res["timestamp"], dtype=np.int64)

    total = ts.size
    if total == 0:
        print("Không có nến nào cho timeframe này.")
        return

    ts.sort()
    dup = total - np.unique(ts).size
    ts_min, ts_max = int(ts[0]), int(ts[-1])
    step = tf_minutes * 60_000
    expected = (ts_max - ts_min) // step + 1
    missing = expected - total

    print(f"tổng nến   : {total:,}")
    print(f"min        : {ts_min}  ({_fmt_ts(ts_min)} UTC)")
    print(f"max        : {ts_max}  ({_fmt_ts(ts_max)} UTC)")
    print(f"trùng lặp  : {dup}")
    print(f"step        : {step} ms ({tf_minutes} phút)")
    print(f"expected    : {expected:,} (nếu liền mạch)")
    print(f"thiếu       : {missing:,}")

    diffs = np.diff(ts)
    gap_idx = np.flatnonzero(diffs != step)
    print(f"số điểm gap : {gap_idx.size}")
    if gap_idx.size:
        # Sắp xếp gap theo độ lớn giảm dần, in tối đa 10 cái
        gap_sizes = diffs[gap_idx]
        order = np.argsort(gap_sizes)[::-1][:10]
        print(f"  {'gap lớn nhất (tối đa 10)':<40} | nến thiếu")
        for i in order:
            gi = gap_idx[i]
            g = int(gap_sizes[i])
            n_missing = g // step - 1
            print(f"  {_fmt_ts(int(ts[gi]))} → {_fmt_ts(int(ts[gi+1]))} | {n_missing:,}")
    else:
        print("  Liền mạch tuyệt đối — không gap.")


def view(array_path: str, tf_minutes: int) -> None:
    """In nến OHLCV trong khoảng [FROM, TO]."""
    if FROM is None:
        return

    start_ms = _date_to_ms(FROM)
    end_ms = _date_to_ms(TO) if TO else (np.iinfo(np.int64).max - 10_000)

    attrs = ["open", "high", "low", "close", "volume"]

    print(f"\n{'='*70}\n== CANDLES {SYMBOL} tf{tf_minutes}m  {FROM} → {TO or 'hết'} ==\n{'='*70}")

    with tiledb.open(array_path, "r") as A:
        res = A.query(dims=["timestamp"], attrs=attrs)[tf_minutes, start_ms:end_ms]

    ts = np.asarray(res["timestamp"], dtype=np.int64)
    n = ts.size
    if n == 0:
        print("Không có nến trong khoảng này.")
        return

    order = np.argsort(ts)
    ts = ts[order]
    cols = {a: np.asarray(res[a])[order] for a in attrs}

    header = f"{'timestamp (UTC)':<21} " + " ".join(f"{a:>12}" for a in attrs)
    print(f"tổng nến trong khoảng: {n:,}\n")
    print(header)
    print("-" * len(header))

    def _print_rows(idxs):
        for i in idxs:
            row = " ".join(f"{cols[a][i]:>12.4f}" for a in attrs)
            print(f"{_fmt_ts(int(ts[i])):<21} {row}")

    if n <= MAX_ROWS * 2:
        _print_rows(range(n))
    else:
        _print_rows(range(MAX_ROWS))
        print(f"   ... ({n - MAX_ROWS*2:,} dòng giữa bị ẩn) ...")
        _print_rows(range(n - MAX_ROWS, n))


def main() -> None:
    array_path = _resolve_array_path(SYMBOL)
    if array_path is None:
        print(f"Không tìm thấy symbol '{SYMBOL}' trong SYMBOLS_CONFIG.")
        return

    tf_minutes = _resolve_tf_minutes(TIMEFRAME)
    if tf_minutes is None:
        print(f"Không tìm thấy timeframe '{TIMEFRAME}' trong TIMEFRAMES/CUSTOM_TIMEFRAMES.")
        return

    if tiledb.object_type(array_path) != "array":
        print(f"Array chưa tồn tại: {array_path}")
        return

    print(f"Array: {array_path}")
    overview(array_path, tf_minutes)
    view(array_path, tf_minutes)


if __name__ == "__main__":
    main()
