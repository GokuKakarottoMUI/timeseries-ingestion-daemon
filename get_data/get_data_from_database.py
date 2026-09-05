import os
import time
import picologging as logging
import numpy as np
import tiledb
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingestion.config_fetch_data import (
    TIMEFRAMES,
    CUSTOM_TIMEFRAMES,
    SYMBOLS_CONFIG,
    build_array_path,
)
from ingestion.rwlock import read_lock   # flock SH: chờ nếu writer đang consolidate+vacuum array này
from get_data.config.config_query import QUERY_CONFIG

# Số lần thử lại đọc 1 symbol khi gặp lỗi transient (race vacuum↔read tàn dư / FS hiccup).
# vacuum KHÔNG mất data → reopen thấy fragment hợp nhất đủ data. Hết retry mới warn-drop (không phá luồng).
_READ_ATTEMPTS = 3

# Thứ tự cột của khối nguyên trả về — mọi shape đều derive từ len(BASE_COLS).
BASE_COLS = ("open", "high", "low", "close", "volume")


class GetDataFromDatabase:
    """Reader layer: pulls every active symbol × timeframe out of TileDB for training.

    One thread per symbol, one array handle per symbol (all timeframes sliced off the
    same handle), and each result assembled into a single Fortran-ordered
    ``(N, len(BASE_COLS))`` float64 block whose per-column views are zero-copy.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._array_exists_cache = {}
        self.max_workers = QUERY_CONFIG.get('max_workers', 8)

    def _get_active_symbols(self):
        """
        Filter active symbols với hierarchical check.
        Chỉ return symbols khi TẤT CẢ tầng bố active=true.

        Returns:
            list of tuples: [(market_category, symbol_category, symbol), ...]
        """
        active_symbols = []

        for market_name, market_data in SYMBOLS_CONFIG['market'].items():
            if not market_data.get('active', False):
                continue

            for symbol_category, symbol_cat_data in market_data['symbols_config'].items():
                if not symbol_cat_data.get('active', False):
                    continue

                for symbol, symbol_data in symbol_cat_data['symbols'].items():
                    if symbol_data.get('active', False):
                        active_symbols.append((market_name, symbol_category, symbol))

        return active_symbols

    def _get_active_timeframes(self):
        """
        Filter active standard timeframes.
        Chỉ return timeframes có active=true AND active_featured=true.

        Returns:
            list of tuples: [(tf_name, tf_minutes), ...]
        """
        active_tfs = []

        for tf_name, tf_data in TIMEFRAMES.items():
            if tf_data.get('active', False) and tf_data.get('active_featured', False):
                minutes = tf_data.get('minutes', 0)
                active_tfs.append((tf_name, minutes))

        return active_tfs

    def _get_active_custom_timeframes(self):
        """
        Filter active custom timeframes.
        Chỉ return khi enable=true VÀ interval có active=true AND active_featured=true.

        Returns:
            list of tuples: [(custom_tf_name, tf_minutes), ...]
        """
        active_custom_tfs = []

        if not CUSTOM_TIMEFRAMES.get('enable', False):
            return active_custom_tfs

        custom_intervals = CUSTOM_TIMEFRAMES.get('custom_intervals', {})
        for interval_name, interval_data in custom_intervals.items():
            if interval_data.get('active', False) and interval_data.get('active_featured', False):
                minutes = interval_data.get('minutes', 0)
                active_custom_tfs.append((interval_name, minutes))

        return active_custom_tfs

    def _array_exists(self, array_path):
        """
        Check TileDB array tồn tại.
        Cache results để tránh check lặp lại (giờ chỉ gọi 1 lần/symbol).

        Args:
            array_path: Full path to TileDB array

        Returns:
            bool: True nếu array tồn tại
        """
        if array_path in self._array_exists_cache:
            return self._array_exists_cache[array_path]

        exists = os.path.exists(array_path) and tiledb.object_type(array_path) == "array"
        self._array_exists_cache[array_path] = exists
        return exists

    def _date_to_timestamp(self, year, month, day):
        """
        Convert date to timestamp milliseconds.

        Args:
            year: int
            month: int
            day: int

        Returns:
            int: timestamp in milliseconds
        """
        dt = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _assemble_block(self, data):
        """Pack one TileDB result into a single Fortran-ordered float64 block.

        One allocation, one vectorized C-level assignment per column, no intermediate
        arrays. Fortran order makes every column contiguous, so the by-name entries in
        the returned dict are zero-copy views into the same buffer (and stay zero-copy
        when handed to torch downstream).

        ``timestamp`` is kept separate as int64: ms-epoch (~1.7e12) exceeds the exact
        integer range of float64's mantissa-free use here, and mixing it into the float
        block would cost precision for no benefit.

        Dtype is float64 end to end, exactly as stored. Down-casting to float32 here
        would be lossy in a way that matters: with a 0.01 exchange tick, float32's ulp
        already exceeds one cent above ~131 072 USD, so two distinct prices can collapse
        onto the same value. Any down-cast belongs at the very end of a feature pipeline,
        once, not at the storage boundary.

        Args:
            data: dict of numpy arrays returned by TileDB (float64 attrs + int64 'timestamp').

        Returns:
            dict with ``timestamp`` (int64[N]), ``block``
            (float64[N, len(BASE_COLS)], Fortran-order), ``columns`` (BASE_COLS) and one
            zero-copy column view per name in BASE_COLS.
        """
        ts = data['timestamp']                                   # int64, view từ TileDB
        n = ts.shape[0]
        # Shape derive từ BASE_COLS — thêm/bớt cột chỉ cần sửa BASE_COLS, không sót chỗ nào.
        block = np.empty((n, len(BASE_COLS)), dtype=np.float64, order='F')
        for j, col in enumerate(BASE_COLS):
            block[:, j] = data[col]                              # float64 → float64, không ép kiểu

        result = {'timestamp': ts, 'block': block, 'columns': BASE_COLS}
        for j, col in enumerate(BASE_COLS):
            result[col] = block[:, j]                            # view cột (contiguous vì F-order)
        return result

    def _query_symbol(self, market_category, symbol_category, symbol,
                      array_path, tfs, start_timestamp):
        """
        Đọc TẤT CẢ timeframe của 1 symbol — MỞ ARRAY ĐÚNG 1 LẦN, slice từng TF trên cùng handle.

        - TileDB schema cell_order='row-major' ⇒ kết quả ĐÃ tăng dần theo timestamp → KHÔNG sort.
        - TF rỗng (không có nến trong dải) → bỏ qua, KHÔNG coi là lỗi.
        - Mỗi TF bọc try/except: lỗi đọc → gom vào `errors`, tiếp tục TF khác (không nuốt im lặng).

        Args:
            market_category, symbol_category, symbol: định danh symbol
            array_path: full path tới TileDB array của symbol
            tfs: list[(tf_name, tf_minutes)]
            start_timestamp: int (ms) hoặc None (query toàn bộ)

        Returns:
            tuple (symbol_result, errors):
              symbol_result: {tf_name: blockdict}  (blockdict do _assemble_block trả)
              errors:        list[(symbol_id, tf_name, error_str)]
        """
        sym_id = f"{market_category}/{symbol_category}/{symbol}"

        if not self._array_exists(array_path):
            return {}, [(sym_id, tf_name, "array not found") for tf_name, _ in tfs]

        # read_lock (flock SH) giữ SUỐT 1 lần đọc trọn symbol → writer (continuous_fetch) đang vacuum array
        # này thì CHỜ; ta không đọc giữa lúc fragment bị xoá → diệt race vacuum↔read liên-process.
        # Retry dự phòng: nếu open/đọc vẫn ném lỗi (race tàn dư / FS hiccup) → reopen (vacuum KHÔNG mất data
        # ⇒ fragment list mới có đủ). Per-TF lỗi lẻ vẫn warn-only inner, KHÔNG retry cả symbol vì 1 TF hỏng thật.
        last_exc = None
        for attempt in range(_READ_ATTEMPTS):
            symbol_result = {}
            tf_errors = []
            try:
                with read_lock(array_path):
                    with tiledb.open(array_path, mode='r') as array:
                        for tf_name, tf_minutes in tfs:
                            try:
                                if start_timestamp is not None:
                                    data = array[tf_minutes, start_timestamp:]
                                else:
                                    data = array[tf_minutes, :]

                                if data['timestamp'].shape[0] == 0:
                                    continue  # TF rỗng cho symbol này — không phải lỗi

                                symbol_result[tf_name] = self._assemble_block(data)
                            except Exception as e:
                                tf_errors.append((sym_id, tf_name, str(e)))
                return symbol_result, tf_errors   # open OK → trả về (per-TF lỗi lẻ đã gom warn-only)
            except Exception as e:
                last_exc = e
                if attempt < _READ_ATTEMPTS - 1:
                    time.sleep(0.2 * (attempt + 1))   # backoff nhẹ, để qua khe vacuum rồi reopen

        # hết retry → mở array thất bại thật → toàn bộ TF của symbol rớt (warn-only, KHÔNG phá luồng)
        return {}, [(sym_id, tf_name, f"open failed sau {_READ_ATTEMPTS} lần: {last_exc}") for tf_name, _ in tfs]

    def query_data_for_training(self):
        """Read every active symbol × timeframe out of TileDB, in parallel.

        One task per SYMBOL, so each array is opened once and every timeframe is sliced
        off the same handle. Only symbols and timeframes whose whole ``active`` hierarchy
        is true are read; custom timeframes join automatically when enabled in the JSON
        config. Mode, ``start_date`` and ``max_workers`` come from ``query_config.yaml``.

        Read errors are never swallowed silently: they are collected and logged as one
        warning summary at the end of the run. This method does not raise — a failing
        symbol is dropped from the result rather than aborting the whole read.

        Returns:
            Nested dict ``{market_category: {symbol_category: {symbol: {timeframe: block}}}}``
            where each ``block`` is the dict returned by ``_assemble_block``: ``timestamp``
            (int64[N]), ``block`` (float64[N, len(BASE_COLS)], Fortran-order), ``columns``,
            and one zero-copy column view per name in BASE_COLS.
        """
        self.logger.info("Starting query data for training (multi-threaded, per-symbol)...")

        # Read query config từ YAML - CHỈ 1 FLAG "unlimited"
        start_timestamp = None
        unlimited = QUERY_CONFIG['query_mode']['unlimited']

        if not unlimited:
            # unlimited = false → query from start_date
            start_date = QUERY_CONFIG['start_date']
            start_timestamp = self._date_to_timestamp(
                start_date['year'],
                start_date['month'],
                start_date['day']
            )
            self.logger.info(
                f"Query mode: from_date -> "
                f"{start_date['year']}-{start_date['month']:02d}-{start_date['day']:02d}, "
                f"timestamp: {start_timestamp}"
            )
        else:
            # unlimited = true → query all
            self.logger.info("Query mode: unlimited (query toàn bộ data)")

        active_symbols = self._get_active_symbols()
        self.logger.info(f"Found {len(active_symbols)} active symbols")

        active_tfs = self._get_active_timeframes()
        self.logger.info(f"Found {len(active_tfs)} active standard timeframes")

        active_custom_tfs = self._get_active_custom_timeframes()
        if active_custom_tfs:
            self.logger.info(f"Found {len(active_custom_tfs)} active custom timeframes")

        all_timeframes = active_tfs + active_custom_tfs

        if not active_symbols:
            self.logger.warning("No active symbols found. Check config.")
            return {}

        if not all_timeframes:
            self.logger.warning("No active timeframes found. Check config.")
            return {}

        self.logger.info(
            f"Submitting {len(active_symbols)} symbol tasks to thread pool "
            f"(max_workers={self.max_workers})"
        )

        result = {}
        all_errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sym = {
                executor.submit(
                    self._query_symbol,
                    market_category, symbol_category, symbol,
                    build_array_path(market_category, symbol_category, symbol),
                    all_timeframes,
                    start_timestamp
                ): (market_category, symbol_category, symbol)
                for market_category, symbol_category, symbol in active_symbols
            }

            for future in as_completed(future_to_sym):
                market_category, symbol_category, symbol = future_to_sym[future]
                try:
                    symbol_result, errors = future.result()
                except Exception as e:
                    # _query_symbol đã tự bọc lỗi đọc; tới đây nghĩa là task crash ngoài dự kiến
                    all_errors.append((f"{market_category}/{symbol_category}/{symbol}", "*", f"task crashed: {e}"))
                    continue

                all_errors.extend(errors)

                if symbol_result:
                    # gộp ở MAIN THREAD → không mutate dict chung trong worker (khỏi race)
                    result.setdefault(market_category, {}).setdefault(symbol_category, {})[symbol] = symbol_result
                    for tf_name, d in symbol_result.items():
                        self.logger.info(
                            f"Loaded {market_category}/{symbol_category}/{symbol} @ {tf_name}: "
                            f"{len(d['timestamp'])} candles"
                        )

        # Gom lỗi → CẢNH BÁO tổng kết cuối run (không nuốt im lặng, KHÔNG raise / KHÔNG phá vỡ luồng)
        if all_errors:
            summary = "\n".join(f"  - {sid} @ {tf}: {err}" for sid, tf, err in all_errors)
            self.logger.warning(f"Query gặp {len(all_errors)} lỗi đọc (vẫn tiếp tục, trả phần đọc được):\n{summary}")

        self.logger.info("Query completed.")
        return result
