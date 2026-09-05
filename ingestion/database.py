import os
import sys
import time
import threading
import tiledb
import numpy as np
import picologging as logging
from concurrent.futures import ThreadPoolExecutor, as_completed, wait as futures_wait

from ingestion.config_fetch_data import (
    HISTORICAL_DATA_CONFIG, SYMBOLS_CONFIG, DATABASE_STRUCTURE,
    DATABASE_ROOT_PATH, DB_GROUP_NAME, DB_GROUP_ROOT, build_array_path,
)
from ingestion.rwlock import write_lock   # flock EX: chặn reader (get_data) trong lúc consolidate+vacuum (race liên-process)

logger = logging.getLogger("database")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# ── Write rate config đọc 1 lần ở module level ────────────────────────────────
_WR                = HISTORICAL_DATA_CONFIG["write_rate"]
_WR_UNLIMITED      = _WR["unlimited"]["value"]
_WR_DEFAULT        = _WR["default"]["value"]      # nến/s khi dùng start_date
_WR_FETCH_ALL      = _WR["fetch_all"]["value"]     # nến/s khi fetch_all

_MW                = HISTORICAL_DATA_CONFIG["multi_write"]
_MW_ACTIVE         = _MW["active"]["value"]
_MW_UNLIMITED      = _MW["unlimited"]["value"]
_MW_MAX_WORKERS    = _MW["max_workers"]["value"]

_FETCH_ALL_ACTIVE  = HISTORICAL_DATA_CONFIG["fetch_all"]["active"]["value"]

# Pre-compute 1 lần — tránh ternary lặp lại ở _apply_write_rate + _log_write_mode
_WRITE_RATE        = _WR_FETCH_ALL if _FETCH_ALL_ACTIVE else _WR_DEFAULT


class DatabaseManager:
    """Writer layer: creates the TileDB group/array tree and batch-writes OHLCV.

    One sparse 2D array per symbol, dimensions ``(timeframe_minutes, timestamp)``,
    five float64 attributes ``open/high/low/close/volume``. Writes are serialised
    per array path and parallel across symbols.
    """

    def __init__(self, calc_tf):
        self.calc_tf = calc_tf
        self.root_path = DATABASE_ROOT_PATH
        self._write_rate_logged = False
        self._write_rate_lock   = threading.Lock()   # bảo vệ _write_rate_logged khỏi race

        # E2: cache các array đã xác nhận tồn tại — tránh tiledb.object_type() mỗi lần ghi
        self._existing_arrays: set[str] = set()
        # P3.15: lock per-array-path — ghi cùng 1 array tuần tự (không conflict fragment),
        # array khác (symbol khác) vẫn song song. _locks_guard bảo vệ dict.
        self._array_locks: dict[str, threading.Lock] = {}
        self._locks_guard  = threading.Lock()
        # Pool ghi TÁI DÙNG — continuous flush mỗi TF mỗi chu kỳ, dựng pool mới mỗi lần
        # = thread churn ⇒ arena glibc churn ⇒ RSS ratchet. Cỡ pool là HẰNG SỐ y hệt
        # `ThreadPoolExecutor(max_workers=None|_MW_MAX_WORKERS)` cũ nên concurrency không đổi.
        self._write_executor: ThreadPoolExecutor | None = None
        self._write_exec_guard = threading.Lock()

        config = tiledb.Config({
            "sm.compute_concurrency_level": "8",
            "sm.io_concurrency_level": "8"
        })
        self.ctx = tiledb.Ctx(config)

    def _get_array_lock(self, array_path: str) -> threading.Lock:
        """Lấy/ tạo lock cho 1 array_path (P3.15) — đảm bảo ghi cùng array không đua nhau."""
        lock = self._array_locks.get(array_path)
        if lock is None:
            with self._locks_guard:
                lock = self._array_locks.get(array_path)
                if lock is None:
                    lock = threading.Lock()
                    self._array_locks[array_path] = lock
        return lock

    # ── Database structure ─────────────────────────────────────────────────────

    def create_database_structure(self):
        """Create the TileDB group tree, then every symbol array in parallel.

        Groups are created parent→child sequentially (TileDB requires the parent
        group to exist first); the leaf arrays are independent so they go through
        a thread pool.
        """
        try:
            root_path  = DB_GROUP_ROOT
            market_cats = DATABASE_STRUCTURE[DB_GROUP_NAME]["market_categories"]

            group_paths = sorted({
                path
                for mk, md in market_cats.items() if md.get("active")
                for path in (
                    f"{root_path}/{mk}",
                    *(
                        f"{root_path}/{mk}/{sk}"
                        for sk, sd in md["symbol_categories"].items() if sd.get("active")
                    )
                )
            })

            if tiledb.object_type(root_path, ctx=self.ctx) != "group":
                tiledb.group_create(root_path, ctx=self.ctx)
                logger.info(f"Đã tạo group gốc: {root_path}")

            for gp in group_paths:
                if tiledb.object_type(gp, ctx=self.ctx) != "group":
                    tiledb.group_create(gp, ctx=self.ctx)
                    logger.info(f"Đã tạo group: {gp}")

            array_paths = [
                f"{root_path}/{mk}/{sk}/{ak}"
                for mk, md in market_cats.items() if md.get("active")
                for sk, sd in md["symbol_categories"].items() if sd.get("active")
                for ak, ad in sd["arrays"].items() if ad.get("active")
            ]

            def _check_and_create(path: str):
                if tiledb.object_type(path, ctx=self.ctx) != "array":
                    self.create_symbol_array(path)
                self._existing_arrays.add(path)   # E2: nạp cache array đã tồn tại

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_check_and_create, p): p for p in array_paths}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        logger.error(f"Lỗi tạo array {futures[fut]}: {exc}")
                        raise exc

        except Exception as e:
            logger.error(f"Lỗi tạo database: {str(e)}")
            raise

    def create_symbol_array(self, array_path: str) -> None:
        """Create the sparse 2D array for one symbol.

        Dimensions ``timeframe_minutes`` (int32) and ``timestamp`` (int64 ms epoch)
        let every timeframe of a symbol live in a single array, so one open handle
        serves all of them on the read path.
        """
        try:
            data_filters = tiledb.FilterList([tiledb.ZstdFilter(level=7)])
            dim_filters  = tiledb.FilterList([tiledb.DoubleDeltaFilter(), tiledb.ZstdFilter(level=7)])

            max_timestamp         = np.iinfo(np.int64).max - 10000
            max_timeframe_minutes = 5270400

            domain = tiledb.Domain(
                tiledb.Dim(name="timeframe_minutes", domain=(0, max_timeframe_minutes),
                           tile=100, dtype=np.int32, filters=dim_filters),
                tiledb.Dim(name="timestamp", domain=(0, max_timestamp),
                           tile=10000, dtype=np.int64, filters=dim_filters)
            )

            schema = tiledb.ArraySchema(
                domain=domain,
                sparse=True,
                cell_order='row-major',
                tile_order='row-major',
                capacity=10000,
                allows_duplicates=False,
                attrs=[
                    tiledb.Attr(name="open",     dtype=np.float64, filters=data_filters),
                    tiledb.Attr(name="high",     dtype=np.float64, filters=data_filters),
                    tiledb.Attr(name="low",      dtype=np.float64, filters=data_filters),
                    tiledb.Attr(name="close",    dtype=np.float64, filters=data_filters),
                    tiledb.Attr(name="volume",   dtype=np.float64, filters=data_filters),
                ],
                coords_filters=dim_filters,
                offsets_filters=dim_filters
            )

            tiledb.Array.create(array_path, schema, ctx=self.ctx)
            logger.info(f"Array 2D {array_path} đã được tạo")

        except Exception as e:
            logger.error(f"Lỗi tạo array {array_path}: {str(e)}")
            raise

    # ── Write helpers ──────────────────────────────────────────────────────────

    def _apply_write_rate(self, n_candles: int, write_duration: float,
                          symbol: str, timeframe_name: str) -> None:
        """
        Log + sleep nếu cần để tuân thủ write_rate.
        Chỉ gọi khi _WR_UNLIMITED = False.
        """
        write_speed = n_candles / write_duration if write_duration > 0 else 0

        required_duration = n_candles / _WRITE_RATE
        if write_duration < required_duration:
            time.sleep(required_duration - write_duration)
            total_duration = required_duration
        else:
            total_duration = write_duration

        actual_rate = n_candles / total_duration
        logger.info(
            f"BATCH WRITE: {n_candles} candles cho {symbol} ({timeframe_name}) "
            f"trong {total_duration:.3f}s "
            f"(TileDB: {write_speed:.0f} c/s, Giới hạn: {actual_rate:.0f} c/s)"
        )

    def _log_write_mode(self) -> None:
        """Ghi log mode 1 lần duy nhất — thread-safe, double-checked locking."""
        if self._write_rate_logged:          # fast path — không lock sau lần đầu
            return
        with self._write_rate_lock:
            if self._write_rate_logged:      # re-check trong lock tránh race
                return
            if _WR_UNLIMITED:
                logger.info("Write mode: unlimited (không giới hạn tốc độ ghi)")
            else:
                logger.info(f"Write mode: limited — {_WRITE_RATE} nến/s")
            self._write_rate_logged = True

    def _write_single(self, array_path: str, timeframe_minutes: int,
                      candle_array: np.ndarray, symbol: str, timeframe_name: str,
                      timestamps_ms: np.ndarray) -> int:
        """
        Ghi 1 array vào TileDB — column views contiguous, không copy thừa.
        candle_array: (N, 6) float64 [timestamp, open, high, low, close, volume]
        timestamps_ms: (N,) int64 — truyền vào từ batch_insert_data, không recompute.

        TileDB nhận buffer THEO CỘT và cần từng cột contiguous. candle_array tới đây
        là C-order (np.array/np.concatenate ở tầng fetch đều trả C-order) nên
        arr[:, k] bị strided → TileDB sẽ tự copy LẺ TỪNG CỘT. Vì vậy chuyển 1 lần
        sang F-order và CHỈ 5 cột thực sự ghi (bỏ cột 0 = timestamp, đã có
        timestamps_ms int64 riêng cho dimension): 1 cấp phát (N,5) duy nhất thay cho
        5 lần copy rời rạc, sau đó mọi cột là view contiguous.
        """
        cols = np.asfortranarray(candle_array[:, 1:6])
        timeframe_array = np.full(len(candle_array), timeframe_minutes, dtype=np.int32)

        array_data = {
            'open':   cols[:, 0],
            'high':   cols[:, 1],
            'low':    cols[:, 2],
            'close':  cols[:, 3],
            'volume': cols[:, 4],
        }

        # P3.15: serialize ghi CÙNG 1 array (không conflict fragment); array khác vẫn song song
        start_time = time.perf_counter()
        with self._get_array_lock(array_path):
            with tiledb.open(array_path, mode='w', ctx=self.ctx) as array:
                array[timeframe_array, timestamps_ms] = array_data
        write_duration = time.perf_counter() - start_time

        n = len(candle_array)
        if _WR_UNLIMITED:
            write_speed = n / write_duration if write_duration > 0 else 0
            logger.info(
                f"Batch write: {n} candles cho {symbol} ({timeframe_name}) "
                f"trong {write_duration:.3f}s ({write_speed:.0f} candles/s)"
            )
        else:
            self._apply_write_rate(n, write_duration, symbol, timeframe_name)

        return n

    # ── Public API ─────────────────────────────────────────────────────────────

    def batch_insert_data(self, market_category: str, symbol_category: str, symbol: str,
                          timeframe_name: str, candle_array: np.ndarray) -> int:
        """Write one symbol/timeframe batch into TileDB.

        Args:
            candle_array: ``(N, 6)`` float64 ``[timestamp, open, high, low, close, volume]``.

        Returns:
            Number of candles written (0 for an empty batch).
        """
        try:
            if len(candle_array) == 0:
                return 0

            array_path = build_array_path(market_category, symbol_category, symbol)

            # E2: chỉ object_type() lần đầu cho mỗi array → cache; bỏ syscall mỗi lần ghi
            if array_path not in self._existing_arrays:
                if tiledb.object_type(array_path, ctx=self.ctx) != "array":
                    logger.warning(f"Array {array_path} chưa tồn tại, đang tạo mới...")
                    self.create_symbol_array(array_path)
                self._existing_arrays.add(array_path)

            timeframe_minutes = self.calc_tf._get_timeframe_minutes(timeframe_name)
            if timeframe_minutes == 0:
                logger.error(f"Không tìm thấy minutes cho timeframe {timeframe_name}")
                return 0

            # Dedup theo timestamp — tính 1 lần, tái dùng cho cả dedup lẫn write
            timestamps_ms = candle_array[:, 0].astype(np.int64)
            _, unique_indices = np.unique(timestamps_ms, return_index=True)
            if len(unique_indices) < len(candle_array):
                logger.warning(
                    f"Phát hiện {len(candle_array) - len(unique_indices)} "
                    f"timestamps trùng lặp, đang loại bỏ..."
                )
                candle_array  = candle_array[unique_indices]
                timestamps_ms = timestamps_ms[unique_indices]   # slice, không recompute

            self._log_write_mode()
            return self._write_single(
                array_path, timeframe_minutes, candle_array,
                symbol, timeframe_name, timestamps_ms
            )

        except Exception as e:
            logger.error(f"Lỗi batch insert: {str(e)}")
            raise

    def batch_insert_multi(self, tasks: list[tuple]) -> dict[str, int]:
        """
        Ghi nhiều symbols đồng thời theo config multi_write.

        tasks: list of (market_category, symbol_category, symbol, timeframe_name, candle_array)
        Returns: dict {symbol: n_candles_written}

        multi_write.active = false  → ghi tuần tự
        multi_write.active = true:  → pool TÁI DÙNG (_get_write_executor), cỡ pool bằng
            unlimited = true        → mặc định ThreadPoolExecutor: min(32, cpu_count + 4)
            unlimited = false       → multi_write.max_workers
        """
        results: dict[str, int] = {}

        if not _MW_ACTIVE:
            # Tuần tự
            for market_cat, sym_cat, symbol, tf_name, arr in tasks:
                results[symbol] = self.batch_insert_data(market_cat, sym_cat, symbol, tf_name, arr)
            return results

        # Song song — pool TÁI DÙNG (xem _get_write_executor)
        def _task(args):
            market_cat, sym_cat, symbol, tf_name, arr = args
            return symbol, self.batch_insert_data(market_cat, sym_cat, symbol, tf_name, arr)

        pool    = self._get_write_executor()
        futures = {pool.submit(_task, t): t[2] for t in tasks}
        failure = None
        try:
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    # Cancel tất cả futures chưa start — giải phóng sớm, không chờ hết
                    for f in futures:
                        f.cancel()
                    logger.error(f"Lỗi ghi symbol {futures[fut]}: {exc}")
                    failure = exc
                    break
                symbol, n = fut.result()
                results[symbol] = n
        finally:
            # Pool sống tiếp nên phải TỰ join — giữ đúng ngữ nghĩa cũ: không có task ghi
            # nào còn chạy nền sau khi hàm này trả về / ném lỗi.
            futures_wait(futures)

        if failure is not None:
            raise failure

        return results

    def _get_write_executor(self) -> ThreadPoolExecutor:
        """
        Pool ghi dùng chung cả run — thay cho pool dựng mới mỗi lần flush.
        Cỡ pool = HẰNG SỐ, đúng bằng cỡ pool cũ:
          - unlimited → mặc định của ThreadPoolExecutor: min(32, cpu_count + 4)
          - limited   → _MW_MAX_WORKERS
        nên số worker chạy song song KHÔNG đổi so với trước.
        """
        pool = self._write_executor
        if pool is not None:
            return pool
        with self._write_exec_guard:
            if self._write_executor is None:
                max_workers = (
                    min(32, (os.cpu_count() or 1) + 4) if _MW_UNLIMITED else _MW_MAX_WORKERS
                )
                self._write_executor = ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="tiledb_write"
                )
                logger.info(f"Pool ghi TileDB tái dùng: {max_workers} worker")
            return self._write_executor

    def query_candles(
        self,
        array_path:        str,
        timeframe_minutes: int,
        start_ts:          int,
        end_ts:            int,
    ) -> np.ndarray:
        """Read OHLCV of one timeframe back out of TileDB.

        Used as the ``source_array`` for ``_aggregate_candles_batch_numpy()`` when
        building custom timeframes out of an already-stored base timeframe.

        Returns:
            ``(N, 6)`` float64 ``[timestamp, open, high, low, close, volume]``, or
            ``(0, 6)`` when the array does not exist or the range holds no candle.
        """
        _EMPTY = np.empty((0, 6), dtype=np.float64)

        try:
            if tiledb.object_type(array_path, ctx=self.ctx) != "array":
                logger.warning(f"Array {array_path} không tồn tại, bỏ qua query")
                return _EMPTY

            with tiledb.open(array_path, mode='r', ctx=self.ctx) as array:
                res = array.query(
                    dims  = ['timestamp'],
                    attrs = ['open', 'high', 'low', 'close', 'volume'],
                )[timeframe_minutes, start_ts:end_ts]

            timestamps = res['timestamp']   # (N,) int64
            n = len(timestamps)
            if n == 0:
                return _EMPTY

            # Pre-allocate (N, 6) — gán thẳng từng col, không copy thừa
            out = np.empty((n, 6), dtype=np.float64)
            out[:, 0] = timestamps          # int64 → float64: đồng nhất với pipeline
            out[:, 1] = res['open']
            out[:, 2] = res['high']
            out[:, 3] = res['low']
            out[:, 4] = res['close']
            out[:, 5] = res['volume']

            logger.info(
                f"Query {n} candles từ {array_path} "
                f"(tf={timeframe_minutes}, {start_ts}→{end_ts})"
            )
            return out

        except Exception as e:
            logger.error(f"Lỗi query candles từ {array_path}: {str(e)}")
            return _EMPTY

    def consolidate_array(self, array_path: str) -> None:
        """Consolidate + Vacuum TileDB array để gộp fragments."""
        try:
            # tiledb.object_type thay os.path.exists
            if tiledb.object_type(array_path, ctx=self.ctx) != "array":
                logger.warning(f"Array {array_path} không tồn tại, bỏ qua consolidate")
                return

            logger.info(f"Bắt đầu consolidate {array_path}...")
            start_time = time.perf_counter()

            # Dùng self.ctx thay vì tạo Config mới.
            # write_lock (flock EX): vacuum XOÁ fragment vật lý → chặn reader (get_data, process khác) trong
            # lúc này (reader đang đọc thì CHỜ; reader mới bị chặn) → diệt race vacuum↔read liên-process.
            with write_lock(array_path):
                tiledb.consolidate(array_path, config=self.ctx.config(), ctx=self.ctx)
                tiledb.vacuum(array_path, ctx=self.ctx)

            logger.info(
                f"Consolidate hoàn thành trong "
                f"{time.perf_counter() - start_time:.2f}s cho {array_path}"
            )

        except Exception as e:
            logger.error(f"Lỗi consolidate {array_path}: {str(e)}")
