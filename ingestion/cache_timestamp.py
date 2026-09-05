from typing import Dict
import threading
import orjson
import os
import tiledb
import numpy as np
import picologging as logging
import sys

from ingestion.config_fetch_data import HISTORICAL_DATA_CONFIG
from ingestion.timestamp_scanner import TimestampScanner

logger = logging.getLogger("cache_timestamp")
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

# ── Module-level sentinels — tính 1 lần, tái sử dụng xuyên suốt ───────────────
# Tuple (np.int64, np.int64) — không tạo dict, không unpack key string
_EMPTY_RANGE:     tuple[np.int64, np.int64] = (np.int64(0), np.int64(0))
# np.ndarray (0,2) int64 — return khi không có intervals
_EMPTY_INTERVALS: np.ndarray = np.empty((0, 2), dtype=np.int64)
# Upper bound timestamp cố định
_TS_MAX: np.int64 = np.int64(np.iinfo(np.int64).max - 10000)
_ZERO:   np.int64 = np.int64(0)
# np.ndarray (0,) int64 — return khi probe không thấy timestamp nào
_EMPTY_TS: np.ndarray = np.empty(0, dtype=np.int64)

# ── Scan mode flag đọc 1 lần — tránh .get() chain mỗi lần gọi ─────────────────
_SCAN_MODE_ACTIVE: bool = (
    HISTORICAL_DATA_CONFIG
    .get("scan_missing_timestamps", {})
    .get("active", {})
    .get("value", False)
)


class CacheManager:
    def __init__(self, cache_file: str, calc_tf, logger: logging.Logger):
        self.cache_file        = cache_file
        self.calc_tf           = calc_tf
        self.timestamp_scanner = TimestampScanner()
        self.logger            = logger
        self._cache_lock       = threading.Lock()   # bảo vệ read-modify-write trên cache RAM
        # A3: nạp cache vào RAM 1 lần; mọi update vào RAM, ghi file 1 lần cuối (flush()).
        self._cache: Dict[str, Dict[str, int]] = self._load_cache()
        self._dirty            = False

    # ── I/O helpers ────────────────────────────────────────────────────────────

    def _load_cache(self) -> Dict[str, Dict[str, int]]:
        """Đọc cache từ file — binary mode, không decode thừa."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return orjson.loads(f.read())
            return {}
        except Exception as e:
            self.logger.warning(f"Lỗi đọc cache {self.cache_file}: {str(e)}, trả về cache rỗng")
            return {}

    def _save_cache(self, cache: Dict[str, Dict[str, int]]) -> None:
        """Lưu cache — binary mode, ghi bytes thẳng.

        np.int64 cần ép int() trước orjson — đây là chỗ DUY NHẤT convert, bắt buộc.
        """
        try:
            with open(self.cache_file, 'wb') as f:
                f.write(orjson.dumps(cache))
            self.logger.debug(f"Đã cập nhật cache {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Lỗi lưu cache {self.cache_file}: {str(e)}")

    def flush(self) -> None:
        """A3: ghi cache RAM xuống file (gọi cuối mỗi symbol + cuối run). Chỉ ghi khi dirty."""
        with self._cache_lock:
            if self._dirty:
                self._save_cache(self._cache)
                self._dirty = False

    # ── TileDB query helpers ────────────────────────────────────────────────────

    def _get_timestamp_range(self, array_path: str, timeframe_minutes: int) -> tuple[np.int64, np.int64]:
        """
        Lấy (timestamp_min, timestamp_max) từ DB cho timeframe_minutes.
        Return tuple (np.int64, np.int64) — không tạo dict, không unpack key.
        Chỉ query DIMENSIONS, KHÔNG query attributes.
        """
        try:
            if tiledb.object_type(array_path) != "array":
                return _EMPTY_RANGE

            with tiledb.open(array_path, mode='r') as array:
                timestamps = array.query(
                    dims=['timeframe_minutes', 'timestamp'], attrs=[]
                )[timeframe_minutes, :]['timestamp']

                if len(timestamps) == 0:
                    return _EMPTY_RANGE

                # np.min/max trả về np.int64 thẳng — không ép int() thừa
                return (np.min(timestamps), np.max(timestamps))

        except Exception as e:
            self.logger.warning(
                f"Lỗi truy vấn timestamp range từ {array_path} "
                f"(tf={timeframe_minutes}): {str(e)}, trả về 0"
            )
            return _EMPTY_RANGE

    def _check_timestamp_exists(self, array_path: str, timestamp: np.int64, timeframe_minutes: int) -> bool:
        """
        Kiểm tra timestamp tồn tại — chỉ query DIMENSIONS, KHÔNG query attributes.
        """
        try:
            if tiledb.object_type(array_path) != "array":
                return False

            with tiledb.open(array_path, mode='r') as array:
                try:
                    data = array.query(
                        dims=['timeframe_minutes', 'timestamp'], attrs=[]
                    ).multi_index[timeframe_minutes, timestamp]
                    return len(data['timestamp']) > 0
                except IndexError:
                    return False
        except Exception as e:
            self.logger.warning(
                f"Lỗi kiểm tra timestamp {timestamp}, tf={timeframe_minutes}: {str(e)}"
            )
            return False

    def _probe_cache_state(
        self, array_path: str, timeframe_minutes: int,
        timestamp_min: np.int64, timestamp_max: np.int64,
    ) -> tuple[bool, bool, np.ndarray]:
        """
        GỘP 3 lệnh đọc của nhánh "cache có dữ liệu" thành **1 query duy nhất**:
        _check_timestamp_exists(min) + _check_timestamp_exists(max) + range (max+1 → _TS_MAX).

        Vì sao gộp: libtiledb 0.36.1 RÒ ~15KB mỗi lệnh đọc sparse (đo bằng mallinfo2:
        uordblks tăng tuyến tính, fordblks đứng yên ⇒ cấp phát không free, KHÔNG phải
        phân mảnh; `del ctx` không lấy lại được). Số query là thứ DUY NHẤT điều khiển
        được tốc độ rò → 3 open + 3 query mỗi TF mỗi chu kỳ ⇒ còn 1 open + 1 query.

        multi_index lấy 2 range: [min,min] (điểm) và [max,_TS_MAX] (max + mọi nến mới
        hơn) → suy ra CẢ BA thông tin cũ. GIÁ TRỊ KHÔNG ĐỔI.

        Returns: (min_exists, max_exists, timestamps_lớn_hơn_max)
        """
        try:
            if tiledb.object_type(array_path) != "array":
                return False, False, _EMPTY_TS

            # min >= max (cache 1 điểm hoặc suy biến min>max) → [max, _TS_MAX] đã phủ
            # CẢ min lẫn max, khỏi 2 range chồng nhau (TileDB không nhận range trùng).
            if timestamp_min < timestamp_max:
                ranges = [slice(timestamp_min, timestamp_min), slice(timestamp_max, _TS_MAX)]
            else:
                ranges = [slice(timestamp_max, _TS_MAX)]

            with tiledb.open(array_path, mode='r') as array:
                found = array.query(
                    dims=['timestamp'], attrs=[]
                ).multi_index[timeframe_minutes, ranges]['timestamp']

            if len(found) == 0:
                return False, False, _EMPTY_TS

            min_exists = bool((found == timestamp_min).any())   # C-level, mảng vài phần tử
            max_exists = bool((found == timestamp_max).any())
            return min_exists, max_exists, found[found > timestamp_max]

        except Exception as e:
            self.logger.warning(
                f"Lỗi probe cache state {array_path} (tf={timeframe_minutes}): {str(e)}"
            )
            return False, False, _EMPTY_TS

    # ── Interval builder helpers ────────────────────────────────────────────────

    @staticmethod
    def _build_intervals(*pairs: tuple[int, int]) -> np.ndarray:
        """
        Build np.ndarray (N,2) int64 từ các cặp (start, end).
        Tránh tạo list rồi convert — stack thẳng.
        """
        return np.array(pairs, dtype=np.int64)

    # ── Main logic ──────────────────────────────────────────────────────────────

    def get_fetch_timestamps(
        self, array_path: str, timeframe_name: str,
        end_ts: int, start_ts: int
    ) -> np.ndarray:
        """
        Hàm chính — quyết định cache mode hay scan mode.
        start_ts: đã tính chuẩn từ api_fetch, KHÔNG fallback, KHÔNG tự tính lại.
        Returns: np.ndarray (N,2) int64 — các intervals cần fetch
        """
        if _SCAN_MODE_ACTIVE:
            self.logger.info(f"SCAN MODE: Quét dimension tìm timestamps thiếu cho {timeframe_name}")
            return self.get_fetch_timestamps_by_scan(array_path, timeframe_name, end_ts, start_ts)
        else:
            self.logger.debug(f"CACHE MODE: Sử dụng cache xác định intervals cho {timeframe_name}")
            return self.get_fetch_timestamps_by_cache(array_path, timeframe_name, end_ts, start_ts)

    def get_fetch_timestamps_by_cache(
        self, array_path: str, timeframe_name: str,
        end_ts: int, starts_ts: int
    ) -> np.ndarray:
        """
        Xác định intervals cần fetch dựa trên cache và database.
        Returns: np.ndarray (N,2) int64

        - _load_cache() gọi 1 lần duy nhất
        - _get_timestamp_range() trả về tuple (np.int64, np.int64) — không dict
        - Output là np.ndarray (N,2) int64 — không List[Tuple]
        - _save_cache() ép int() tại chỗ gán vào dict — chỗ duy nhất convert bắt buộc
        - _cache_lock bảo vệ toàn bộ read-modify-write — tránh race condition
        """
        array_name        = os.path.basename(array_path)
        timeframe_minutes = self.calc_tf._get_timeframe_minutes(timeframe_name)

        if timeframe_minutes == 0:
            self.logger.error(f"Không tìm thấy minutes cho timeframe {timeframe_name}")
            return _EMPTY_INTERVALS

        cache_key = f"{array_name}_tf{timeframe_minutes}"

        with self._cache_lock:
            # A3: dùng cache RAM (đã nạp 1 lần ở __init__) — không đọc file mỗi lần
            cache           = self._cache
            config_start_ts = starts_ts
            array_cache     = cache.get(cache_key, {'timestamp_min': 0, 'timestamp_max': 0})
            timestamp_min   = np.int64(array_cache['timestamp_min'])
            timestamp_max   = np.int64(array_cache['timestamp_max'])

            # ── Nhánh 1: cache rỗng ───────────────────────────────────────────────
            if timestamp_min == _ZERO and timestamp_max == _ZERO:
                db_min, db_max = self._get_timestamp_range(array_path, timeframe_minutes)

                if db_min == _ZERO and db_max == _ZERO:
                    self.logger.info(f"Cache và database rỗng cho {cache_key}, fetch từ config")
                    cache[cache_key] = {'timestamp_min': 0, 'timestamp_max': 0}
                    self._dirty = True   # A3: defer ghi file tới flush()
                    return self._build_intervals((config_start_ts, end_ts))

                self.logger.info(f"Cache rỗng nhưng database có dữ liệu cho {cache_key}, cập nhật cache")
                # int() bắt buộc tại đây vì orjson không serialize np.int64
                cache[cache_key] = {'timestamp_min': int(db_min), 'timestamp_max': int(db_max)}
                self._dirty = True   # A3: defer ghi file tới flush()
                timestamp_min = db_min
                timestamp_max = db_max

            # ── Nhánh 2: cache có dữ liệu — validate ─────────────────────────────
            else:
                # 1 open + 1 query thay cho 3+3 — xem _probe_cache_state (rò libtiledb)
                min_exists, max_exists, timestamps = self._probe_cache_state(
                    array_path, timeframe_minutes, timestamp_min, timestamp_max
                )

                if min_exists and max_exists:
                    self.logger.debug(f"Cache hợp lệ cho {cache_key}")
                    if len(timestamps) > 0:
                        real_max = np.max(timestamps)
                        self.logger.info(f"Tìm thấy timestamp_max lớn hơn: {real_max}, cập nhật cache")
                        cache[cache_key]['timestamp_max'] = int(real_max)
                        self._dirty = True   # A3: defer ghi file tới flush()
                        timestamp_max = real_max
                else:
                    self.logger.info(f"Cache không hợp lệ cho {cache_key}, truy vấn database")
                    db_min, db_max = self._get_timestamp_range(array_path, timeframe_minutes)

                    if db_min == _ZERO and db_max == _ZERO:
                        self.logger.info(f"Database rỗng cho {cache_key}")
                        cache[cache_key] = {'timestamp_min': 0, 'timestamp_max': 0}
                        self._dirty = True   # A3: defer ghi file tới flush()
                        return self._build_intervals((config_start_ts, end_ts))

                    self.logger.info(f"Cập nhật cache từ database cho {cache_key}")
                    cache[cache_key] = {'timestamp_min': int(db_min), 'timestamp_max': int(db_max)}
                    self._dirty = True   # A3: defer ghi file tới flush()
                    timestamp_min = db_min
                    timestamp_max = db_max

            # ── Build intervals — np.ndarray (N,2) int64 ─────────────────────────
            if config_start_ts is not None and config_start_ts < timestamp_min:
                with tiledb.open(array_path, mode='r') as array:
                    timestamps = array.query(
                        dims=['timeframe_minutes', 'timestamp'], attrs=[]
                    )[timeframe_minutes, config_start_ts:timestamp_min]['timestamp']

                    if len(timestamps) == 0:
                        result = self._build_intervals(
                            (config_start_ts, int(timestamp_min) - 1),
                            (int(timestamp_max) + 1, end_ts)
                        )
                    else:
                        actual_min = np.min(timestamps)
                        if actual_min < timestamp_min:
                            cache[cache_key]['timestamp_min'] = int(actual_min)
                            self._dirty = True   # A3: defer ghi file tới flush()
                            timestamp_min = actual_min
                        result = self._build_intervals((int(timestamp_max) + 1, end_ts))
            else:
                # timestamp_min <= config_start_ts: fetch từ sau max trở đi
                result = self._build_intervals((int(timestamp_max) + 1, end_ts))

            if len(result) == 0:
                self.logger.info(f"Không có khoảng thời gian cần fetch cho {cache_key}")
                return _EMPTY_INTERVALS

            self.logger.info(f"Fetch intervals cho {cache_key}: {result}")
            return result

    def get_fetch_timestamps_by_scan(
        self, array_path: str, timeframe_name: str,
        end_ts: int, start_ts: int
    ) -> np.ndarray:
        """
        Delegate thẳng sang TimestampScanner.
        start_ts + end_ts đã tính chuẩn từ api_fetch — KHÔNG tự tính lại.
        Returns: np.ndarray (N,2) int64
        """
        return self.timestamp_scanner.get_missing_timestamps_by_scan(
            array_path, timeframe_name, end_ts, start_ts
        )

    # ── Cache update helpers ────────────────────────────────────────────────────

    def update_cache(self, array_path: str, timeframe_name: str, timestamps: np.ndarray) -> None:
        """
        Cập nhật cache sau khi fetch + ghi dữ liệu.
        timestamps: np.ndarray int64 — không copy nếu đã đúng dtype.
        int() chỉ dùng khi gán vào dict để orjson serialize — bắt buộc.
        _cache_lock bảo vệ toàn bộ read-modify-write — tránh race condition.
        """
        if len(timestamps) == 0:
            self.logger.debug("Không có timestamps mới để cập nhật cache")
            return

        array_name        = os.path.basename(array_path)
        timeframe_minutes = self.calc_tf._get_timeframe_minutes(timeframe_name)
        cache_key         = f"{array_name}_tf{timeframe_minutes}"

        # Không copy nếu đã là np.ndarray int64
        ts_arr = timestamps if (isinstance(timestamps, np.ndarray) and timestamps.dtype == np.int64) \
                 else np.asarray(timestamps, dtype=np.int64)
        new_min = np.min(ts_arr)
        new_max = np.max(ts_arr)

        with self._cache_lock:
            cache       = self._cache   # A3: cache RAM
            array_cache = cache.get(cache_key, {'timestamp_min': 0, 'timestamp_max': 0})

            cur_min = array_cache['timestamp_min']
            cur_max = array_cache['timestamp_max']
            if cur_min != 0:
                new_min = np.minimum(np.int64(cur_min), new_min)
            if cur_max != 0:
                new_max = np.maximum(np.int64(cur_max), new_max)

            # int() bắt buộc — orjson không serialize np.int64
            cache[cache_key] = {'timestamp_min': int(new_min), 'timestamp_max': int(new_max)}
            self._dirty = True   # A3: defer ghi file tới flush()

        self.logger.info(f"Đã cập nhật cache cho {cache_key}: min={new_min}, max={new_max}")

    def update_cache_after_write(self, array_path: str, timeframe_name: str) -> None:
        """
        Cập nhật cache sau khi ghi xong (scan mode).
        Query DB trước — không load cache file nếu DB rỗng.
        _cache_lock bảo vệ read-modify-write — tránh race condition.
        """
        array_name        = os.path.basename(array_path)
        timeframe_minutes = self.calc_tf._get_timeframe_minutes(timeframe_name)

        if timeframe_minutes == 0:
            return

        cache_key      = f"{array_name}_tf{timeframe_minutes}"
        db_min, db_max = self._get_timestamp_range(array_path, timeframe_minutes)

        if db_min == _ZERO and db_max == _ZERO:
            self.logger.debug(f"Database rỗng cho {cache_key}, không cập nhật cache")
            return

        with self._cache_lock:
            cache = self._cache   # A3: cache RAM
            # int() bắt buộc — orjson không serialize np.int64
            cache[cache_key] = {'timestamp_min': int(db_min), 'timestamp_max': int(db_max)}
            self._dirty = True   # A3: defer ghi file tới flush()

        self.logger.info(
            f"Đã cập nhật cache sau ghi xong cho {cache_key}: "
            f"min={db_min}, max={db_max}"
        )
