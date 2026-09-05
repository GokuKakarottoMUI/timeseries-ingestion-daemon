from __future__ import annotations
import tiledb
import numpy as np
import picologging as logging
import sys

from ingestion.config_fetch_data import TIMEFRAMES, CUSTOM_TIMEFRAMES
from ingestion.calculate_tf_and_custom_tf import Calculate_Tf_And_CustomTF

logger = logging.getLogger("timestamp_scanner")
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

# ── Sentinels module level ─────────────────────────────────────────────────────
_EMPTY_INTERVALS: np.ndarray = np.empty((0, 2), dtype=np.int64)
_ZERO = np.int64(0)
_MS_PER_MIN = np.int64(60_000)


class TimestampScanner:
    """
    Quét TileDB dimension để tìm timestamps còn thiếu.

    Nguyên tắc:
    - Chỉ query DIMENSIONS, KHÔNG query attributes
    - end_ts truyền vào từ caller — KHÔNG gọi _get_current_closed_candle_time
      (tránh off-by-one khi caller đã tính sẵn 1 lần)
    - Output: np.ndarray (N,2) int64 — đồng nhất với CacheManager
    """

    def __init__(self):
        """
        Args:
            calc_tf: instance Calculate_Tf_And_CustomTF —
                     dùng _get_timeframe_minutes +
                     _generate_expected_slot_starts
        """
        self.calc_tf = Calculate_Tf_And_CustomTF()
        self.logger  = logger

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_missing_timestamps_by_scan(
        self,
        array_path:     str,
        timeframe_name: str,
        end_ts:         int,
        start_ts:       int,
    ) -> np.ndarray:
        """
        Hàm chính — quét dimension tìm timestamps thiếu.

        Args:
            array_path:     đường dẫn TileDB array
            timeframe_name: tên timeframe
            end_ts:         timestamp kết thúc (ms) — cây nến đóng cuối,
                            đã tính 1 lần từ caller, KHÔNG tính lại ở đây
            start_ts:       timestamp bắt đầu (ms) — đã tính chuẩn từ api_fetch,
                            KHÔNG fallback, KHÔNG tự tính lại

        Returns:
            np.ndarray (N,2) int64 — intervals còn thiếu [(start,end), ...]
        """
        if tiledb.object_type(array_path) != "array":
            self.logger.warning(f"Array {array_path} không tồn tại")
            return _EMPTY_INTERVALS

        timeframe_minutes = self.calc_tf._get_timeframe_minutes(timeframe_name)
        if timeframe_minutes == 0:
            self.logger.error(f"Không tìm thấy minutes cho timeframe {timeframe_name}")
            return _EMPTY_INTERVALS

        config_start_ts = np.int64(start_ts)

        # Phân nhánh regular vs custom timeframe
        is_custom = (
            CUSTOM_TIMEFRAMES.get("enable", False) and
            timeframe_name in CUSTOM_TIMEFRAMES.get("custom_intervals", {})
        )

        if is_custom:
            return self._scan_custom_timeframe(
                array_path, timeframe_name, timeframe_minutes,
                config_start_ts, np.int64(end_ts)
            )
        else:
            return self._scan_regular_timeframe(
                array_path, timeframe_minutes,
                config_start_ts, np.int64(end_ts)
            )

    # ── Regular timeframe ──────────────────────────────────────────────────────

    def _scan_regular_timeframe(
        self,
        array_path:        str,
        timeframe_minutes: int,
        start_ts:          np.int64,
        end_ts:            np.int64,
    ) -> np.ndarray:
        """
        Quét regular timeframe:
        1. Query toàn bộ timestamps từ TileDB dimension (attrs=[])
        2. Generate expected timestamps bằng np.arange — vectorized
        3. So sánh bằng np.isin — C-level
        4. Gộp gaps thành intervals

        Chỉ kiểm tra nến đã đóng cửa hoàn toàn (timestamp <= end_ts).
        """
        interval_ms = np.int64(timeframe_minutes) * _MS_PER_MIN

        # ── 1. Query dimension only — không pull attributes ───────────────────
        with tiledb.open(array_path, mode='r') as array:
            existing = array.query(
                dims=['timeframe_minutes', 'timestamp'], attrs=[]
            )[timeframe_minutes, :]['timestamp']   # np.ndarray int64

        # ── 2. Generate expected timestamps — vectorized, không Python loop ───
        # Align start_ts về timeframe boundary — binary search trả timestamp lẻ,
        # TileDB lưu timestamps đã align (exchange trả vậy) → phải match
        aligned_start = (start_ts // interval_ms) * interval_ms
        expected = np.arange(aligned_start, end_ts + interval_ms, interval_ms, dtype=np.int64)
        # Chỉ giữ timestamps <= end_ts (nến đã đóng)
        expected = expected[expected <= end_ts]

        if len(expected) == 0:
            return _EMPTY_INTERVALS

        if len(existing) == 0:
            # Toàn bộ expected đều thiếu
            return self._merge_missing_intervals(expected, interval_ms)

        # ── 3. Tìm missing — np.isin C-level, không Python loop ──────────────
        # assume_unique=True vì TileDB allows_duplicates=False → tăng tốc
        missing = expected[~np.isin(expected, existing, assume_unique=True)]

        if len(missing) == 0:
            self.logger.debug(f"Không có timestamp nào thiếu cho tf={timeframe_minutes}")
            return _EMPTY_INTERVALS

        self.logger.info(f"Tìm thấy {len(missing)} timestamps thiếu cho tf={timeframe_minutes}")
        return self._merge_missing_intervals(missing, interval_ms)

    # ── Custom timeframe ───────────────────────────────────────────────────────

    def _scan_custom_timeframe(
        self,
        array_path:        str,
        timeframe_name:    str,
        timeframe_minutes: int,
        start_ts:          np.int64,
        end_ts:            np.int64,
    ) -> np.ndarray:
        """
        Quét custom timeframe:
        1. Query timestamps hiện có từ TileDB
        2. Generate expected slot_starts từ _generate_expected_slot_starts
           (logic đúng đã có sẵn, tránh duplicate)
        3. So sánh → tìm missing → merge thành intervals
        """
        interval_ms = np.int64(timeframe_minutes) * _MS_PER_MIN

        # ── 1. Query dimension only ───────────────────────────────────────────
        with tiledb.open(array_path, mode='r') as array:
            existing = array.query(
                dims=['timeframe_minutes', 'timestamp'], attrs=[]
            )[timeframe_minutes, :]['timestamp']   # np.ndarray int64

        # ── 2. Generate expected — dùng lại logic từ Calculate_Tf_And_CustomTF
        #       filter_intervals=None → lấy toàn bộ slots trong [start_ts, end_ts]
        expected = self.calc_tf._generate_expected_slot_starts(
            target_timeframe = timeframe_name,
            start_ts         = int(start_ts),
            end_ts           = int(end_ts),
            filter_intervals = None,
        )   # np.ndarray (N,) int64

        if len(expected) == 0:
            return _EMPTY_INTERVALS

        if len(existing) == 0:
            return self._merge_missing_intervals(expected, interval_ms)

        # ── 3. Missing — np.isin C-level ─────────────────────────────────────
        missing = expected[~np.isin(expected, existing, assume_unique=True)]

        if len(missing) == 0:
            self.logger.debug(f"Không có timestamp nào thiếu cho {timeframe_name}")
            return _EMPTY_INTERVALS

        self.logger.info(f"Tìm thấy {len(missing)} timestamps thiếu cho {timeframe_name}")
        return self._merge_missing_intervals(missing, interval_ms)

    # ── Interval helpers ───────────────────────────────────────────────────────

    def _merge_missing_intervals(
        self,
        missing_ts:  np.ndarray,
        interval_ms: np.int64,
    ) -> np.ndarray:
        """
        Gộp timestamps thiếu liên tiếp thành intervals (start, end).

        Thuật toán vectorized:
        - timestamps liên tiếp nếu diff == interval_ms
        - tìm chỗ break (diff != interval_ms) → boundary của intervals
        - stack starts + ends thành (N,2) int64

        Args:
            missing_ts:  np.ndarray (N,) int64 — sorted tăng dần
            interval_ms: np.int64 — bước nhảy ms của timeframe

        Returns:
            np.ndarray (K,2) int64 — [(start1,end1), (start2,end2), ...]
        """
        if len(missing_ts) == 0:
            return _EMPTY_INTERVALS

        if len(missing_ts) == 1:
            # 1 timestamp duy nhất → 1 interval [ts, ts]
            return np.array([[missing_ts[0], missing_ts[0]]], dtype=np.int64)

        # ── Vectorized boundary detection ─────────────────────────────────────
        # diff giữa các timestamps liên tiếp
        diffs = np.diff(missing_ts)   # (N-1,) — view, không copy data

        # Vị trí break: diff > interval_ms (gap thật sự)
        breaks = np.flatnonzero(diffs > interval_ms)   # C-level

        # starts: index 0 + sau mỗi break
        start_indices = np.empty(len(breaks) + 1, dtype=np.intp)
        start_indices[0]  = 0
        start_indices[1:] = breaks + 1

        # ends: mỗi break + index cuối
        end_indices = np.empty(len(breaks) + 1, dtype=np.intp)
        end_indices[:-1] = breaks
        end_indices[-1]  = len(missing_ts) - 1

        # Build (K,2) — pre-allocate, gán thẳng
        k = len(start_indices)
        result = np.empty((k, 2), dtype=np.int64)
        result[:, 0] = missing_ts[start_indices]   # start của mỗi interval
        result[:, 1] = missing_ts[end_indices]      # end của mỗi interval

        return result