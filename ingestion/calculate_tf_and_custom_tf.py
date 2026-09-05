from __future__ import annotations
from datetime import datetime, timezone, timedelta
import numpy as np
import picologging as logging

from ingestion.config_fetch_data import CUSTOM_TIMEFRAMES, DAILY_OPEN, HISTORICAL_DATA_CONFIG, TIMEFRAMES

# ── Constants module level ─────────────────────────────────────────────────────
_UTC             = timezone.utc
_MS_PER_DAY      = np.int64(86_400_000)
_MS_PER_MIN      = np.int64(60_000)
_MS_PER_SEC      = np.int64(1_000)
_SEC_PER_DAY     = np.int64(86_400)

_DAILY_OPEN_H, _DAILY_OPEN_M = map(int, DAILY_OPEN.split(':'))
_DAILY_OPEN_SECONDS = np.int64(_DAILY_OPEN_H * 3600 + _DAILY_OPEN_M * 60)

# Empty sentinels đồng nhất kiểu
_EMPTY_SLOTS:   np.ndarray = np.empty((0,), dtype=np.int64)
_EMPTY_CANDLES: np.ndarray = np.empty((0, 6), dtype=np.float64)

logger = logging.getLogger('Calculate_Tf_And_CustomTF')


# ── Leap year — thuần math ─────────────────────────────────────────────────────
def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def _days_in_year(year: int) -> int:
    return 366 if _is_leap(year) else 365


class Calculate_Tf_And_CustomTF:
    def __init__(self):
        pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_timeframe_minutes(self, timeframe_name: str) -> int:
        """Lấy số minutes từ timeframe name."""
        if timeframe_name in TIMEFRAMES:
            return TIMEFRAMES[timeframe_name].get("minutes", 0)
        if CUSTOM_TIMEFRAMES.get("enable", False):
            custom_intervals = CUSTOM_TIMEFRAMES.get("custom_intervals", {})
            if timeframe_name in custom_intervals:
                return custom_intervals[timeframe_name].get("minutes", 0)
        logger.warning(f"Không tìm thấy minutes cho timeframe {timeframe_name}")
        return 0

    def _calculate_start_ts(self) -> int:
        """Tính timestamp bắt đầu từ config."""
        fetch_all = HISTORICAL_DATA_CONFIG["fetch_all"]["active"]["value"]
        if fetch_all:
            return 0
        sd = HISTORICAL_DATA_CONFIG["start_date"]
        config_start = datetime(
            sd["year"]["value"],
            sd["month"]["value"],
            sd["day"]["value"],
            tzinfo=_UTC
        )
        return int(config_start.timestamp() * 1000)

    def _get_current_closed_candle_time(self, timeframe: str) -> int:
        """Lấy timestamp cây nến đóng cửa gần nhất."""
        now = datetime.now(_UTC).replace(second=0, microsecond=0)

        # Custom TF (aggregate): chỉ trả biên trên = now. Việc chốt cây đóng cuối
        # (kể cả remainder cuối ngày, vd 10') giao cho _generate_expected_slot_starts
        # — nó tự clamp upper=min(end_ts, now) & chỉ nhận slot có slot_end <= upper.
        # KHÔNG tự tính start+interval ở đây để khỏi sai biên remainder, và tránh
        # lệch lưới epoch vs daily-open với TF không chia hết ngày (11m, 22m...).
        if (timeframe not in TIMEFRAMES
                and CUSTOM_TIMEFRAMES.get("enable", False)
                and timeframe in CUSTOM_TIMEFRAMES.get("custom_intervals", {})):
            return int(now.timestamp() * 1000)

        timeframe_config = TIMEFRAMES.get(timeframe)
        if not timeframe_config and CUSTOM_TIMEFRAMES.get("enable", False):
            timeframe_config = CUSTOM_TIMEFRAMES["custom_intervals"].get(timeframe)

        if not timeframe_config:
            raise ValueError(f"Timeframe {timeframe} không được định nghĩa trong config")

        if timeframe_config.get('minutes', 0) > 0:
            interval_minutes = timeframe_config['minutes']
        elif timeframe_config.get('hours', 0) > 0:
            interval_minutes = timeframe_config['hours'] * 60
        elif timeframe_config.get('days', 0) > 0:
            closed = (now - timedelta(days=timeframe_config['days'])).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return int(closed.timestamp() * 1000)
        else:
            raise ValueError(f"Timeframe {timeframe} không có cấu hình thời gian hợp lệ")

        interval_seconds      = interval_minutes * 60
        now_seconds           = int(now.timestamp())
        closed_candle_seconds = (now_seconds // interval_seconds) * interval_seconds - interval_seconds
        if closed_candle_seconds < 0:
            closed_candle_seconds = 0

        candle_time = datetime.fromtimestamp(closed_candle_seconds, tz=_UTC)
        logger.info(f"Closed candle time for {timeframe}: {candle_time}")
        return int(candle_time.timestamp() * 1000)

    # ── Time-slot generation — dùng chung cho aggregate + scanner ─────────────

    @staticmethod
    def _in_intervals(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """
        D2: mask membership 'value ∈ MỘT interval bất kỳ [start,end]' bằng searchsorted
        thay broadcast (N×K). C-level, không phình RAM khi nhiều interval.
        Đúng cả khi interval chồng lấn: sort theo start → cummax(end) → value hợp lệ ⟺
        tồn tại interval có start<=value và max-end(các interval start<=value) >= value.
        Bao trùm cả 2 biên (>=start, <=end) — khớp đúng broadcast cũ.
        """
        if len(values) == 0 or len(starts) == 0:
            return np.zeros(len(values), dtype=bool)
        order = np.argsort(starts, kind='stable')
        s = starts[order]
        cmax = np.maximum.accumulate(ends[order])          # max end trong các interval start<=
        idx  = np.searchsorted(s, values, side='right') - 1  # interval có start lớn nhất <= value
        out  = idx >= 0
        ii   = idx[out]
        out[out] = values[out] <= cmax[ii]
        return out

    def _generate_expected_slot_starts(
        self,
        target_timeframe: str,
        start_ts: int,
        end_ts: int,
        filter_intervals: np.ndarray | None = None,   # (K,2) int64 hoặc None
    ) -> np.ndarray:
        """
        Generate tất cả expected slot_starts cho custom timeframe
        trong khoảng [start_ts, end_ts].

        Dùng chung cho:
        - _aggregate_candles_batch_numpy  (aggregate OHLCV)
        - TimestampScanner._scan_custom_timeframe (tìm timestamps thiếu)

        Args:
            target_timeframe: tên custom timeframe
            start_ts:         timestamp bắt đầu (ms)
            end_ts:           timestamp kết thúc — cây nến đóng cuối (ms)
            filter_intervals: np.ndarray (K,2) int64 — chỉ giữ slots nằm trong
                              các intervals này. None = giữ tất cả.

        Returns:
            np.ndarray (N,) int64 — slot_starts đã filter, sorted tăng dần.
            _EMPTY_SLOTS nếu không có slot nào hợp lệ.
        """
        target_config = CUSTOM_TIMEFRAMES.get("custom_intervals", {}).get(target_timeframe)
        if not target_config or not target_config.get("active", False):
            return _EMPTY_SLOTS

        target_minutes = target_config.get("minutes", 0) or target_config.get("hours", 0) * 60
        if target_minutes <= 0:
            return _EMPTY_SLOTS

        interval_ms = np.int64(target_minutes) * _MS_PER_MIN
        now_ms      = np.int64(int(datetime.now(_UTC).timestamp() * 1000))
        upper_ms    = min(np.int64(end_ts), now_ms)   # không vượt quá nến đã đóng

        # ── filter_intervals (D2: dùng _in_intervals thay broadcast) ──────────
        fi_starts = fi_ends = None
        if filter_intervals is not None and len(filter_intervals) > 0:
            fi_starts = filter_intervals[:, 0]   # (K,) int64 view
            fi_ends   = filter_intervals[:, 1]   # (K,) int64 view

        end_ts_i64 = np.int64(end_ts)
        min_ts     = np.int64(start_ts)

        # ════════ DAY-CASE (target_minutes <= 1440) — VECTORIZED (D1) ═════════
        # Bỏ loop-theo-ngày (trước: hàng nghìn vòng). Sinh slot toàn dải bằng arange
        # 2D + lọc validity vector hóa — output GIỮ NGUYÊN (parity).
        if target_minutes <= 1440:
            # cs0 = mốc daily-open của period đầu (GIỮ NGUYÊN logic align cũ)
            min_ts_s   = min_ts // _MS_PER_SEC
            day_base_s = (min_ts_s // _SEC_PER_DAY) * _SEC_PER_DAY
            open_s     = day_base_s + _DAILY_OPEN_SECONDS
            if min_ts_s < open_s:
                open_s -= _SEC_PER_DAY
            cs0 = np.int64(open_s) * _MS_PER_SEC
            if cs0 > end_ts_i64:
                return _EMPTY_SLOTS

            fpp           = 1440 // target_minutes
            remainder_min = 1440 % target_minutes

            # day_starts: mọi period (ngày) có start <= end_ts (đúng điều kiện while cũ)
            n_days     = int((end_ts_i64 - cs0) // _MS_PER_DAY) + 1
            day_starts = cs0 + np.arange(n_days, dtype=np.int64) * _MS_PER_DAY   # (D,)

            # Regular slots: day_start + j*im, j=0..fpp-1 → hợp lệ ⟺ slot+im <= upper_ms
            # (tương đương 'slot <= min(period_end,upper)-im' của bản cũ; với regular,
            #  slot <= period_end-im luôn đúng nên ràng buộc rút về upper_ms).
            reg_off = np.arange(fpp, dtype=np.int64) * interval_ms               # (fpp,)
            reg     = (day_starts[:, None] + reg_off[None, :]).ravel()           # (D*fpp,)
            reg     = reg[reg + interval_ms <= upper_ms]

            if remainder_min > 0:
                # Remainder slot (lẻ cuối ngày) tại offset fpp*im → hợp lệ ⟺ ngày đóng đủ
                # (r_end = period_end <= upper_ms ⟺ day_start + 1ngày <= upper_ms)
                rem = day_starts + np.int64(fpp) * interval_ms                   # (D,)
                rem = rem[day_starts + _MS_PER_DAY <= upper_ms]
                slots = np.concatenate((reg, rem))
                slots.sort(kind='stable')   # reg + rem → về tăng dần (như thứ tự append cũ)
            else:
                slots = reg   # đã tăng dần sẵn (day_starts↑, reg_off↑)

            if fi_starts is not None and len(slots) > 0:
                slots = slots[self._in_intervals(slots, fi_starts, fi_ends)]
            return slots if len(slots) > 0 else _EMPTY_SLOTS

        # ════════ YEAR-CASE (target_minutes > 1440) — giữ loop NĂM (~15 vòng) ══
        start_year         = datetime.fromtimestamp(int(min_ts) / 1000, tz=_UTC).year
        start_period       = datetime(start_year, 1, 1, tzinfo=_UTC)
        current_start_ts   = np.int64(int(start_period.timestamp() * 1000))
        minutes_per_period = _days_in_year(start_year) * 1440
        full_per_period    = minutes_per_period // target_minutes
        remainder_min      = minutes_per_period % target_minutes

        all_slot_starts: list[np.ndarray] = []
        while current_start_ts <= end_ts_i64:
            period_start_ts = current_start_ts
            cur_year      = datetime.fromtimestamp(int(period_start_ts) / 1000, tz=_UTC).year
            next_yr       = datetime(cur_year + 1, 1, 1, tzinfo=_UTC)
            period_end_ts = np.int64(int(next_yr.timestamp() * 1000))

            slot_starts = np.arange(
                period_start_ts,
                period_start_ts + np.int64(full_per_period) * interval_ms,
                interval_ms, dtype=np.int64,
            )
            local_upper = min(period_end_ts, upper_ms)
            slot_starts = slot_starts[slot_starts <= (local_upper - interval_ms)]

            if len(slot_starts) > 0:
                if fi_starts is not None:
                    slot_starts = slot_starts[self._in_intervals(slot_starts, fi_starts, fi_ends)]
                if len(slot_starts) > 0:
                    all_slot_starts.append(slot_starts)

            if remainder_min > 0:
                r_start = period_start_ts + np.int64(full_per_period) * interval_ms
                r_end   = r_start + np.int64(remainder_min) * _MS_PER_MIN
                if r_end <= local_upper:
                    if fi_starts is None or bool(((r_start >= fi_starts) & (r_start <= fi_ends)).any()):
                        all_slot_starts.append(np.array([r_start], dtype=np.int64))

            current_start_ts   = period_end_ts
            ny                 = datetime.fromtimestamp(int(current_start_ts) / 1000, tz=_UTC).year
            minutes_per_period = _days_in_year(ny) * 1440
            full_per_period    = minutes_per_period // target_minutes
            remainder_min      = minutes_per_period % target_minutes

        if not all_slot_starts:
            return _EMPTY_SLOTS
        return np.concatenate(all_slot_starts)

    # ── Core aggregation ───────────────────────────────────────────────────────

    def _aggregate_candles_batch_numpy(
        self,
        source_array:          np.ndarray,
        source_timeframe:      str,
        target_timeframe:      str,
        aggregation_intervals: np.ndarray | None = None,   # (K,2) int64
    ) -> np.ndarray:
        """
        Aggregate OHLCV cho custom timeframe.

        Args:
            source_array:          (N, 6+) float64
            source_timeframe:      tên timeframe nguồn
            target_timeframe:      tên custom timeframe đích
            aggregation_intervals: np.ndarray (K,2) int64 — chỉ aggregate các
                                   khoảng này. None = aggregate toàn bộ.

        Returns:
            np.ndarray (M, 6) float64 — [ts, o, h, l, c, v]
            _EMPTY_CANDLES nếu không có dữ liệu.
        """
        if not CUSTOM_TIMEFRAMES.get("enable", False):
            return _EMPTY_CANDLES

        target_config = CUSTOM_TIMEFRAMES.get("custom_intervals", {}).get(target_timeframe)
        if not target_config or not target_config.get("active", False):
            return _EMPTY_CANDLES

        if len(source_array) == 0:
            return _EMPTY_CANDLES

        target_minutes = target_config.get("minutes", 0) or target_config.get("hours", 0) * 60
        if target_minutes <= 0:
            return _EMPTY_CANDLES

        source_minutes = self._get_timeframe_minutes(source_timeframe)
        if source_minutes == 0:
            return _EMPTY_CANDLES

        # View col 0-5 — không copy
        source_ohlcv     = source_array[:, :6]
        interval_ms      = np.int64(target_minutes)  * _MS_PER_MIN
        source_interval_ms = np.int64(source_minutes) * _MS_PER_MIN

        # ── Filter + dedup source theo aggregation_intervals (D2: searchsorted) ─
        if aggregation_intervals is not None and len(aggregation_intervals) > 0:
            buffer_ms = interval_ms * np.int64(2)
            ts_col    = source_ohlcv[:, 0].astype(np.int64)
            # nới biên ±buffer_ms rồi membership bằng _in_intervals (thay broadcast N×K)
            agg_s     = aggregation_intervals[:, 0] - buffer_ms
            agg_e     = aggregation_intervals[:, 1] + buffer_ms
            mask = self._in_intervals(ts_col, agg_s, agg_e)
            source_ohlcv = source_ohlcv[mask]
            if len(source_ohlcv) == 0:
                return _EMPTY_CANDLES
            # unique_idx từ np.unique → sorted by timestamp (không np.sort)
            _, unique_idx = np.unique(source_ohlcv[:, 0], return_index=True)
            source_ohlcv  = source_ohlcv[unique_idx]

        min_ts        = int(source_ohlcv[0, 0])
        max_ts        = int(source_ohlcv[-1, 0])
        last_close_ms = np.int64(max_ts) + source_interval_ms

        # ── Generate expected slot_starts — dùng method tách riêng ────────────
        slot_starts = self._generate_expected_slot_starts(
            target_timeframe   = target_timeframe,
            start_ts           = min_ts,
            end_ts             = int(last_close_ms),
            filter_intervals   = aggregation_intervals,
        )

        if len(slot_starts) == 0:
            return _EMPTY_CANDLES

        slot_ends = slot_starts + interval_ms

        # ── Filter slots: s_end <= last_close_ms — vectorized ─────────────────
        valid_slot_mask = slot_ends <= last_close_ms
        slot_starts = slot_starts[valid_slot_mask]
        slot_ends   = slot_ends[valid_slot_mask]

        if len(slot_starts) == 0:
            return _EMPTY_CANDLES

        ts_col = source_ohlcv[:, 0]   # view, sorted ascending

        # ── Assign mỗi source candle vào slot — np.searchsorted O(N log S) ───
        # side='right': ts == slot_start → vào đúng slot đó
        slot_ids      = np.searchsorted(slot_starts, ts_col, side='right') - 1
        slot_ids_safe = slot_ids.clip(0, len(slot_starts) - 1)

        # Filter candles nằm trong slot range
        in_range = (slot_ids >= 0) & (ts_col < slot_ends[slot_ids_safe])
        if not in_range.any():
            return _EMPTY_CANDLES

        valid_idx      = np.where(in_range)[0]
        slot_ids_valid = slot_ids[valid_idx]      # (M,) slot index per valid candle, sorted
        valid_src      = source_ohlcv[valid_idx]  # (M, 6) valid candles, ts-sorted

        # ── Group boundaries — np.unique O(M) ────────────────────────────────
        unique_slot_ids, group_starts, counts = np.unique(
            slot_ids_valid, return_index=True, return_counts=True
        )
        # group_starts[i] = index trong valid_src nơi group i bắt đầu

        # ── Aggregate — all C-level, O(M) ─────────────────────────────────────
        timestamps = slot_starts[unique_slot_ids]               # (S_ne,)
        opens      = valid_src[group_starts, 1]                 # (S_ne,) first candle open
        closes     = valid_src[group_starts + counts - 1, 4]   # (S_ne,) last candle close
        highs      = np.maximum.reduceat(valid_src[:, 2], group_starts)  # (S_ne,) C-level
        lows       = np.minimum.reduceat(valid_src[:, 3], group_starts)  # (S_ne,) C-level
        vols       = np.add.reduceat(valid_src[:, 5], group_starts)      # (S_ne,) C-level

        # ── Pre-allocate output (S_ne, 6) float64 — ghi thẳng từng col ───────
        S_ne         = len(group_starts)
        candle_array = np.empty((S_ne, 6), dtype=np.float64)
        candle_array[:, 0] = timestamps
        candle_array[:, 1] = opens
        candle_array[:, 2] = highs
        candle_array[:, 3] = lows
        candle_array[:, 4] = closes
        candle_array[:, 5] = vols

        return candle_array
