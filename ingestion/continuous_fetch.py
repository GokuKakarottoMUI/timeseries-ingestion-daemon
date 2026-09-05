import asyncio
import ctypes
import gc
import picologging as logging
import time
import os
import sys
import signal
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ingestion.config_fetch_data import FETCH_MODE_CONFIG
from ingestion.api_fetch import DataFetcher, _setup_nonblocking_logging

# Khởi tạo logger
logger = logging.getLogger('continuous_fetch')
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


# ══════════════════════════════════════════════════════════════════════════════
# Chống RSS ratchet — tinh chỉnh glibc + trả bộ nhớ về OS mỗi chu kỳ
# ══════════════════════════════════════════════════════════════════════════════
#
# Process treo dài ngày phình RSS KHÔNG phải vì leak Python (mọi dict state đều
# bounded theo symbol × TF) mà vì 2 cơ chế của glibc:
#   1. Dynamic mmap threshold — numpy/TileDB free block vài MB (mmap) → glibc TỰ
#      NÂNG mmap_threshold (tới 32MB) → cấp phát lớn sau đó lấy từ heap sbrk và
#      KHÔNG BAO GIỜ trả về OS ⇒ RSS chỉ đi lên.
#   2. Arena nở theo thread — 16 core cho tới 8×16 = 128 arena, mỗi arena giữ tới
#      64MB đã free() mà không trả OS.
# Khóa 2 ngưỡng + chặn số arena + malloc_trim(0) cuối mỗi chu kỳ (lúc idle) là
# fix ở đúng tầng allocator: KHÔNG đụng logic, KHÔNG copy thêm mảng nào.

# Hằng số mallopt từ glibc malloc.h
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3
_M_ARENA_MAX      = -8

_MALLOC_THRESHOLD_BYTES = 128 * 1024   # 128KB — sàn mặc định glibc, khóa cứng ở đây
_MALLOC_ARENA_MAX       = 4            # đủ cho 2 executor + TileDB, thay vì 128

_PAGE_SIZE = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096


def _load_libc():
    """Nạp libc 1 LẦN ở module level — non-glibc trả None, mọi thứ vẫn chạy."""
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


_LIBC = _load_libc()


def _tune_malloc() -> None:
    """
    Khóa hành vi allocator TRƯỚC khi sinh bất kỳ thread nào (executor/TileDB):
      - M_MMAP_THRESHOLD cố định → chặn cơ chế leo thang ngưỡng (nguyên nhân #1)
      - M_TRIM_THRESHOLD thấp    → heap trim sớm, không giữ đuôi lớn
      - M_ARENA_MAX giới hạn     → chặn nở arena theo thread (nguyên nhân #2)
    """
    if _LIBC is None or not hasattr(_LIBC, "mallopt"):
        logger.warning("Không có glibc mallopt — bỏ qua tinh chỉnh allocator")
        return
    try:
        _LIBC.mallopt(_M_MMAP_THRESHOLD, _MALLOC_THRESHOLD_BYTES)
        _LIBC.mallopt(_M_TRIM_THRESHOLD, _MALLOC_THRESHOLD_BYTES)
        _LIBC.mallopt(_M_ARENA_MAX, _MALLOC_ARENA_MAX)
        logger.info(
            f"Đã khóa allocator: mmap_threshold=trim_threshold="
            f"{_MALLOC_THRESHOLD_BYTES // 1024}KB, arena_max={_MALLOC_ARENA_MAX}"
        )
    except Exception as e:
        logger.warning(f"mallopt lỗi (bỏ qua): {e}")


def _malloc_trim() -> None:
    """Trả phần heap đã free về OS — chỗ DUY NHẤT làm RSS thật sự tụt xuống."""
    if _LIBC is None or not hasattr(_LIBC, "malloc_trim"):
        return
    try:
        _LIBC.malloc_trim(0)
    except Exception as e:
        logger.warning(f"malloc_trim lỗi (bỏ qua): {e}")


def _rss_bytes() -> int:
    """RSS hiện tại — 1 read + split trên /proc/self/statm (lõi C, không loop)."""
    try:
        with open('/proc/self/statm', 'rb') as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except Exception:
        return 0

class ContinuousDataFetcher:
    def __init__(self):
        self.continuous_mode = FETCH_MODE_CONFIG["continuous"]
        self.fetch_cycle_seconds = FETCH_MODE_CONFIG["fetch_interval"]
        self.sleep_time = FETCH_MODE_CONFIG["sleep_interval"]
        self.continuous_sleep_time = FETCH_MODE_CONFIG["continuous_sleep_interval"]

        self.rss_limit_bytes = int(FETCH_MODE_CONFIG.get("rss_restart_mb", 0)) * 1024 * 1024

        self.logger = logging.getLogger('continuous_fetch')
        self.running = True
        self.is_paused = False
        self.log_listener = None      # main() gán vào — cần stop trước khi re-exec
        self.data_fetcher = DataFetcher()

        # 1 EVENT LOOP BỀN cho cả run — cold-start (executor/session/DNS) dựng 1 LẦN,
        # mỗi chu kỳ chỉ scan+fetch+flush. KHÔNG build lại mỗi chu kỳ (đúng triết lý "1 loop chung").
        self._loop: asyncio.AbstractEventLoop | None = None

        # Mốc RSS để ĐO độ phình theo thời gian (đặt ở cold-start, xem _reclaim_memory)
        self._rss_base: int = 0
        self._rss_prev: int = 0

        self.data_fetcher.db_manager.create_database_structure()

    def setup_signal_handlers(self):
        """Thiết lập các signal handlers."""
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        # Xử lý signal khác nhau tùy theo hệ điều hành
        if os.name == 'nt':  # Windows
            signal.signal(signal.SIGBREAK, self._handle_pause)
        else:  # Linux/Unix
            signal.signal(signal.SIGUSR1, self._handle_pause)
            signal.signal(signal.SIGUSR2, self._handle_resume)

    def _handle_pause(self, signum, frame):
        """Xử lý signal tạm dừng."""
        self.is_paused = True
        self.logger.info("Đã nhận tín hiệu tạm dừng")
        while self.is_paused and self.running:
            time.sleep(1)

    def _handle_resume(self, signum, frame):
        """Xử lý signal tiếp tục."""
        self.is_paused = False
        self.logger.info("Đã nhận tín hiệu tiếp tục")

    def _handle_exit(self, signum, frame):
        """Xử lý signal thoát."""
        self.logger.info("Đã nhận tín hiệu dừng, đang thoát...")
        self.running = False
        sys.exit(0)

    def _reclaim_memory(self) -> None:
        """
        Cuối mỗi chu kỳ, trong CỬA SỔ IDLE (không nằm trên đường nóng fetch):
        gc.collect() gom rác vòng tham chiếu → malloc_trim(0) TRẢ heap về OS.
        Rồi log RSS + delta để ĐO được độ phình thay vì đoán.
        """
        gc.collect()
        _malloc_trim()

        rss = _rss_bytes()
        if rss == 0:
            return
        if self._rss_base == 0:
            self._rss_base = rss
        mb = 1024.0 * 1024.0
        d_cycle = (rss - self._rss_prev) / mb if self._rss_prev else 0.0
        d_total = (rss - self._rss_base) / mb
        self._rss_prev = rss
        self.logger.info(
            f"RSS {rss / mb:.1f}MB (Δchu kỳ {d_cycle:+.1f}MB, Δtừ cold-start {d_total:+.1f}MB)"
        )

    def _maybe_restart(self) -> None:
        """
        TRẦN CỨNG cho RSS. libtiledb 0.36.1 rò ~15KB mỗi lệnh đọc sparse — đo bằng
        mallinfo2: uordblks tăng tuyến tính suốt 6000 query, fordblks đứng yên ⇒ cấp
        phát KHÔNG free (không phải phân mảnh), `del ctx` không lấy lại được, và 0.36.1
        đã là bản mới nhất nên KHÔNG có bản vá. Trong tiến trình không cách nào thu hồi.

        Nên: vượt ngưỡng thì tự `os.execv` chính mình NGAY TẠI RANH GIỚI CHU KỲ — sau
        khi đã fetch/flush/consolidate/drain xong, không còn array mở, không còn flock.
        Cùng PID (GUI theo dõi không mất dấu), RAM về mốc nền, chạy tiếp. Tốn ~5s cold-start.
        rss_restart_mb = 0 → tắt.
        """
        if self.rss_limit_bytes <= 0:
            return
        rss = _rss_bytes()
        if rss <= 0 or rss < self.rss_limit_bytes:
            return

        mb = 1024.0 * 1024.0
        self.logger.warning(
            f"RSS {rss / mb:.0f}MB ≥ trần {self.rss_limit_bytes / mb:.0f}MB "
            f"(rò libtiledb, không thu hồi được) → tự khởi động lại tiến trình"
        )

        # Teardown sạch — y hệt nhánh finally của start(), để không bỏ dở việc gì.
        try:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.run_until_complete(self.data_fetcher._teardown_phase1())
        except Exception as e:
            self.logger.warning(f"Teardown trước re-exec lỗi (bỏ qua): {e}")
        try:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.close()
        except Exception:
            pass
        try:
            self.data_fetcher.cache_manager.flush()   # cache RAM → file, không mất mốc
        except Exception as e:
            self.logger.warning(f"Flush cache trước re-exec lỗi (bỏ qua): {e}")
        if self.log_listener is not None:
            try:
                self.log_listener.stop()             # xả nốt log đang nằm trong queue
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()

        os.execv(sys.executable, [sys.executable] + sys.argv)   # thay ảnh tiến trình

    def start(self):
        """Bắt đầu chu trình quét dữ liệu liên tục."""
        self.setup_signal_handlers()
        self.logger.info(f"Bắt đầu quét dữ liệu {'liên tục realtime' if self.continuous_mode else 'theo chu kỳ'}")

        # Cold-start 1 LẦN trên loop bền: precompute (no-op khi fetch_all off) + prime executor + prewarm DNS.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self.data_fetcher.precompute_earliest()                          # 1 lần (chỉ chạy khi fetch_all)
            self._loop.run_until_complete(self.data_fetcher._setup_phase1())  # prime 16 worker + prewarm DNS 1 LẦN

            # Toàn bộ object khởi tạo (config, TileDB ctx, session, executor) là BẤT BIẾN
            # suốt run → freeze sang permanent generation: GC không quét lại chúng ở mọi
            # gc.collect() sau này (rẻ hơn + ít chạm trang hơn).
            gc.collect()
            gc.freeze()
            self._rss_base = self._rss_prev = _rss_bytes()
            self.logger.info(
                f"Đã cold-start 1 lần — vào vòng quét realtime (tái dùng executor/session/DNS), "
                f"RSS nền {self._rss_base / (1024.0 * 1024.0):.1f}MB"
            )

            while self.running:
                if self.is_paused:
                    time.sleep(1)
                    continue
                start_time = time.time()
                self.logger.info(f"Bắt đầu chu kỳ quét dữ liệu mới tại {datetime.now(timezone.utc).astimezone(ZoneInfo('Asia/Ho_Chi_Minh'))}")
                self.fetch_all_new_data()
                elapsed_time = time.time() - start_time
                self._reclaim_memory()   # trả RAM về OS TRƯỚC khi nghỉ (cửa sổ idle)
                self._maybe_restart()    # trần cứng — re-exec nếu vượt (không quay lại)
                if self.continuous_mode:
                    self.logger.debug(f"Hoàn thành chu kỳ quét trong {elapsed_time:.2f} giây, nghỉ {self.continuous_sleep_time} giây")
                    time.sleep(self.continuous_sleep_time)
                else:
                    self.logger.info(f"Hoàn thành chu kỳ quét trong {elapsed_time:.2f} giây")
                    self.logger.info(f"Đang nghỉ {self.sleep_time} giây trước khi quét tiếp theo")
                    time.sleep(self.sleep_time)
        except Exception as e:
            self.logger.critical(f"Lỗi nghiêm trọng trong quá trình quét: {str(e)}")
            raise
        finally:
            # Teardown 1 LẦN cuối run: đóng session + shutdown executor (best-effort, không che lỗi gốc).
            try:
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.run_until_complete(self.data_fetcher._teardown_phase1())
            except Exception as e:
                self.logger.warning(f"Teardown phase1 lỗi (bỏ qua): {e}")
            finally:
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.close()
            self.logger.info("Đã kết thúc quá trình quét dữ liệu")

    def fetch_all_new_data(self):
        """
        1 chu kỳ NHẸ trên loop bền: scan dimension → fetch nến mới → flush.
        KHÔNG dựng lại executor/session/DNS (đã setup 1 lần). Aggregate tự skip khi
        custom off / không slot mới; consolidate CHỈ chạy khi chu kỳ có ghi nến mới.
        """
        self.logger.info("Bắt đầu quét dữ liệu mới cho tất cả cấu hình active...")
        self._loop.run_until_complete(self.data_fetcher._fetch_pass())   # Phase 1 — tái dùng loop/executor/session
        self.data_fetcher.aggregate_custom_timeframes()                  # Phase 2 — tự skip khi không có gì để dựng
        if self.data_fetcher.consume_consolidate_flag():                 # chỉ gộp fragment khi vừa ghi nến mới
            self.data_fetcher.consolidate_all_arrays()
        else:
            self.logger.info("Không có nến mới ghi → bỏ qua consolidate")
        self._drain_tensor_buffer()                                      # không để nến sót đi qua ranh giới chu kỳ
        self.logger.info("Hoàn thành quét dữ liệu mới cho tất cả cấu hình")

    def _drain_tensor_buffer(self) -> None:
        """
        Rào chắn cuối chu kỳ: TensorBuffer.add_candles NỐI (np.concatenate) vào key cũ,
        còn _fetch_pass NUỐT exception theo symbol — nên nếu 1 chu kỳ chết giữa
        add_candles và _flush_keys, nến sót ở lại buffer và chu kỳ sau nối tiếp vào,
        tích tụ không có trần. Flush nốt phần sót (đúng cái _flush_to_db vẫn làm ở
        đường aggregate) → buffer luôn RỖNG khi bước sang chu kỳ mới.
        """
        leftover = self.data_fetcher.tensor_buffer.get_all_keys()
        if not leftover:
            return
        self.logger.warning(f"Còn {len(leftover)} key sót trong buffer → flush nốt, không để tích tụ")
        try:
            self.data_fetcher._flush_to_db()
        except Exception as e:
            self.logger.error(f"Flush phần sót lỗi: {e}")

def main():
    """Hàm chính để chạy quá trình lấy dữ liệu liên tục."""
    _tune_malloc()                                # TRƯỚC mọi thread (executor/TileDB) — xem ghi chú allocator
    log_listener = _setup_nonblocking_logging()   # diệt deadlock picologging đa luồng (như api_fetch.main)
    logger.info("Bắt đầu quá trình quét dữ liệu liên tục")
    try:
        fetcher = ContinuousDataFetcher()
        fetcher.log_listener = log_listener   # để _maybe_restart xả log trước khi execv
        fetcher.start()
    finally:
        log_listener.stop()   # flush + dừng listener thread
        logger.info("Kết thúc quá trình quét dữ liệu liên tục")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng khi chạy chương trình: {str(e)}")
        raise e
