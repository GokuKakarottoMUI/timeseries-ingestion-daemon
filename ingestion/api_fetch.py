from __future__ import annotations
import os
import sys
import time
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import socket
import queue
import gc

import aiohttp
import numpy as np
import tiledb
import picologging as logging
from picologging.handlers import QueueHandler, QueueListener

from ingestion.config_fetch_data import (
    CUSTOM_TIMEFRAMES, HISTORICAL_DATA_CONFIG, SYMBOLS_CONFIG,
    TIMEFRAMES, EXCHANGE_CONFIGS, build_array_path,
)
from ingestion.database import DatabaseManager
from ingestion.cache_timestamp import CacheManager
from ingestion.timestamp_scanner import TimestampScanner
from ingestion.exchange_utils import (
    ExchangeURLBuilder, ExchangeFormatter, RateLimiter, _VARIANT_LOOKUP,
    _EXCHANGE_TS_FORMAT,
)

logger = logging.getLogger('api_fetch')
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

# ── Multi-fetch config đọc 1 lần ở module level ──────────────────────────────
_MF             = HISTORICAL_DATA_CONFIG["multi_fetch"]
_MF_ACTIVE      = _MF["active"]["value"]
_MF_UNLIMITED   = _MF["unlimited"]["value"]
_MF_MAX_WORKERS = _MF["max_workers"]["value"]

# fetch_all=True → binary search tìm mốc sớm nhất từ API
# fetch_all=False → start_date đã thiết lập → dùng _calculate_start_ts() trực tiếp
_FETCH_ALL_ACTIVE: bool = (
    HISTORICAL_DATA_CONFIG
    .get("fetch_all", {})
    .get("active", {})
    .get("value", False)
)

# Scan mode → Phase 1 lặp scan dimension + fetch tới khi không còn nến thiếu
_SCAN_MODE_ACTIVE: bool = (
    HISTORICAL_DATA_CONFIG
    .get("scan_missing_timestamps", {})
    .get("active", {})
    .get("value", False)
)

_EMPTY_CANDLES: np.ndarray = np.empty((0, 6), dtype=np.float64)

# ── Concurrency "làn" + rate + backoff per exchange — đọc 1 lần ──────────────
# Mỗi "làn" = 1 slot semaphore chạy đồng thời, tự tuân thủ rate_limit độc lập.
# Tốc độ tổng ≈ số_làn / rate_limit (req/s), tự ghìm về ngưỡng an toàn nhờ 429 backoff.
def _pick_rate(cfg: dict) -> float:
    """rate_limit per-lane (giây). fetch_all dùng rate_limit_fetch_all nếu có."""
    if _FETCH_ALL_ACTIVE:
        return cfg.get("rate_limit_fetch_all", cfg.get("rate_limit", 0.2))
    return cfg.get("rate_limit", 0.2)

_EXCHANGE_RATE: Dict[str, float] = {
    name: _pick_rate(cfg) for name, cfg in EXCHANGE_CONFIGS.items()
}
_EXCHANGE_MAX_CONCURRENCY: Dict[str, int] = {
    name: int(cfg.get("max_concurrency", 10)) for name, cfg in EXCHANGE_CONFIGS.items()
}
_EXCHANGE_BACKOFF_BASE: Dict[str, float] = {
    name: float(cfg.get("backoff_429_base", 2.0)) for name, cfg in EXCHANGE_CONFIGS.items()
}
_EXCHANGE_MAX_BACKOFF: Dict[str, float] = {
    name: float(cfg.get("max_backoff", 30.0)) for name, cfg in EXCHANGE_CONFIGS.items()
}
# Trần cứng số socket đồng thời khi multi_fetch.unlimited — tránh mở 7400 socket cùng lúc
_HARD_CONCURRENCY_CAP: int = 80

# P3.14 — RAM guard CANH ĐỘNG (theo % RAM máy thật, không hard-code số cứng)
_RAM_MIN_FREE_RATIO: float = 0.15   # giữ trống ≥ 15% RAM máy
_RAM_WAIT_MAX:       int   = 10     # chờ tối đa 10 × 0.5s = 5s rồi tiếp tục (không block vô hạn)


# ══════════════════════════════════════════════════════════════════════════════
# TensorBuffer — in-memory ndarray accumulator, thread-safe
# ══════════════════════════════════════════════════════════════════════════════

class TensorBuffer:
    """
    Buffer ndarray(N,6) per queue_key.
    queue_key = (market_category, symbol_category, symbol, timeframe)
    Thread-safe — dùng threading.Lock cho read-modify-write.
    """

    def __init__(self):
        self._data: Dict[tuple, np.ndarray] = {}
        self._lock = threading.Lock()

    def add_candles(self, queue_key: tuple, candles: np.ndarray) -> None:
        """np.concatenate — C-level, không Python loop."""
        if candles is None or len(candles) == 0:
            return
        with self._lock:
            if queue_key in self._data:
                self._data[queue_key] = np.concatenate([self._data[queue_key], candles])
            else:
                self._data[queue_key] = candles

    def pop(self, queue_key: tuple) -> Optional[np.ndarray]:
        with self._lock:
            return self._data.pop(queue_key, None)

    def get_all_keys(self) -> List[tuple]:
        with self._lock:
            return list(self._data.keys())

    @property
    def buffer_sizes(self) -> Dict[tuple, int]:
        with self._lock:
            return {k: len(v) for k, v in self._data.items()}


# ══════════════════════════════════════════════════════════════════════════════
# SlidingWindowRateLimiter — leaky-bucket pacer: RẢI ĐỀU theo thời gian + self-governing 429
# ══════════════════════════════════════════════════════════════════════════════

class SlidingWindowRateLimiter:
    """
    Pacer kiểu leaky-bucket — KHÔNG cho bùng nổ đồng thời tuyệt đối: mỗi request
    được cấp 1 "slot" cách đều nhau (1/rate giây) thay vì 80 cái đập cùng 1 thời điểm.
    Giống "điều phối giao thông" trên 80 làn: vẫn 80 làn, nhưng xe vào rải đều.

    rate_per_second = trần request/giây = số_làn / rate_limit_per_lane.
    429 → throttle() giãn slot (giảm nửa rate); thành công → recover() hồi dần về base.

    DÙNG CHUNG (global per-exchange) cho cả run — reserve slot rồi sleep NGOÀI lock
    để các coroutine đặt chỗ song song, mỗi cái nhận 1 mốc thời gian riêng cách đều.
    """

    def __init__(self, rate_per_second: float):
        self._base_rate: float = max(1.0, rate_per_second)
        self._rate: float = self._base_rate
        self._next: float = 0.0          # mốc monotonic sớm nhất được phép phát kế tiếp
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        # Reserve 1 slot cách đều trong lock (rẻ, không sleep trong lock), rồi sleep ngoài
        async with self._lock:
            now      = time.monotonic()
            interval = 1.0 / self._rate
            start    = self._next if self._next > now else now
            self._next = start + interval
            delay = start - now
        if delay > 0:
            await asyncio.sleep(delay)

    def throttle(self) -> None:
        """Giãn slot khi gặp 429 — giảm nửa rate, sàn 1 req/s."""
        self._rate = max(1.0, self._rate / 2.0)

    def recover(self) -> None:
        """Hồi phục dần về base rate sau mỗi request thành công."""
        if self._rate < self._base_rate:
            self._rate = min(self._base_rate, self._rate + 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# AsyncRequestBatcher — semaphore "làn" + sliding-window + 429/Retry-After backoff
# ══════════════════════════════════════════════════════════════════════════════

class AsyncRequestBatcher:
    """
    Batch fetch async với aiohttp.
    Mô hình "nhiều làn": Semaphore giới hạn số request đồng thời (số làn),
    SlidingWindowRateLimiter cho bùng nổ song song trong trần rate.
    parse_response(bytes, exchange) → ndarray(N,6) trực tiếp — zero-copy, zero dict.
    """
    MAX_RETRIES = 3

    @staticmethod
    def _parse_retry_after(resp: aiohttp.ClientResponse) -> Optional[float]:
        """Đọc header Retry-After (giây) nếu có."""
        val = resp.headers.get("Retry-After")
        if not val:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    async def fetch_single(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        rate_limiter: SlidingWindowRateLimiter,
        url: str,
        params: dict,
        exchange: str,
    ) -> np.ndarray:
        """1 request với retry + 429/Retry-After backoff. Trả ndarray(N,6) float64."""
        backoff_base = _EXCHANGE_BACKOFF_BASE.get(exchange, 2.0)
        max_backoff = _EXCHANGE_MAX_BACKOFF.get(exchange, 30.0)
        async with semaphore:
            for attempt in range(self.MAX_RETRIES):
                await rate_limiter.wait()
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status in (429, 418):
                            # Sàn đẩy lùi → siết rate + chờ Retry-After/exponential backoff
                            rate_limiter.throttle()
                            retry_after = self._parse_retry_after(resp)
                            delay = retry_after if retry_after is not None else min(
                                backoff_base * (2 ** attempt), max_backoff
                            )
                            logger.warning(
                                f"HTTP {resp.status} từ {exchange} → throttle, "
                                f"chờ {delay:.1f}s (retry {attempt+1}/{self.MAX_RETRIES})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        if resp.status != 200:
                            logger.warning(
                                f"HTTP {resp.status} từ {exchange}, retry {attempt+1}/{self.MAX_RETRIES}"
                            )
                            await asyncio.sleep(min(backoff_base * (2 ** attempt), max_backoff))
                            continue
                        raw = await resp.read()
                        rate_limiter.recover()
                        return ExchangeFormatter.parse_response(raw, exchange)
                except Exception as e:
                    logger.warning(f"Lỗi fetch {exchange} attempt {attempt+1}: {e}")
                    await asyncio.sleep(min(backoff_base * (2 ** attempt), max_backoff))
            return _EMPTY_CANDLES

    async def fetch_batch(
        self,
        request_batch: List[Tuple[str, dict, dict]],
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        rate_limiter: "SlidingWindowRateLimiter",
    ) -> np.ndarray:
        """
        Fetch song song qua semaphore + pacer GLOBAL per-exchange (truyền vào, dùng chung).
        request_batch: [(url, params, request_info), ...], info phải có 'exchange'.
        Trả ndarray(N_total, 6) float64 — np.concatenate kết quả.

        session/semaphore/rate_limiter dùng CHUNG cho cả run (1 event loop):
        diệt DNS storm + reuse connection; trần 80 là GLOBAL, pacer rải đều nhịp phát.
        """
        if not request_batch:
            return _EMPTY_CANDLES

        coros = [
            self.fetch_single(session, semaphore, rate_limiter, url, params, info['exchange'])
            for url, params, info in request_batch
        ]
        results: List[np.ndarray] = await asyncio.gather(*coros)

        non_empty = [r for r in results if len(r) > 0]
        if not non_empty:
            return _EMPTY_CANDLES
        return np.concatenate(non_empty)


# ══════════════════════════════════════════════════════════════════════════════
# DataFetcher — orchestrator chính
# ══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """Fetch dữ liệu từ exchange + aggregate custom TF + ghi TileDB."""

    def __init__(self):
        # Share calc_tf instance từ TimestampScanner — single source of truth
        self.timestamp_scanner = TimestampScanner()
        calc_tf = self.timestamp_scanner.calc_tf

        self.db_manager = DatabaseManager(calc_tf=calc_tf)
        self.cache_manager = CacheManager(
            cache_file=os.path.join(self.db_manager.root_path, "cache.json"),
            calc_tf=calc_tf,
            logger=logger,
        )
        self.exchange_url_builder = ExchangeURLBuilder()
        self.exchange_formatter = ExchangeFormatter()
        self.rate_limiter = RateLimiter()              # sync — dùng binary search
        self.async_batcher = AsyncRequestBatcher()

        # State
        self.fetch_results: Dict[tuple, bool] = {}
        self.symbol_data_start_cache: Dict[str, int] = {}
        self.tensor_buffer = TensorBuffer()

        # HTTP resources DÙNG CHUNG (global per-exchange), tạo trong event loop của run_phase1.
        # {exchange: (ClientSession, asyncio.Semaphore(80), SlidingWindowRateLimiter)}
        self._http: Dict[str, tuple] = {}

        # Executor RIÊNG cho TileDB blocking (scan/flush) — TÁCH khỏi default executor
        # mà aiohttp dùng cho DNS getaddrinfo, để cold-start không bị bỏ đói DNS.
        self._io_executor: Optional[ThreadPoolExecutor] = None
        # Executor RIÊNG cho aiohttp DNS getaddrinfo (set làm default executor của loop).
        self._dns_executor: Optional[ThreadPoolExecutor] = None
        # Executor TÁI DÙNG cho Phase 2 aggregate — continuous gọi aggregate mỗi chu kỳ,
        # tạo pool mới mỗi layer/chu kỳ = thread churn → mỗi thread mới có thể chiếm 1
        # arena glibc mới (RSS ratchet). Giữ 1 pool, chỉ dựng lại khi cần NHIỀU worker hơn.
        self._agg_executor: Optional[ThreadPoolExecutor] = None
        self._agg_workers: int = 0

        # TF active tính 1 lần ở _setup_phase1, tái dùng mỗi _fetch_pass (continuous).
        self._active_tfs: List[str] = []
        # Cờ "có ghi nến mới" — chỉ consolidate khi True (cả continuous lẫn one-shot).
        self._consolidate_dirty: bool = False


    # ── Shared HTTP per-exchange (1 session + semaphore + pacer GLOBAL) ──────────

    async def _get_http(self, exchange: str) -> tuple:
        """
        Lấy (session, semaphore, pacer) dùng chung cho exchange — tạo 1 lần/run.
        1 ClientSession → 1 connector → DNS resolve 1 lần (diệt DNS storm).
        Semaphore = trần concurrency GLOBAL (80 khi unlimited); pacer rải đều nhịp phát.
        """
        res = self._http.get(exchange)
        if res is None:
            cap = _HARD_CONCURRENCY_CAP if _MF_UNLIMITED else _EXCHANGE_MAX_CONCURRENCY.get(exchange, 10)
            lanes = max(1, int(cap))
            rate_per_lane   = _EXCHANGE_RATE.get(exchange, 0.2)
            rate_per_second = lanes / rate_per_lane if rate_per_lane > 0 else float(lanes)
            connector = aiohttp.TCPConnector(
                family=socket.AF_INET,
                limit=lanes,
                limit_per_host=lanes,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            session   = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=30))
            semaphore = asyncio.Semaphore(lanes)
            pacer     = SlidingWindowRateLimiter(rate_per_second)
            logger.info(
                f"{exchange}: trần GLOBAL {lanes} làn + pacing rải đều "
                f"~{rate_per_second:.0f} req/s (rate/làn={rate_per_lane}s)"
            )
            res = (session, semaphore, pacer)
            self._http[exchange] = res
        return res

    async def _close_http(self) -> None:
        """Đóng tất cả session dùng chung cuối run_phase1."""
        for session, _, _ in self._http.values():
            await session.close()
        self._http.clear()

    async def _to_io(self, fn, *args):
        """
        Chạy hàm blocking TileDB (scan/flush) trên executor RIÊNG (_io_executor),
        TÁCH khỏi default executor mà aiohttp dùng cho DNS getaddrinfo → cold-start
        không bị bỏ đói DNS. _io_executor được set ở đầu run_phase1.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._io_executor, fn, *args)

    @staticmethod
    def _prime_executor(ex: ThreadPoolExecutor, n: int) -> None:
        """
        Warm-up: ép start ĐỦ n worker NGAY (lúc chưa có log contention), rồi để idle sẵn.
        Sau prime, mọi submit chỉ tái dùng worker → KHÔNG bao giờ phải thread.start()
        giữa event loop (nguyên nhân treo: thread.start() chờ bootstrap khi worker khác
        đang giữ GIL trong picologging emit). Dùng Barrier để buộc đúng n thread start.
        """
        barrier = threading.Barrier(n + 1)
        for _ in range(n):
            ex.submit(barrier.wait)
        barrier.wait()   # đợi đủ n worker đã start + chạm barrier → tất cả đã sống

    @staticmethod
    def _ram_available_ratio() -> float:
        """P3.14: tỉ lệ RAM trống = MemAvailable/MemTotal đọc từ /proc/meminfo (lõi OS,
        chỉ đọc file + parse 2 dòng, không loop trên data). Lỗi → 1.0 (bỏ qua guard)."""
        try:
            with open('/proc/meminfo', 'rb') as f:
                data = f.read()

            def _kb(key: bytes) -> float:
                i = data.find(key)
                if i < 0:
                    return 0.0
                return float(data[i + len(key): data.find(b'\n', i)].split()[0])

            total = _kb(b'MemTotal:')
            avail = _kb(b'MemAvailable:')
            return avail / total if total else 1.0
        except Exception:
            return 1.0

    async def _ram_guard(self, stage: str = "") -> None:
        """
        P3.14: canh RAM ĐỘNG trước tác vụ nặng. Nếu trống < ngưỡng %RAM máy → gc.collect()
        + chờ ngắn (có timeout, KHÔNG block vô hạn) cho RAM hồi. Thích nghi mọi cỡ máy.
        """
        ratio = self._ram_available_ratio()
        if ratio >= _RAM_MIN_FREE_RATIO:
            return
        logger.warning(
            f"RAM trống {ratio*100:.0f}% < {_RAM_MIN_FREE_RATIO*100:.0f}% ({stage}) "
            f"→ gc.collect() + chờ hồi RAM"
        )
        gc.collect()
        for _ in range(_RAM_WAIT_MAX):
            await asyncio.sleep(0.5)
            if self._ram_available_ratio() >= _RAM_MIN_FREE_RATIO:
                logger.info(f"RAM đã hồi ({stage})")
                return
        logger.warning(f"RAM vẫn thấp sau chờ ({stage}) → tiếp tục thận trọng")

    def _prewarm_dns(self) -> None:
        """
        Pre-resolve DNS host của mọi exchange active 1 lần (sync) → OS DNS cache ấm,
        aiohttp resolve lần đầu hit cache tức thì, không vướng executor lúc cold-start.
        """
        from urllib.parse import urlparse
        seen: set = set()
        for name, cfg in EXCHANGE_CONFIGS.items():
            if not cfg.get("active", False):
                continue
            host = urlparse(cfg.get("api_url", "")).hostname
            if not host or host in seen:
                continue
            seen.add(host)
            try:
                socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
                logger.info(f"Pre-resolved DNS {host}")
            except Exception as e:
                logger.warning(f"Pre-resolve DNS {host} lỗi: {e}")

    # ── Flush buffer → TileDB ──────────────────────────────────────────────────

    def _flush_keys(self, keys: List[tuple]) -> None:
        """
        Ghi các queue_key chỉ định xuống TileDB qua batch_insert_multi (multi_write).
        Mỗi TF flush key của mình → không đua giữa các coroutine cùng symbol.
        Gọi qua _to_io (executor RIÊNG) để không chặn event loop / không đói DNS.
        """
        tasks: List[tuple] = []
        for queue_key in keys:
            candles = self.tensor_buffer.pop(queue_key)
            if candles is None or len(candles) == 0:
                continue
            market_category, symbol_category, symbol, timeframe = queue_key
            tasks.append((market_category, symbol_category, symbol, timeframe, candles))

        if not tasks:
            return

        results = self.db_manager.batch_insert_multi(tasks)

        # Update cache sau khi ghi xong (giữ nguyên logic update_cache_after_write)
        for market_category, symbol_category, symbol, timeframe, _ in tasks:
            array_path = build_array_path(market_category, symbol_category, symbol)
            self.cache_manager.update_cache_after_write(array_path, timeframe)

        total = sum(results.values())
        if total > 0:
            self._consolidate_dirty = True   # có ghi nến mới → cần consolidate gộp fragment
        logger.info(f"Flush {len(tasks)} arrays, tổng {total} candles vào TileDB")

    def _flush_to_db(self) -> None:
        """Flush toàn bộ buffer — wrapper quanh _flush_keys (giữ tương thích Phase 2)."""
        self._flush_keys(self.tensor_buffer.get_all_keys())

    # ── Conflict resolution helpers ────────────────────────────────────────────

    def _record_fetch_result(self, symbol: str, timeframe: str, has_data: bool) -> None:
        key = (symbol, timeframe)
        self.fetch_results[key] = has_data
        status = "CÓ DATA" if has_data else "KHÔNG CÓ DATA"
        logger.debug(f"Ghi nhận fetch: {symbol} ({timeframe}) → {status}")

    def _check_regular_timeframe_has_data(self, symbol: str, timeframe: str) -> bool:
        return self.fetch_results.get((symbol, timeframe), False)

    def _should_skip_custom_timeframe(self, symbol: str, timeframe_name: str) -> bool:
        """Skip aggregate nếu timeframe đã được fetch như regular và có data."""
        if timeframe_name not in TIMEFRAMES:
            return False
        if not TIMEFRAMES[timeframe_name].get("active", False):
            return False
        has_data = self._check_regular_timeframe_has_data(symbol, timeframe_name)
        if has_data:
            logger.info(f"{symbol} ({timeframe_name}): TIMEFRAMES đã có data → Bỏ qua aggregate")
            return True
        logger.info(f"{symbol} ({timeframe_name}): TIMEFRAMES không data → Vẫn aggregate")
        return False

    def _get_active_custom_timeframes(self, symbol: str) -> Dict[str, dict]:
        """Trả {tf_name: tf_data} cho custom TF active + không bị skip do conflict."""
        if not CUSTOM_TIMEFRAMES.get("enable", False):
            return {}
        custom_intervals = CUSTOM_TIMEFRAMES.get("custom_intervals", {})
        return {
            tf_name: tf_data
            for tf_name, tf_data in custom_intervals.items()
            if tf_data.get("active", False)
            and not self._should_skip_custom_timeframe(symbol, tf_name)
        }

    # ── Exchange selection ─────────────────────────────────────────────────────

    def _find_suitable_exchange(self, symbol_pair: str) -> Optional[str]:
        """Tìm exchange active đầu tiên. _VARIANT_LOOKUP đã xác nhận symbol có variants."""
        variants = _VARIANT_LOOKUP.get(symbol_pair.upper(), [])
        if not variants:
            logger.warning(f"Không có variants cho {symbol_pair}")
            return None
        for exchange_name, exchange_data in EXCHANGE_CONFIGS.items():
            if exchange_data.get('active', False):
                return exchange_name
        return None

    # ── Time range ─────────────────────────────────────────────────────────────

    def _calculate_time_range(self, timeframe: str) -> int:
        """Trả end_ts = mốc candle đóng cửa gần nhất cho timeframe."""
        return self.timestamp_scanner.calc_tf._get_current_closed_candle_time(timeframe)

    # ── Binary search tìm earliest data start ─────────────────────────────────

    def _find_earliest_data_start_for_symbol(
        self, market_category: str, symbol_category: str,
        symbol: str, exchange: str,
    ) -> Optional[int]:
        """Tìm mốc data start sớm nhất API có cho symbol — in-memory cache per run."""
        cache_key = f"{exchange}_{symbol}"

        # In-memory cache — tránh binary search lặp cho cùng symbol trong 1 run
        if cache_key in self.symbol_data_start_cache:
            cached_ts = self.symbol_data_start_cache[cache_key]
            logger.info(f"Cached start {cache_key}: {cached_ts}")
            return cached_ts

        logger.info(f"Tìm mốc data start cho {symbol} (exchange={exchange})...")

        active_timeframes = {
            tf_name: tf_data for tf_name, tf_data in TIMEFRAMES.items()
            if tf_data.get("active", False)
        }
        if not active_timeframes:
            logger.error("Không có timeframe nào active")
            return None

        # Sort theo minutes giảm dần — TF lớn tìm trước (ít nến → nhanh hơn)
        sorted_tfs = sorted(
            active_timeframes.items(),
            key=lambda x: x[1].get("minutes", x[1].get("hours", 0) * 60),
            reverse=True,
        )

        config_start_ts = self.timestamp_scanner.calc_tf._calculate_start_ts()

        for tf_name, tf_data in sorted_tfs:
            tf_minutes = tf_data.get("minutes", tf_data.get("hours", 0) * 60)
            interval_ms = tf_minutes * 60_000   # step size = 1 nến
            logger.info(f"Thử timeframe {tf_name} ({tf_minutes} phút)")
            end_ts = self._calculate_time_range(tf_name)

            # Variants — _VARIANT_LOOKUP O(1), bao gồm uppercase/lowercase + cached priority
            all_variants = list(_VARIANT_LOOKUP.get(symbol.upper(), []))
            sym_cache_key = f"{exchange}_{symbol}"
            cached_sym = ExchangeURLBuilder._successful_symbols.get(sym_cache_key)
            if cached_sym:
                if cached_sym in all_variants:
                    all_variants.remove(cached_sym)
                all_variants.insert(0, cached_sym)

            for variant in all_variants:
                logger.info(f"  Thử variant {variant}...")
                earliest_ts = self._find_data_start_with_binary_search(
                    exchange=exchange, variant=variant, timeframe=tf_name,
                    config_start_ts=config_start_ts, end_ts=end_ts,
                    interval_ms=interval_ms,
                )
                if earliest_ts is not None:
                    logger.info(f"Mốc {symbol} ({tf_name}, {variant}): {earliest_ts}")
                    ExchangeURLBuilder.update_successful_symbol(exchange, symbol, variant)
                    self.symbol_data_start_cache[cache_key] = earliest_ts
                    return earliest_ts

            logger.warning(f"Không tìm thấy data cho {tf_name}, thử TF nhỏ hơn")

        logger.error(f"Không tìm thấy mốc data start cho {symbol}")
        return None

    def _find_data_start_with_binary_search(
        self, exchange: str, variant: str, timeframe: str,
        config_start_ts: int, end_ts: int,
        interval_ms: int,
    ) -> Optional[int]:
        left  = config_start_ts
        right = end_ts

        logger.info(f"Binary search {variant} ({timeframe}): {left} → {right}")

        url, base_params = ExchangeURLBuilder.get_request_params(
            exchange=exchange, symbol=variant, timeframe=timeframe,
            start_ts=left, end_ts=left, limit=1,
        )
        if not url:
            return None

        ts_format   = _EXCHANGE_TS_FORMAT.get(exchange, "milliseconds")
        params_cfg  = EXCHANGE_CONFIGS.get(exchange, {}).get("format", {}).get("params", {})
        start_key   = next((k for k, v in params_cfg.items() if v.get("value") == "{start}"), None)
        end_key     = next((k for k, v in params_cfg.items() if v.get("value") == "{end}"),   None)

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._binary_search_async(
                url, base_params, exchange, left, right,
                interval_ms, ts_format, start_key, end_key, timeframe,
            ))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def _binary_search_async(
        self, url: str, base_params: dict, exchange: str,
        left: int, right: int, interval_ms: int,
        ts_format: str, start_key: Optional[str], end_key: Optional[str],
        timeframe: str,
    ) -> Optional[int]:
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(family=socket.AF_INET)
        # Binary search tuần tự (1 làn) — sliding-window 1/rate_limit req/s + 429 backoff
        rate_per_lane = _EXCHANGE_RATE.get(exchange, 0.2)
        rate_limiter = SlidingWindowRateLimiter(1.0 / rate_per_lane if rate_per_lane > 0 else 5.0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            if await self._check_api_async(session, rate_limiter, url, base_params, exchange):
                logger.info(f"Có data tại left bound {left}")
                return left

            logger.info(f"Không có data tại left bound, binary search...")
            earliest_ts: Optional[int] = None
            iteration = 0

            while left <= right:
                iteration += 1
                mid = (left + right) // 2
                logger.info(f"BS iter {iteration}: mid={mid} (left={left}, right={right})")

                adj = mid // 1000 if ts_format == "seconds" else mid
                params = dict(base_params)
                if start_key: params[start_key] = str(adj)
                if end_key:   params[end_key]   = str(adj)

                if await self._check_api_async(session, rate_limiter, url, params, exchange):
                    earliest_ts = mid
                    right = mid - interval_ms
                    logger.info(f"  Tìm thấy data tại {mid}, lùi trái tìm sớm hơn (right={right})")
                else:
                    left = mid + interval_ms
                    logger.info(f"  Không có data tại {mid}, tiến phải (left={left})")

            if earliest_ts is not None:
                _dt = datetime.fromtimestamp(earliest_ts / 1000, tz=timezone(timedelta(hours=7))).strftime('%H:%M:%S %A/%d/%m/%Y')
                logger.info(f"BS hoàn thành {iteration} iter cho {timeframe}: earliest={earliest_ts} ({_dt})")
            else:
                logger.warning(f"BS không tìm thấy data sau {iteration} iter cho {timeframe}")
            return earliest_ts

    async def _check_api_async(
        self, session: aiohttp.ClientSession, rate_limiter: SlidingWindowRateLimiter,
        url: str, params: dict, exchange: str,
    ) -> bool:
        backoff_base = _EXCHANGE_BACKOFF_BASE.get(exchange, 2.0)
        max_backoff = _EXCHANGE_MAX_BACKOFF.get(exchange, 30.0)
        for attempt in range(3):
            await rate_limiter.wait()
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status in (429, 418):
                        rate_limiter.throttle()
                        retry_after = AsyncRequestBatcher._parse_retry_after(resp)
                        delay = retry_after if retry_after is not None else min(
                            backoff_base * (2 ** attempt), max_backoff
                        )
                        logger.warning(f"HTTP {resp.status} (check) {exchange} → chờ {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue
                    if resp.status != 200:
                        logger.warning(f"API status {resp.status}, retry {attempt+1}/3")
                        await asyncio.sleep(min(backoff_base * (2 ** attempt), max_backoff))
                        continue
                    raw = await resp.read()
                    rate_limiter.recover()
                    return len(ExchangeFormatter.parse_response(raw, exchange)) > 0
            except Exception as e:
                logger.warning(f"Lỗi check data attempt {attempt+1}/3: {e}")
                await asyncio.sleep(min(backoff_base * (2 ** attempt), max_backoff))
        return False

    # ── Fetch + queue ──────────────────────────────────────────────────────────

    async def fetch_symbol_data(
        self, market_category: str, symbol_category: str,
        symbol: str, timeframe: str, exchange: str,
        start_ts: Optional[int] = None,
    ) -> bool:
        """Orchestrate fetch + flush + record cho 1 (symbol, timeframe) — async, 1 loop chung."""
        end_ts = self._calculate_time_range(timeframe)
        queue_key = (market_category, symbol_category, symbol, timeframe)

        if _SCAN_MODE_ACTIVE:
            # Phase 1 lặp: scan dimension → fetch nến thiếu → flush, tới khi hết
            success = await self._fetch_until_complete(
                market_category, symbol_category, symbol,
                timeframe, exchange, start_ts, end_ts,
            )
        else:
            # Cache mode: fetch 1 lượt → flush nguyên khối key này (TileDB blocking qua to_thread)
            success = await self._fetch_and_queue_data(
                market_category, symbol_category, symbol,
                timeframe, exchange, start_ts, end_ts,
            )
            await self._to_io(self._flush_keys, [queue_key])

        self._record_fetch_result(symbol, timeframe, success)
        return success

    def _count_missing_candles(self, intervals: np.ndarray, timeframe: str) -> int:
        """Tổng số nến thiếu trong các intervals (N,2) — đo độ hội tụ của vòng lặp."""
        iv = self.timestamp_scanner.calc_tf._get_timeframe_minutes(timeframe) * 60_000
        if iv <= 0 or intervals is None or len(intervals) == 0:
            return 0
        spans = intervals[:, 1].astype(np.int64) - intervals[:, 0].astype(np.int64)
        return int((spans // iv + 1).clip(min=0).sum())

    async def _fetch_until_complete(
        self, market_category: str, symbol_category: str,
        symbol: str, timeframe: str, exchange: str,
        start_ts: Optional[int], end_ts: Optional[int],
    ) -> bool:
        """
        Scan mode: lặp KHÔNG giới hạn scan dimension → fetch → flush, dừng khi
        "không còn gì để fetch": (1) scan trả 0 nến thiếu, hoặc (2) sau 1 lượt
        fetch số nến thiếu không giảm (API không cấp thêm → gap nguồn).
        Async: scan + flush (TileDB blocking) chạy qua _to_io (executor riêng).
        """
        array_path = build_array_path(market_category, symbol_category, symbol)
        queue_key = (market_category, symbol_category, symbol, timeframe)
        prev_missing: Optional[int] = None
        any_ok = False
        attempt = 0

        while True:
            attempt += 1
            intervals = await self._to_io(
                self.cache_manager.get_fetch_timestamps,
                array_path, timeframe, end_ts, start_ts,
            )
            if intervals is None or len(intervals) == 0:
                logger.info(f"{symbol} ({timeframe}): đầy đủ, không còn nến thiếu (sau {attempt-1} lượt fetch)")
                return True

            n_missing = self._count_missing_candles(intervals, timeframe)
            if n_missing <= 0:
                return True

            if prev_missing is not None and n_missing >= prev_missing:
                logger.warning(
                    f"{symbol} ({timeframe}): còn {n_missing} nến thiếu nhưng API "
                    f"không cấp thêm → HẾT thứ để fetch, dừng"
                )
                return any_ok

            if attempt > 1:
                logger.info(f"{symbol} ({timeframe}) lượt {attempt}: còn {n_missing} nến thiếu → fetch tiếp")
            prev_missing = n_missing

            ok = await self._fetch_and_queue_data(
                market_category, symbol_category, symbol,
                timeframe, exchange, start_ts, end_ts,
                fetch_intervals=intervals,
            )
            await self._to_io(self._flush_keys, [queue_key])
            any_ok = any_ok or ok

    async def _fetch_and_queue_data(
        self, market_category: str, symbol_category: str,
        symbol: str, timeframe: str, exchange: str,
        start_ts: Optional[int], end_ts: Optional[int],
        fetch_intervals: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Lấy intervals cần fetch, thử từng variant qua async batch.
        fetch_intervals: nếu None → tự quét (cache 1 lượt); nếu truyền vào →
        dùng thẳng (vòng lặp scan mode đã quét sẵn, tránh quét 2 lần).
        Async: scan TileDB qua to_thread; fetch qua session chung.
        """
        array_path = build_array_path(market_category, symbol_category, symbol)

        if fetch_intervals is None:
            fetch_intervals = await self._to_io(
                self.cache_manager.get_fetch_timestamps,
                array_path, timeframe, end_ts, start_ts,
            )
        if fetch_intervals is None or len(fetch_intervals) == 0:
            logger.info(f"Không cần fetch {symbol} ({timeframe}), đã cập nhật")
            return True

        # Variants — _VARIANT_LOOKUP O(1)
        all_variants = list(_VARIANT_LOOKUP.get(symbol.upper(), []))
        cache_key = f"{exchange}_{symbol}"
        cached_sym = ExchangeURLBuilder._successful_symbols.get(cache_key)
        if cached_sym:
            if cached_sym in all_variants:
                all_variants.remove(cached_sym)
            all_variants.insert(0, cached_sym)

        queue_key = (market_category, symbol_category, symbol, timeframe)
        config_start_ts = self.timestamp_scanner.calc_tf._calculate_start_ts()

        # Điều chỉnh interval đầu nếu start_ts (binary search mốc) sớm hơn
        # fetch_intervals là ndarray (N,2) int64
        adjusted_intervals = fetch_intervals
        if start_ts is not None and len(adjusted_intervals) > 0:
            first_start = int(adjusted_intervals[0][0])
            first_end = int(adjusted_intervals[0][1])
            if (first_start <= config_start_ts or first_start == 0) and start_ts < first_start:
                logger.info(f"Điều chỉnh interval đầu: {start_ts} → {first_end}")
                adjusted_intervals = adjusted_intervals.copy()
                adjusted_intervals[0, 0] = start_ts

        for variant in all_variants:
            logger.info(f"Thử variant {variant} cho {symbol} trên {exchange}")
            success = await self._fetch_intervals_async(
                exchange, variant, timeframe, adjusted_intervals, queue_key, symbol,
            )
            if success:
                ExchangeURLBuilder.update_successful_symbol(exchange, symbol, variant)
                size = self.tensor_buffer.buffer_sizes.get(queue_key, 0)
                logger.info(f"Đã fetch {size} nến {symbol} ({timeframe}), chờ flush")
                return True

        logger.error(f"Thử {len(all_variants)} variants cho {symbol} không thành công")
        return False

    async def _fetch_intervals_async(
        self, exchange: str, variant: str, timeframe: str,
        fetch_intervals: np.ndarray, queue_key: tuple, symbol: str,
    ) -> bool:
        """Build request_batch + await fetch_batch qua session/semaphore/pacer CHUNG."""
        try:
            exchange_config = EXCHANGE_CONFIGS.get(exchange, {})
            if not exchange_config:
                logger.error(f"Không có config exchange {exchange}")
                return False

            request_limit = (
                exchange_config.get("format", {}).get("params", {})
                .get("limit", {}).get("value", 1000)
            )
            tf_minutes = TIMEFRAMES.get(timeframe, {}).get("minutes", 60)
            interval_ms = request_limit * tf_minutes * 60 * 1000

            # Build URL + base params 1 lần — chỉ start/end thay đổi per chunk
            url, base_params = ExchangeURLBuilder.get_request_params(
                exchange=exchange, symbol=variant, timeframe=timeframe,
                start_ts=int(fetch_intervals[0][0]), end_ts=int(fetch_intervals[0][1]),
                limit=request_limit,
            )
            if not url:
                logger.warning(f"Không build được URL variant {variant}")
                return False

            ts_format = _EXCHANGE_TS_FORMAT.get(exchange, "milliseconds")
            params_cfg = EXCHANGE_CONFIGS.get(exchange, {}).get("format", {}).get("params", {})
            start_key = next((k for k, v in params_cfg.items() if v.get("value") == "{start}"), None)
            end_key = next((k for k, v in params_cfg.items() if v.get("value") == "{end}"), None)

            # Build request batch — chunk mỗi interval, chỉ update start/end
            request_batch: List[Tuple[str, dict, dict]] = []
            for row in fetch_intervals:
                interval_start_ts = int(row[0])
                interval_end_ts = int(row[1])
                current_start_ts = interval_start_ts
                while current_start_ts <= interval_end_ts:
                    current_end_ts = min(current_start_ts + interval_ms, interval_end_ts)
                    params = dict(base_params)
                    adj_s = current_start_ts // 1000 if ts_format == "seconds" else current_start_ts
                    adj_e = current_end_ts // 1000 if ts_format == "seconds" else current_end_ts
                    if start_key: params[start_key] = str(adj_s)
                    if end_key:   params[end_key] = str(adj_e)
                    request_batch.append((url, params, {
                        'exchange': exchange,
                        'variant': variant,
                        'timeframe': timeframe,
                        'start_ts': current_start_ts,
                        'end_ts': current_end_ts,
                        'symbol': symbol,
                    }))
                    current_start_ts += interval_ms

            if not request_batch:
                return False

            logger.info(f"Chuẩn bị {len(request_batch)} requests cho {symbol}")

            # Session + semaphore + pacer GLOBAL per-exchange — 1 event loop chung,
            # KHÔNG tạo loop/session riêng nữa (diệt DNS storm + reuse connection)
            session, semaphore, pacer = await self._get_http(exchange)
            all_candles = await self.async_batcher.fetch_batch(
                request_batch, session, semaphore, pacer
            )

            if len(all_candles) == 0:
                logger.warning(f"Không có nến từ {len(request_batch)} requests cho {symbol}")
                return False

            logger.info(f"Nhận {len(all_candles)} nến từ {len(request_batch)} requests")
            self.tensor_buffer.add_candles(queue_key, all_candles)
            return True

        except Exception as e:
            logger.error(f"Lỗi _fetch_intervals_async: {e}")
            return False

    # ── Orchestrators Phase 1 — 1 EVENT LOOP CHUNG, fetch theo SYMBOL ───────────

    def precompute_earliest(self) -> None:
        """
        Pre-pass (SYNC) tính mốc data start sớm nhất cho mọi symbol active — chỉ khi fetch_all.
        Chạy TRƯỚC asyncio.run() nên binary-search GIỮ NGUYÊN (tuần tự từng symbol,
        không deadlock); kết quả nạp vào symbol_data_start_cache để run_phase1 dùng.
        """
        if not _FETCH_ALL_ACTIVE:
            return
        for market_category, market_data in SYMBOLS_CONFIG["market"].items():
            if not market_data.get("active", False):
                continue
            for symbol_category, symbol_data in market_data["symbols_config"].items():
                if not symbol_data.get("active", False):
                    continue
                for symbol_pair, symbol_info in symbol_data["symbols"].items():
                    if not symbol_info.get("active", False):
                        continue
                    exchange = self._find_suitable_exchange(symbol_pair)
                    if not exchange:
                        continue
                    self._find_earliest_data_start_for_symbol(
                        market_category, symbol_category, symbol_pair, exchange,
                    )

    async def _setup_phase1(self) -> None:
        """
        Cold-start Phase 1 — chạy 1 LẦN: tính active_tfs, tạo + prime 2 executor,
        set default executor (DNS), prewarm DNS. Continuous gọi 1 lần đầu run rồi tái
        dùng xuyên suốt; one-shot run_phase1 gọi rồi teardown ngay sau 1 lượt fetch.
        """
        self._active_tfs = [tf for tf, td in TIMEFRAMES.items() if td.get("active", False)]
        if not self._active_tfs:
            logger.info("Không có timeframe nào active")
            return

        loop = asyncio.get_running_loop()

        # Executor RIÊNG cho TileDB blocking (scan/flush) — KHÔNG dùng default executor.
        n_io = max(8, len(self._active_tfs) * 2)
        self._io_executor = ThreadPoolExecutor(max_workers=n_io, thread_name_prefix="tiledb_io")

        # Executor RIÊNG cho aiohttp DNS getaddrinfo (đặt làm default executor của loop).
        n_dns = 8
        self._dns_executor = ThreadPoolExecutor(max_workers=n_dns, thread_name_prefix="aiohttp_dns")
        loop.set_default_executor(self._dns_executor)

        # PRIME cả 2: start sẵn toàn bộ worker TRƯỚC khi fetch (lúc chưa có log contention)
        # → trong vòng fetch, submit chỉ tái dùng worker, KHÔNG bao giờ thread.start() block loop.
        self._prime_executor(self._io_executor, n_io)
        self._prime_executor(self._dns_executor, n_dns)
        logger.info(f"Đã prime executor: io={n_io} worker, dns={n_dns} worker")

        self._prewarm_dns()   # ấm OS DNS cache trước khi fetch

    def consume_consolidate_flag(self) -> bool:
        """Trả + reset cờ 'có ghi nến mới'. True ⇒ cần consolidate gộp fragment; False ⇒ bỏ qua."""
        if self._consolidate_dirty:
            self._consolidate_dirty = False
            return True
        return False

    async def _teardown_phase1(self) -> None:
        """Đóng session + shutdown executor (continuous: lúc thoát; one-shot: ngay sau 1 lượt fetch)."""
        await self._close_http()
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=False)
            self._io_executor = None
        if self._dns_executor is not None:
            self._dns_executor.shutdown(wait=False)
            self._dns_executor = None
        if self._agg_executor is not None:
            self._agg_executor.shutdown(wait=False)
            self._agg_executor = None
            self._agg_workers = 0

    async def run_phase1(self) -> None:
        """
        Phase 1 one-shot (api_fetch.main): setup → 1 lượt fetch → teardown, trong 1 event loop chung.
        Continuous mode KHÔNG dùng hàm này — nó lái _setup_phase1/_fetch_pass/_teardown_phase1
        trên 1 loop BỀN (tái dùng executor/session/DNS xuyên suốt, không build lại mỗi chu kỳ).
        """
        await self._setup_phase1()
        try:
            await self._fetch_pass()
        finally:
            await self._teardown_phase1()

    async def _fetch_pass(self) -> None:
        """
        1 LƯỢT fetch toàn bộ symbol × TF active trong loop chung (đã _setup_phase1 sẵn).
        Duyệt từng SYMBOL; với mỗi symbol asyncio.gather toàn bộ TF active CHẠY SONG SONG
        (session/semaphore/pacer GLOBAL per-exchange). Mỗi TF tự flush key của mình.
        Continuous gọi lặp mỗi chu kỳ; one-shot gọi 1 lần qua run_phase1.
        """
        active_tfs = self._active_tfs
        if not active_tfs:
            return

        for market_category, market_data in SYMBOLS_CONFIG["market"].items():
            if not market_data.get("active", False):
                continue
            for symbol_category, symbol_data in market_data["symbols_config"].items():
                if not symbol_data.get("active", False):
                    continue
                for symbol_pair, symbol_info in symbol_data["symbols"].items():
                    if not symbol_info.get("active", False):
                        continue

                    # P3.13: isolation — 1 symbol lỗi KHÔNG làm sập cả run
                    try:
                        exchange = self._find_suitable_exchange(symbol_pair)
                        if not exchange:
                            logger.warning(f"Không tìm thấy exchange cho {symbol_pair}, bỏ qua")
                            continue

                        if _FETCH_ALL_ACTIVE:
                            start_ts = self.symbol_data_start_cache.get(f"{exchange}_{symbol_pair}")
                            if start_ts is None:
                                logger.warning(f"Không có mốc earliest cho {symbol_pair}, bỏ qua")
                                continue
                        else:
                            start_ts = self.timestamp_scanner.calc_tf._calculate_start_ts()

                        await self._ram_guard(f"trước symbol {symbol_pair}")   # P3.14

                        logger.info(
                            f"Symbol {symbol_pair}: fetch {len(active_tfs)} TF song song "
                            f"(1 loop chung) — {', '.join(active_tfs)}"
                        )
                        results = await asyncio.gather(
                            *[
                                self.fetch_symbol_data(
                                    market_category, symbol_category, symbol_pair,
                                    tf_name, exchange, start_ts,
                                )
                                for tf_name in active_tfs
                            ],
                            return_exceptions=True,
                        )
                        for tf_name, res in zip(active_tfs, results):
                            if isinstance(res, Exception):
                                logger.error(f"{symbol_pair} ({tf_name}): {res}")
                            else:
                                logger.info(f"{symbol_pair} ({tf_name})")
                        logger.info(f"Hoàn thành symbol {symbol_pair}")
                    except Exception as e:
                        logger.error(f"Symbol {symbol_pair} lỗi → bỏ qua, KHÔNG sập run: {e}")
                    finally:
                        # A3: ghi cache RAM xuống file cuối mỗi symbol (an toàn nếu crash sau đó)
                        await self._to_io(self.cache_manager.flush)

    def fetch_data_by_intervals(self) -> None:
        """Phase 1 entry — precompute earliest (sync) rồi chạy 1 event loop chung."""
        logger.info("PHASE 1 — 1 EVENT LOOP CHUNG, fetch theo symbol")
        self.precompute_earliest()
        asyncio.run(self.run_phase1())

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Aggregate custom timeframes (chỉ chạy sau Phase 1 done)
    # ══════════════════════════════════════════════════════════════════════════

    def aggregate_custom_timeframes(self) -> None:
        logger.info("Bắt đầu tổng hợp custom timeframes...")
        if not CUSTOM_TIMEFRAMES.get("enable", False):
            logger.info("Custom timeframes tắt, bỏ qua")
            return

        if _MF_ACTIVE:
            logger.info("CHẾ ĐỘ MULTI AGGREGATE")
            self._aggregate_custom_timeframes_multi_threaded()
        else:
            logger.info("CHẾ ĐỘ SEQUENTIAL AGGREGATE")
            self._aggregate_custom_timeframes_sequential()

        logger.info("Hoàn thành tổng hợp custom timeframes")

    def _build_source_to_tasks(self) -> Dict[Tuple, List[Tuple[str, np.ndarray]]]:
        """
        Build {source_key: [(custom_tf, intervals), ...]} với conflict resolution.
        source_key = (market_category, symbol_category, symbol, source_timeframe)
        """
        source_to_tasks: Dict[Tuple, List[Tuple[str, np.ndarray]]] = defaultdict(list)
        current_time = self._calculate_time_range  # method reference, gọi per TF

        for market_category, market_data in SYMBOLS_CONFIG["market"].items():
            if not market_data.get("active", False):
                continue
            for symbol_category, symbol_data in market_data["symbols_config"].items():
                if not symbol_data.get("active", False):
                    continue
                for symbol_pair, symbol_info in symbol_data["symbols"].items():
                    if not symbol_info.get("active", False):
                        continue

                    active_custom_tfs = self._get_active_custom_timeframes(symbol_pair)
                    if not active_custom_tfs:
                        continue

                    exchange = self._find_suitable_exchange(symbol_pair)
                    if not exchange:
                        continue

                    if _FETCH_ALL_ACTIVE:
                        symbol_earliest_ts = self._find_earliest_data_start_for_symbol(
                            market_category, symbol_category, symbol_pair, exchange,
                        )
                        if symbol_earliest_ts is None:
                            continue
                    else:
                        symbol_earliest_ts = self.timestamp_scanner.calc_tf._calculate_start_ts()

                    array_path = build_array_path(market_category, symbol_category, symbol_pair)

                    for custom_tf, custom_tf_data in active_custom_tfs.items():
                        source_timeframe = custom_tf_data.get("source")
                        if not source_timeframe:
                            logger.error(f"Custom {custom_tf} không có source")
                            continue

                        end_ts = current_time(custom_tf)
                        aggregation_intervals = self.cache_manager.get_fetch_timestamps(
                            array_path, custom_tf, end_ts, start_ts=symbol_earliest_ts,
                        )
                        if aggregation_intervals is None or len(aggregation_intervals) == 0:
                            logger.debug(f"{symbol_pair} ({custom_tf}): đầy đủ, skip")
                            continue

                        logger.info(
                            f"{symbol_pair} ({custom_tf}): "
                            f"{len(aggregation_intervals)} khoảng cần aggregate"
                        )
                        source_key = (market_category, symbol_category, symbol_pair, source_timeframe)
                        source_to_tasks[source_key].append((custom_tf, aggregation_intervals))

        return source_to_tasks

    def _aggregate_custom_timeframes_sequential(self) -> None:
        """Sequential: source-grouped, sort theo TF tăng dần."""
        source_to_tasks = self._build_source_to_tasks()
        logger.info(f"{len(source_to_tasks)} sources cần query")

        def get_source_priority(source_key: Tuple) -> int:
            _, _, _, source_timeframe = source_key
            return self.timestamp_scanner.calc_tf._get_timeframe_minutes(source_timeframe)

        sorted_sources = sorted(source_to_tasks.items(), key=lambda x: get_source_priority(x[0]))

        for idx, (source_key, tasks) in enumerate(sorted_sources):
            logger.info(f"\n{'='*80}")
            logger.info(f"[{idx+1}/{len(sorted_sources)}] Source: {source_key[2]} ({source_key[3]})")
            logger.info(f"{'='*80}")
            self._aggregate_source_task(source_key, tasks)

            # Flush sau mỗi source — cache update + free buffer
            if self.tensor_buffer.get_all_keys():
                logger.info(f"Flush source {source_key[3]}")
                self._flush_to_db()

        logger.info(f"HOÀN THÀNH sequential aggregate: {len(sorted_sources)} sources")

    def _get_agg_executor(self, workers: int) -> ThreadPoolExecutor:
        """
        Pool TÁI DÙNG cho aggregate — thay cho `ThreadPoolExecutor` dựng mới mỗi layer,
        mỗi chu kỳ (thread churn ⇒ arena churn ⇒ RSS ratchet khi treo lâu).

        Ngữ nghĩa concurrency GIỮ NGUYÊN: submit N task vào pool có max_workers ≥ N chạy
        song song y hệt pool dựng riêng cỡ N. Chỉ dựng lại khi layer cần NHIỀU worker hơn
        pool hiện có → sau vài chu kỳ pool hội tụ về layer lớn nhất và hết churn.
        Chế độ limited: workers đã bị `min(_MF_MAX_WORKERS, layer_size)` chặn ở caller nên
        pool không bao giờ vượt trần cấu hình.
        """
        need = max(1, workers)
        ex = self._agg_executor
        if ex is not None and self._agg_workers >= need:
            return ex
        if ex is not None:
            ex.shutdown(wait=True)   # layer trước đã xong (as_completed duyệt hết) → an toàn
        self._agg_executor = ThreadPoolExecutor(max_workers=need, thread_name_prefix="agg")
        self._agg_workers = need
        return self._agg_executor

    def _aggregate_custom_timeframes_multi_threaded(self) -> None:
        """Multi: dependency layers, parallel trong layer, sequential giữa layers."""
        source_to_tasks = self._build_source_to_tasks()
        if not source_to_tasks:
            logger.info("Không có source nào cần aggregate")
            return

        total_sources = len(source_to_tasks)
        execution_layers = self._build_dependency_layers(source_to_tasks)

        if _MF_UNLIMITED:
            logger.info(f"UNLIMITED AGGREGATE — {total_sources} sources, {len(execution_layers)} layers")
        else:
            logger.info(f"LIMITED AGGREGATE — max {_MF_MAX_WORKERS} workers")

        total_completed = 0
        for layer_idx, layer_sources in enumerate(execution_layers):
            layer_size = len(layer_sources)
            workers = layer_size if _MF_UNLIMITED else min(_MF_MAX_WORKERS, layer_size)
            logger.info(f"Layer {layer_idx}/{len(execution_layers)-1}: {layer_size} sources, {workers} workers")

            layer_completed = 0
            executor = self._get_agg_executor(workers)
            future_to_source = {
                executor.submit(self._aggregate_source_task, sk, source_to_tasks[sk]): sk
                for sk in layer_sources
            }
            # as_completed duyệt HẾT future của layer → vẫn là rào chắn giữa 2 layer y hệt
            # lúc thoát `with ThreadPoolExecutor(...)`, chỉ khác là pool không bị hủy.
            for fut in as_completed(future_to_source):
                source_key = future_to_source[fut]
                layer_completed += 1
                total_completed += 1
                try:
                    result = fut.result()
                    status = "" if result else ""
                    logger.info(
                        f"{status} [L{layer_idx} {layer_completed}/{layer_size}] "
                        f"[{total_completed}/{total_sources}] {source_key[2]} ({source_key[3]})"
                    )
                except Exception as e:
                    logger.error(
                        f"[L{layer_idx} {layer_completed}/{layer_size}] "
                        f"{source_key[2]} ({source_key[3]}): {e}"
                    )

            # Flush sau mỗi layer
            if self.tensor_buffer.get_all_keys():
                logger.info(f"Flush Layer {layer_idx}")
                self._flush_to_db()

        logger.info(
            f"HOÀN THÀNH multi aggregate: {total_completed}/{total_sources} sources, "
            f"{len(execution_layers)} layers"
        )

    def _aggregate_source_task(
        self, source_key: Tuple, tasks: List[Tuple[str, np.ndarray]],
    ) -> bool:
        """Worker — query source 1 lần, reuse cho tất cả custom TFs từ source đó."""
        try:
            market_category, symbol_category, symbol_pair, source_timeframe = source_key
            custom_tf_names = [tf for tf, _ in tasks]
            logger.info(
                f"Source {symbol_pair} ({source_timeframe}) → "
                f"{len(tasks)} custom: {', '.join(custom_tf_names)}"
            )

            # P5: TÍNH TRƯỚC slot custom dựng được (rẻ, KHÔNG query DB) để bó sát phạm vi
            # query source. Interval phantom (vùng không có slot, vd gap trước nến custom
            # đầu) hoặc slot chưa đóng → rỗng → KHÔNG kéo merged_start về tận config_start.
            calc   = self.timestamp_scanner.calc_tf
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)   # chặn trên: slot ĐÃ ĐÓNG

            slot_min = slot_max = None
            max_target_ms = 0
            for custom_tf, ivs in tasks:
                tmin_ms = calc._get_timeframe_minutes(custom_tf) * 60_000
                if tmin_ms > max_target_ms:
                    max_target_ms = tmin_ms
                slots = calc._generate_expected_slot_starts(
                    custom_tf, int(ivs[:, 0].min()), now_ms, filter_intervals=ivs,
                )
                if len(slots) == 0:
                    continue
                lo = int(slots.min()); hi = int(slots.max())
                slot_min = lo if slot_min is None else min(slot_min, lo)
                slot_max = hi if slot_max is None else max(slot_max, hi)

            # Không có slot custom nào dựng được → KHỎI query source (hết phí query cả trăm nến)
            if slot_min is None:
                logger.info(
                    f"{symbol_pair} ({source_timeframe}): không có slot custom mới cần dựng "
                    f"→ bỏ qua query source"
                )
                return False

            # Query source bó sát [slot_min, slot_max + target] — đủ phủ cửa sổ mọi slot cần dựng.
            # _aggregate_candles_batch_numpy vẫn lọc theo filter_intervals gốc → OUTPUT y hệt.
            merged_start   = slot_min
            merged_end_src = slot_max + max_target_ms

            array_path = build_array_path(market_category, symbol_category, symbol_pair)
            source_minutes = calc._get_timeframe_minutes(source_timeframe)
            source_array = self.db_manager.query_candles(
                array_path, source_minutes, merged_start, merged_end_src,
            )
            if len(source_array) == 0:
                logger.warning(f"Không có source data {symbol_pair} ({source_timeframe})")
                return False

            logger.info(f"Query {len(source_array)} nến source (bó sát {len(tasks)} custom TF)")

            aggregated_count = 0
            for custom_tf, aggregation_intervals in tasks:
                aggregated_data = self.timestamp_scanner.calc_tf._aggregate_candles_batch_numpy(
                    source_array=source_array,
                    source_timeframe=source_timeframe,
                    target_timeframe=custom_tf,
                    aggregation_intervals=aggregation_intervals,
                )
                if len(aggregated_data) == 0:
                    logger.debug(f"  - {custom_tf}: không nến mới")
                    continue

                queue_key = (market_category, symbol_category, symbol_pair, custom_tf)
                self.tensor_buffer.add_candles(queue_key, aggregated_data)
                aggregated_count += 1
                logger.info(f"  {custom_tf}: {len(aggregated_data)} nến mới")

            logger.info(f"Aggregate {aggregated_count}/{len(tasks)} custom TFs từ source")
            return aggregated_count > 0

        except Exception as e:
            logger.error(f"Lỗi _aggregate_source_task: {e}")
            return False

    def _build_dependency_layers(
        self, source_to_tasks: Dict[Tuple, List],
    ) -> List[List[Tuple]]:
        """
        Xây dependency layers — regular TF = level 0, custom = level(source) + 1.
        Iterative DAG resolution.
        """
        active_custom_tfs = {
            k: v for k, v in CUSTOM_TIMEFRAMES.get("custom_intervals", {}).items()
            if v.get("active", False)
        }

        timeframe_levels: Dict[str, int] = {}

        # Regular TFs active = level 0
        for tf_name, tf_data in TIMEFRAMES.items():
            if tf_data.get("active", False):
                timeframe_levels[tf_name] = 0

        # Custom TFs — iterative resolution
        max_iterations = 20
        changed = True
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for custom_tf, custom_tf_data in active_custom_tfs.items():
                source_tf = custom_tf_data.get("source")
                if not source_tf or source_tf not in timeframe_levels:
                    continue
                new_level = timeframe_levels[source_tf] + 1
                if custom_tf not in timeframe_levels:
                    timeframe_levels[custom_tf] = new_level
                    changed = True
                elif timeframe_levels[custom_tf] != new_level:
                    timeframe_levels[custom_tf] = new_level
                    changed = True

        # Group sources by level
        level_to_sources: Dict[int, List[Tuple]] = defaultdict(list)
        for source_key in source_to_tasks.keys():
            _, _, _, source_timeframe = source_key
            level = timeframe_levels.get(source_timeframe, 0)
            level_to_sources[level].append(source_key)

        # Sort sources trong mỗi level theo TF minutes
        def get_source_minutes(source_key: Tuple) -> int:
            _, _, _, source_timeframe = source_key
            return self.timestamp_scanner.calc_tf._get_timeframe_minutes(source_timeframe)

        for level in level_to_sources:
            level_to_sources[level].sort(key=get_source_minutes)

        max_level = max(level_to_sources.keys()) if level_to_sources else 0
        execution_layers: List[List[Tuple]] = []
        for level in range(max_level + 1):
            if level in level_to_sources:
                execution_layers.append(level_to_sources[level])

        return execution_layers

    # ══════════════════════════════════════════════════════════════════════════
    # Consolidate
    # ══════════════════════════════════════════════════════════════════════════

    def consolidate_all_arrays(self) -> None:
        """Consolidate tất cả arrays 1 lần cuối — gộp fragments."""
        logger.info("Consolidate tất cả arrays...")
        arrays_to_consolidate: set[str] = set()

        for market_category, market_data in SYMBOLS_CONFIG["market"].items():
            if not market_data.get("active", False):
                continue
            for symbol_category, symbol_data in market_data["symbols_config"].items():
                if not symbol_data.get("active", False):
                    continue
                for symbol_pair, symbol_info in symbol_data["symbols"].items():
                    if not symbol_info.get("active", False):
                        continue
                    array_path = build_array_path(market_category, symbol_category, symbol_pair)
                    if tiledb.object_type(array_path) == "array":
                        arrays_to_consolidate.add(array_path)

        for array_path in arrays_to_consolidate:
            self.db_manager.consolidate_array(array_path)

        logger.info(f"Đã consolidate {len(arrays_to_consolidate)} arrays")


# ══════════════════════════════════════════════════════════════════════════════
# Logging non-blocking — QueueHandler → 1 QueueListener thread (C-core picologging)
# ══════════════════════════════════════════════════════════════════════════════

# Mọi logger module chuyển sang emit qua queue (thread-safe, tức thì) → 1 listener
# thread ghi terminal → KHÔNG còn nhiều thread cùng emit picologging (diệt deadlock).
_LOG_NAMES = (
    'api_fetch', 'database', 'cache_timestamp', 'exchange_utils',
    'timestamp_scanner', 'Calculate_Tf_And_CustomTF',
)


def _setup_nonblocking_logging() -> QueueListener:
    """
    Định tuyến mọi logger module qua QueueHandler (C-ext) → 1 QueueListener thread.
    queue.SimpleQueue: lõi C `_queue`, thread-safe, không lock python thuần — chỉ
    tải log record (text), KHÔNG đụng mảng nến/zero-copy. Diệt deadlock picologging
    đa luồng (worker chỉ put, 1 listener ghi stdout). Vẫn ra terminal, không file.
    """
    log_queue: queue.SimpleQueue = queue.SimpleQueue()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))

    listener = QueueListener(log_queue, stream, respect_handler_level=True)

    qh = QueueHandler(log_queue)
    for name in _LOG_NAMES:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):      # bỏ StreamHandler cũ (emit đa luồng)
            lg.removeHandler(h)
        lg.addHandler(qh)
        lg.setLevel(logging.INFO)
        lg.propagate = False

    listener.start()
    return listener


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Pipeline 2 phase:
    Phase 1: fetch tất cả symbols × regular TFs → wait done
    Phase 2: aggregate custom TFs (chỉ chạy khi Phase 1 hoàn tất hẳn)
    Cuối: consolidate
    """
    log_listener = _setup_nonblocking_logging()   # logging non-blocking TRƯỚC mọi việc
    logger.info("Bắt đầu pipeline fetch data")
    fetcher = DataFetcher()

    fetcher.db_manager.create_database_structure()

    try:
        logger.info("=== Phase 1: FETCH ===")
        fetcher.fetch_data_by_intervals()

        logger.info("=== Phase 2: AGGREGATE CUSTOM TF ===")
        fetcher.aggregate_custom_timeframes()
    finally:
        fetcher.cache_manager.flush()   # A3: ghi cache RAM xuống file lần cuối (an toàn)
        if fetcher.consume_consolidate_flag():   # chỉ consolidate khi run có ghi nến mới
            fetcher.consolidate_all_arrays()
        else:
            logger.info("Không có nến mới ghi → bỏ qua consolidate")
        logger.info("Hoàn thành pipeline")
        log_listener.stop()   # flush + dừng listener thread


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng: {e}")
        raise
