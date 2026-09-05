from __future__ import annotations
import operator
import time
import orjson
import numpy as np
import picologging as logging
import sys

from ingestion.config_fetch_data import TIMEFRAMES, EXCHANGE_CONFIGS, SYMBOLS_CONFIG

logger = logging.getLogger('exchange_utils')
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

# ── Constants module level ─────────────────────────────────────────────────────
# Threshold phân biệt seconds vs milliseconds timestamp
# ts < 1e10 → seconds (năm ~2286 nếu là ms), ts >= 1e10 → milliseconds
_TS_SECONDS_THRESHOLD = 10_000_000_000

# ── Pre-build variant lookup 1 lần lúc module load ────────────────────────────
# Structure: {symbol_upper: [variant1, variant2, ...]} — chỉ active variants
# Tránh 3 vòng lặp lồng Python mỗi lần _find_exchange_symbol gọi
def _build_variant_lookup() -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for market_data in SYMBOLS_CONFIG["market"].values():
        if not market_data.get("active", False):
            continue
        for sc_data in market_data["symbols_config"].values():
            if not sc_data.get("active", False):
                continue
            for sym_name, sym_info in sc_data["symbols"].items():
                if not sym_info.get("active", False):
                    continue
                # Active variants từ config — giữ đúng case vì exchange cần
                active_variants: list[str] = [
                    v for v, vd in sym_info.get("variants", {}).items()
                    if isinstance(vd, dict) and vd.get("active", True)
                ]
                # Thêm upper + lower của từng variant — tùy exchange cần case nào
                extras: list[str] = []
                for v in active_variants:
                    u, l = v.upper(), v.lower()
                    if u not in active_variants:
                        extras.append(u)
                    if l not in active_variants:
                        extras.append(l)
                active_variants.extend(extras)
                # Dedup giữ thứ tự — dict.fromkeys C-level
                lookup[sym_name.upper()] = list(dict.fromkeys(active_variants))
    return lookup

# Lookup build 1 lần — tra O(1) thay vì O(n) loop mỗi lần gọi
_VARIANT_LOOKUP: dict[str, list[str]] = _build_variant_lookup()

# ── Pre-build exchange format cache 1 lần ─────────────────────────────────────
# Tránh .get() chain lặp lại mỗi lần normalize_timestamp / format_timeframe
_EXCHANGE_FORMAT: dict[str, dict] = {
    name: cfg.get("format", {})
    for name, cfg in EXCHANGE_CONFIGS.items()
}
_EXCHANGE_TF_FORMAT: dict[str, str] = {
    name: cfg.get("timeframe_format", "default")
    for name, cfg in EXCHANGE_CONFIGS.items()
}
_EXCHANGE_TS_FORMAT: dict[str, str] = {
    name: cfg.get("timestamp_format", "milliseconds")
    for name, cfg in EXCHANGE_CONFIGS.items()
}
# Reuse _EXCHANGE_FORMAT đã computed — không .get("format", {}) lại
_EXCHANGE_TS_MULTIPLIER: dict[str, int] = {
    name: _EXCHANGE_FORMAT[name].get("timestamp_multiplier", 1)
    for name in EXCHANGE_CONFIGS
}

# ── Required candle fields ─────────────────────────────────────────────────────
_REQUIRED_FIELDS = frozenset({'timestamp', 'open', 'high', 'low', 'close'})


# ══════════════════════════════════════════════════════════════════════════════
# ExchangeFormatter
# ══════════════════════════════════════════════════════════════════════════════

class ExchangeFormatter:

    @staticmethod
    def format_timeframe(timeframe: str, exchange: str) -> str:
        """
        Convert tên timeframe sang format exchange yêu cầu.
        Dùng pre-cached _EXCHANGE_TF_FORMAT — không .get() chain lại.
        """
        tf_config = TIMEFRAMES.get(timeframe)
        if not tf_config:
            logger.warning(f"Không tìm thấy cấu hình cho timeframe {timeframe}")
            return timeframe

        fmt = _EXCHANGE_TF_FORMAT.get(exchange, "default")

        if fmt == "seconds":
            if "seconds" in tf_config:
                return str(tf_config["seconds"])
            if "minutes" in tf_config:
                return str(tf_config["minutes"] * 60)
            if "hours" in tf_config:
                return str(tf_config["hours"] * 3600)
            return "86400"

        if fmt == "minutes":
            if "minutes" in tf_config:
                return str(tf_config["minutes"])
            if "hours" in tf_config:
                return str(tf_config["hours"] * 60)
            return "1440"

        if fmt == "hours":
            if "hours" in tf_config:
                return str(tf_config["hours"])
            if "minutes" in tf_config:
                return str(tf_config["minutes"] // 60)   # integer division — không qua float
            return "24"

        return timeframe

    @staticmethod
    def parse_response(response_bytes: bytes, exchange: str) -> np.ndarray:
        """
        Parse raw HTTP response bytes → np.ndarray (N, 6) float64.
        [timestamp_ms, open, high, low, close, volume]

        Dùng orjson (lõi C) thay json thuần.
        Trả thẳng numpy array — không tạo list-of-dicts trung gian.
        """
        try:
            data      = orjson.loads(response_bytes)
            fmt       = _EXCHANGE_FORMAT.get(exchange, {})
            resp_type = fmt.get('response_type', 'object')

            if resp_type == 'array':
                return ExchangeFormatter._parse_array_response(data, exchange, fmt)
            else:
                return ExchangeFormatter._parse_object_response(data, exchange, fmt)

        except Exception as e:
            logger.error(f"Lỗi parse response từ {exchange}: {e} | raw[:200]={response_bytes[:200]!r}")
            return np.empty((0, 6), dtype=np.float64)

    @staticmethod
    def _parse_array_response(data: list, exchange: str, fmt: dict) -> np.ndarray:
        """
        Parse array-type response (Binance style).
        Fast path: np.array(data, float64) — C-level bulk coercion, zero Python eval loop.
        Fallback: row-by-row cho malformed data (rare).
        """
        if not isinstance(data, list) or len(data) == 0:
            if isinstance(data, dict) and ('code' in data or 'msg' in data):
                logger.error(f"{exchange} trả lỗi API: code={data.get('code')} msg={data.get('msg')}")
            else:
                logger.error(f"Response từ {exchange} không phải list/rỗng: {repr(data)[:300]}")
            return np.empty((0, 6), dtype=np.float64)

        mapping    = fmt.get('mapping', {})
        multiplier = _EXCHANGE_TS_MULTIPLIER.get(exchange, 1)

        # Lấy index của từng field từ mapping — 1 lần duy nhất
        ts_idx = mapping.get('timestamp', {}).get('key', 0)
        o_idx  = mapping.get('open',      {}).get('key', 1)
        h_idx  = mapping.get('high',      {}).get('key', 2)
        l_idx  = mapping.get('low',       {}).get('key', 3)
        c_idx  = mapping.get('close',     {}).get('key', 4)
        v_idx  = mapping.get('volume',    {}).get('key', 5)

        n = len(data)

        try:
            # Fast path: numpy C-level bulk coercion (N, M) float64
            # Handles strings like "0.01634790" → float64 in C, zero Python loop
            full   = np.array(data, dtype=np.float64)
            result = np.empty((n, 6), dtype=np.float64)
            result[:, 0] = full[:, ts_idx]
            result[:, 1] = full[:, o_idx]
            result[:, 2] = full[:, h_idx]
            result[:, 3] = full[:, l_idx]
            result[:, 4] = full[:, c_idx]
            result[:, 5] = full[:, v_idx]

        except (ValueError, TypeError, IndexError):
            # Fallback: row-by-row cho malformed data
            result = np.empty((n, 6), dtype=np.float64)
            valid  = 0
            max_idx = max(ts_idx, o_idx, h_idx, l_idx, c_idx, v_idx)
            for item in data:
                if not isinstance(item, list) or len(item) <= max_idx:
                    logger.warning(f"Item không hợp lệ từ {exchange}: {item}")
                    continue
                try:
                    result[valid, 0] = float(item[ts_idx])
                    result[valid, 1] = float(item[o_idx])
                    result[valid, 2] = float(item[h_idx])
                    result[valid, 3] = float(item[l_idx])
                    result[valid, 4] = float(item[c_idx])
                    result[valid, 5] = float(item[v_idx])
                    valid += 1
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Lỗi parse candle từ {exchange}: {e}. Item: {item}")
            result = result[:valid]
            logger.debug(f"Nhận được {valid} nến (fallback) từ {exchange}")
            return result

        # Vectorized timestamp normalization
        if multiplier != 1:
            result[:, 0] *= multiplier
        else:
            mask = result[:, 0] < _TS_SECONDS_THRESHOLD
            if mask.any():
                result[mask, 0] *= 1000

        logger.debug(f"Nhận được {n} nến hợp lệ từ {exchange}")
        return result

    @staticmethod
    def _parse_object_response(data: dict, exchange: str, fmt: dict) -> np.ndarray:
        """
        Parse object-type response (Bitstamp style).
        Navigate data_path → build np.ndarray (N,6) float64 thẳng.
        operator.itemgetter (C-level) cho OHLC+ts; np.fromiter cho volume.
        """
        # Navigate data_path
        items = data
        for path in fmt.get('data_path', []):
            if isinstance(items, dict) and path in items:
                items = items[path]
            else:
                if isinstance(items, dict) and ('code' in items or 'msg' in items):
                    logger.error(f"{exchange} trả lỗi API: code={items.get('code')} msg={items.get('msg')}")
                else:
                    logger.warning(f"Không tìm thấy path '{path}' trong response của {exchange}: {repr(items)[:300]}")
                return np.empty((0, 6), dtype=np.float64)

        if not isinstance(items, list):
            items = [items] if items else []
        if len(items) == 0:
            return np.empty((0, 6), dtype=np.float64)

        mapping    = fmt.get('mapping', {})
        multiplier = _EXCHANGE_TS_MULTIPLIER.get(exchange, 1)

        # Lấy key string của từng field từ mapping — 1 lần duy nhất
        ts_key = mapping.get('timestamp', {}).get('key', 'timestamp')
        o_key  = mapping.get('open',      {}).get('key', 'open')
        h_key  = mapping.get('high',      {}).get('key', 'high')
        l_key  = mapping.get('low',       {}).get('key', 'low')
        c_key  = mapping.get('close',     {}).get('key', 'close')
        v_key  = mapping.get('volume',    {}).get('key', 'volume')

        # Filter invalid items — list comp (không tránh khỏi do validation cần dict check)
        required = (ts_key, o_key, h_key, l_key, c_key)
        valid_items = [
            item for item in items
            if isinstance(item, dict) and all(k in item for k in required)
        ]
        if not valid_items:
            return np.empty((0, 6), dtype=np.float64)

        # OHLC + ts: operator.itemgetter C-level + map C-level — zero Python loop
        ohlcts_getter = operator.itemgetter(ts_key, o_key, h_key, l_key, c_key)
        ohlcts = np.array(list(map(ohlcts_getter, valid_items)), dtype=np.float64)  # (M, 5)

        # Volume: np.fromiter — no temp list, C-level buffer; generator vì .get() default
        m    = len(valid_items)
        vols = np.fromiter(
            (item.get(v_key, 0.0) for item in valid_items),
            dtype=np.float64, count=m
        )

        # Pre-allocate (M, 6) — assign thẳng, không copy
        result = np.empty((m, 6), dtype=np.float64)
        result[:, :5] = ohlcts
        result[:, 5]  = vols

        # Vectorized timestamp normalization
        if multiplier != 1:
            result[:, 0] *= multiplier
        else:
            mask = result[:, 0] < _TS_SECONDS_THRESHOLD
            if mask.any():
                result[mask, 0] *= 1000

        logger.debug(f"Nhận được {m} nến hợp lệ từ {exchange}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# ExchangeURLBuilder
# ══════════════════════════════════════════════════════════════════════════════

class ExchangeURLBuilder:
    # Cache symbol đã thành công — {f"{exchange}_{symbol}": variant}
    _successful_symbols: dict[str, str] = {}

    @staticmethod
    def get_request_params(
        exchange: str,
        symbol:   str,
        timeframe: str,
        start_ts: int | None = None,
        end_ts:   int | None = None,
        limit:    int = 1000,
    ) -> tuple[str, dict]:
        """
        Build URL + params cho 1 request.
        Tính interval_ms từ limit * timeframe_seconds để không vượt quá limit nến.
        """
        exchange_config = EXCHANGE_CONFIGS.get(exchange, {})
        if not exchange_config:
            logger.error(f"Không tìm thấy cấu hình cho exchange {exchange}")
            return "", {}
        if not exchange_config.get("active", False):
            logger.error(f"Exchange {exchange} không active")
            return "", {}

        adjusted_start = start_ts if start_ts is not None else 1_293_840_000_000

        # Tính interval_ms từ timeframe seconds + limit
        tf_seconds_str = ExchangeFormatter.format_timeframe(timeframe, exchange)
        tf_seconds = int(tf_seconds_str) if tf_seconds_str.isdigit() else 3600

        interval_ms = limit * tf_seconds * 1000
        if end_ts is None:
            adjusted_end = None                                    # no end → API trả từ start về sau
        elif end_ts == 0:
            adjusted_end = adjusted_start + interval_ms            # 0 falsy → dùng interval window
        else:
            adjusted_end = min(adjusted_start + interval_ms, end_ts)

        logger.debug(f"{exchange}: start_ts={adjusted_start}, end_ts={adjusted_end}")

        exchange_symbol = ExchangeURLBuilder._find_exchange_symbol(exchange, symbol)
        return ExchangeURLBuilder._build_request(
            exchange, exchange_config, exchange_symbol,
            timeframe, adjusted_start, adjusted_end, limit
        )

    @staticmethod
    def _find_exchange_symbol(exchange: str, symbol: str) -> str:
        """
        Tìm variant đúng cho exchange — O(1) lookup từ _VARIANT_LOOKUP.
        Cache kết quả đã dùng thành công vào _successful_symbols.
        """
        cache_key = f"{exchange}_{symbol}"
        cached    = ExchangeURLBuilder._successful_symbols.get(cache_key)
        if cached:
            logger.debug(f"Dùng symbol cached cho {exchange}: {cached}")
            return cached

        # O(1) lookup thay vì 3 vòng lặp lồng
        variants = _VARIANT_LOOKUP.get(symbol.upper(), [])
        if not variants:
            logger.warning(f"Không tìm thấy variants cho {symbol}, dùng symbol gốc")
            return symbol

        logger.debug(f"Tìm thấy {len(variants)} variants cho {symbol}: {variants}")
        logger.debug(f"Dùng variant ưu tiên cho {exchange}: {variants[0]}")
        return variants[0]

    @staticmethod
    def _build_request(
        exchange:        str,
        exchange_config: dict,
        symbol:          str,
        timeframe:       str,
        start_ts:        int | None,
        end_ts:          int | None,
        limit:           int,
    ) -> tuple[str, dict]:
        """
        Build URL + params từ exchange config.
        exchange_name truyền thẳng vào — không tìm ngược từ config object.
        """
        try:
            api_url = exchange_config.get('api_url', '')
            if not api_url:
                logger.error(f"Thiếu URL API cho exchange {exchange}")
                return "", {}

            # Replace {symbol} trong URL nếu có
            if '{symbol}' in api_url:
                api_url = api_url.replace('{symbol}', symbol.lower())

            formatted_tf = ExchangeFormatter.format_timeframe(timeframe, exchange)
            ts_format    = _EXCHANGE_TS_FORMAT.get(exchange, "milliseconds")

            # Convert timestamps theo format exchange yêu cầu
            # Dùng `is not None` — 0 (epoch) là giá trị hợp lệ, không được treat là falsy
            adj_start = (start_ts // 1000 if ts_format == "seconds" else start_ts) if start_ts is not None else None
            adj_end   = (end_ts   // 1000 if ts_format == "seconds" else end_ts)   if end_ts   is not None else None

            # Build params từ config — không loop thừa
            params_fmt = _EXCHANGE_FORMAT.get(exchange, {}).get('params', {})
            params: dict = {}
            for param_key, param_data in params_fmt.items():
                value = param_data.get("value", "")
                if isinstance(value, str):
                    if   value == '{symbol}':    params[param_key] = symbol
                    elif value == '{timeframe}': params[param_key] = formatted_tf
                    elif value == '{start}':
                        if adj_start is not None: params[param_key] = str(adj_start)
                    elif value == '{end}':
                        if adj_end   is not None: params[param_key] = str(adj_end)
                    elif value == '{limit}':     params[param_key] = str(limit)
                    else:                        params[param_key] = value
                elif param_key == 'limit':
                    params[param_key] = limit    # config integer → luôn dùng passed limit value
                else:
                    params[param_key] = value

            # Lọc params rỗng
            params = {k: v for k, v in params.items() if v or v == 0}

            logger.debug(f"Đã build URL cho {exchange}: {api_url}, params: {params}")
            return api_url, params

        except Exception as e:
            logger.error(f"Lỗi build request cho {exchange}: {str(e)}")
            return "", {}

    @staticmethod
    def update_successful_symbol(exchange: str, symbol: str, successful_variant: str) -> None:
        """Cache lại variant đã gọi API thành công."""
        cache_key = f"{exchange}_{symbol}"
        ExchangeURLBuilder._successful_symbols[cache_key] = successful_variant
        logger.info(f"Đã cache symbol thành công cho {exchange}: {successful_variant}")


# ══════════════════════════════════════════════════════════════════════════════
# RateLimiter
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Rate limiter per exchange — đọc rate_limit từ EXCHANGE_CONFIGS.
    Dùng time.perf_counter() thay time.time() — độ chính xác cao hơn.
    """

    def __init__(self):
        # Pre-build limits dict — không .get() chain mỗi lần apply
        self._limits: dict[str, float] = {
            name: data.get('rate_limit', 1.0)
            for name, data in EXCHANGE_CONFIGS.items()
        }
        self._last_request: dict[str, float] = {}

    def apply_limit(self, exchange: str) -> None:
        """Sleep nếu chưa đủ interval kể từ request cuối."""
        min_interval = self._limits.get(exchange, 0.5)
        last         = self._last_request.get(exchange, 0.0)
        elapsed      = time.perf_counter() - last

        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limit: sleep {sleep_time:.3f}s cho {exchange}")
            time.sleep(sleep_time)

        self._last_request[exchange] = time.perf_counter()
