import os
import orjson
from typing import Any

# ─────────────────────────────────────────────
# DATA ROOT — luôn nằm TRONG repo, không phụ thuộc máy nào
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Mặc định <repo>/data. TSD_DATA_ROOT là override opt-in (mặc định KHÔNG set)
# cho ai muốn để dữ liệu sang ổ khác — không có fallback nào trỏ ra ngoài repo.
DATABASE_ROOT_PATH = os.path.abspath(
    os.environ.get("TSD_DATA_ROOT") or os.path.join(PROJECT_ROOT, "data")
)

# Tên group TileDB gốc — đổi ở ĐÂY là đổi cho toàn pipeline (writer + reader + tool).
DB_GROUP_NAME = "market_data"
DB_GROUP_ROOT = f"{DATABASE_ROOT_PATH}/{DB_GROUP_NAME}"


def build_array_path(market_category: str, symbol_category: str, symbol: str) -> str:
    """Absolute path of the TileDB array that stores one symbol.

    Layout: ``{data_root}/{group}/{market_category}/{symbol_category}/{symbol}``.
    Writer (``database``), reader (``get_data``) and the inspection tool all go
    through this helper, so the on-disk layout has exactly one definition.
    """
    return f"{DB_GROUP_ROOT}/{market_category}/{symbol_category}/{symbol}"


# ─────────────────────────────────────────────
# PATH HELPER
# ─────────────────────────────────────────────
_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config'))

def get_config_path(filename: str) -> str:
    path = os.path.join(_CONFIG_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return path

# ─────────────────────────────────────────────
# DAILY OPEN UTC — mốc mở cửa 1D, cố định 00:00 UTC
# ─────────────────────────────────────────────
DAILY_OPEN = "00:00"

# ─────────────────────────────────────────────
# GENERIC JSON LOADER (tránh lặp code open/orjson)
# ─────────────────────────────────────────────
def _load_json(filename: str) -> Any:
    """Load JSON bằng orjson (C core). Chỉ gọi 1 lần khi load module."""
    with open(get_config_path(filename), 'rb') as f:
        return orjson.loads(f.read())

# ─────────────────────────────────────────────
# HISTORICAL DATA CONFIG
# Thay vì copy từng field thủ công → giữ nguyên nested dict từ JSON,
# chỉ validate key tồn tại lúc load (fail fast).
# ─────────────────────────────────────────────
def load_historical_data_config() -> dict:
    return _load_json('historical_data_config.json')["historical_data"]

HISTORICAL_DATA_CONFIG = load_historical_data_config()

# ─────────────────────────────────────────────
# EXCHANGE CONFIGS
# ─────────────────────────────────────────────
def load_exchange_configs() -> dict:
    return _load_json('exchange_configs.json')["exchange_configs"]

EXCHANGE_CONFIGS = load_exchange_configs()

# ─────────────────────────────────────────────
# SYMBOLS CONFIG
# Giữ dict-comprehension lồng nhưng gọn + rõ ràng hơn
# ─────────────────────────────────────────────
def _build_symbols_config(raw: dict) -> dict:
    return {
        market: {
            "active": md["active"],
            "symbols_config": {
                sc_name: {
                    "active": sc["active"],
                    "symbols": {
                        sym: {
                            "active": sd["active"],
                            "variants": {
                                v: {"active": vd["active"]}
                                for v, vd in sd["variants"].items()
                            },
                        }
                        for sym, sd in sc["symbols"].items()
                    },
                }
                for sc_name, sc in md["symbols_config"].items()
            },
        }
        for market, md in raw.items()
    }

SYMBOLS_CONFIG: dict = {
    "market": _build_symbols_config(_load_json('symbols_config.json')["market"])
}

# ─────────────────────────────────────────────
# TIMEFRAMES
# ─────────────────────────────────────────────
def _load_all_timeframes() -> tuple[dict, dict]:
    data = _load_json('all_timeframes.json')

    timeframes = {
        name: {
            "active":  td["active"],
            "active_featured": td.get("active_featured", False),
            "active_prediction": td.get("active_prediction", False),
            "minutes": td["minutes"],
            "seconds": td["seconds"],
            "hours":   td.get("hours", 0),
        }
        for name, td in data["timeframes"].items()
    }

    custom = {
        "enable": data["custom_timeframes"]["enable"],
        "custom_intervals": {
            name: {
                "active":  cd["active"],
                "active_featured": cd.get("active_featured", False),
                "active_prediction": cd.get("active_prediction", False),
                "minutes": cd.get("minutes", 0),
                "hours":   cd.get("hours", 0),
                "source":  cd["source"],
            }
            for name, cd in data["custom_timeframes"]["custom_intervals"].items()
        },
    }
    return timeframes, custom

TIMEFRAMES, CUSTOM_TIMEFRAMES = _load_all_timeframes()

# ─────────────────────────────────────────────
# DATABASE STRUCTURE
# Flat comprehension thay vì nested 4 tầng lặp lại
# ─────────────────────────────────────────────
def _build_db_structure(symbols_config: dict) -> dict:
    return {
        DB_GROUP_NAME: {
            "market_categories": {
                market: {
                    "active": md["active"],
                    "symbol_categories": {
                        sc_name: {
                            "active": sc["active"],
                            "arrays": {
                                sym: {"active": sd["active"], "symbol": sym}
                                for sym, sd in sc["symbols"].items()
                            },
                        }
                        for sc_name, sc in md["symbols_config"].items()
                    },
                }
                for market, md in symbols_config["market"].items()
            }
        }
    }

DATABASE_STRUCTURE = _build_db_structure(SYMBOLS_CONFIG)

# QUERY CONFIG đã CHUYỂN sang get_data/config/config_query.py (thuộc tầng query/training).

# ─────────────────────────────────────────────
# CONTINUOUS FETCH MODE
# Làm phẳng {"<key>": {"value": x}} → {"<key>": x} để dùng trực tiếp
# ─────────────────────────────────────────────
def _load_fetch_mode() -> dict:
    raw = _load_json('continuous_fetch_mode.json')["fetch_mode"]
    cfg = {
        k: raw[k]["value"]
        for k in ("continuous", "fetch_interval", "sleep_interval", "continuous_sleep_interval")
    }
    # Trần RSS (MB) — vượt thì continuous_fetch tự re-exec ở ranh giới chu kỳ.
    # Cần vì libtiledb 0.36.1 rò ~15KB mỗi lệnh đọc sparse, không vá được trong tiến trình.
    # 0 = tắt. Key thiếu (config cũ) → mặc định 1536MB, KHÔNG lỗi.
    cfg["rss_restart_mb"] = raw.get("rss_restart_mb", {}).get("value", 1536)
    return cfg

FETCH_MODE_CONFIG = _load_fetch_mode()