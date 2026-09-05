"""Turn stored candles into a feature matrix.

Reads through the existing query layer (``get_data``) rather than talking to
TileDB directly, so the reader's locking, retry and block-assembly behaviour is
shared rather than duplicated.

Output per (symbol, timeframe) is a single Fortran-ordered float64 block: the
OHLCV columns exactly as stored, with the indicator columns appended beside
them. One allocation, one column layout, no intermediate matrices.

Warmup stays NaN. Nothing is forward-filled and no sentinel is substituted, so
the caller decides where its usable history begins instead of inheriting a
choice made here.
"""
from __future__ import annotations

import numpy as np
import picologging as logging

from get_data.get_data_from_database import BASE_COLS, GetDataFromDatabase
from indicators.config.config_indicators import COMPUTE_CONFIG, INDICATORS_CONFIG
from indicators.engines.oscillator_momentum_engines import MACD_Engine, RSI_Engine
from indicators.engines.trend_engines import BollingerBands_Engine, EMA_Engine

logger = logging.getLogger("feature_pipeline")


def build_engines(config: dict | None = None) -> list:
    """Instantiate every engine from configuration, in output-column order."""
    cfg = INDICATORS_CONFIG if config is None else config
    return [
        EMA_Engine(cfg["ema"]["lengths"]),
        MACD_Engine(cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"]),
        RSI_Engine(cfg["rsi"]["length"], cfg["rsi"]["ma_length"]),
        BollingerBands_Engine(cfg["bbands"]["length"], cfg["bbands"]["mult"]),
    ]


def feature_names(engines: list) -> tuple[str, ...]:
    """Column names of the assembled block: the stored columns, then indicators."""
    names: list[str] = list(BASE_COLS)
    for engine in engines:
        names.extend(engine.names)
    return tuple(names)


def compute_block(block: np.ndarray, source: np.ndarray, engines: list,
                  dtype=np.float64) -> np.ndarray:
    """Assemble one ``(N, len(BASE_COLS) + n_indicators)`` feature block.

    Args:
        block:  ``(N, len(BASE_COLS))`` float64 OHLCV, as returned by get_data.
        source: ``(N,)`` price series the indicators read (a column view of
                ``block`` — the scripts all take ``close``).
        engines: engines in column order, from ``build_engines``.
        dtype:  output dtype; float64 keeps the pipeline lossless.

    Returns:
        Fortran-ordered array whose first ``len(BASE_COLS)`` columns are the
        input block unchanged.
    """
    n_base = len(BASE_COLS)
    n_ind = sum(len(engine.names) for engine in engines)

    # Single allocation. Fortran order so every column — base and indicator
    # alike — is contiguous for whatever consumes it next.
    feat = np.empty((block.shape[0], n_base + n_ind), dtype=dtype, order="F")
    feat[:, :n_base] = block

    col = n_base
    for engine in engines:
        width = len(engine.names)
        engine.compute(source, feat[:, col:col + width])
        col += width
    return feat


def run(query=None) -> dict:
    """Compute features for every active symbol and timeframe.

    Args:
        query: object exposing ``query_data_for_training()``; defaults to a
            fresh ``GetDataFromDatabase``. Injectable so tests can drive the
            pipeline without standing up a reader.

    Returns:
        ``{(market, symbol_category, symbol, timeframe): {"names", "feat", "timestamp"}}``
        where ``feat`` is ``(T, len(names))`` and ``timestamp`` is ``int64[T]``,
        row-aligned with ``feat``.
    """
    reader = GetDataFromDatabase() if query is None else query
    engines = build_engines()
    names = feature_names(engines)
    dtype = np.dtype(COMPUTE_CONFIG.get("output_dtype", "float64"))

    data = reader.query_data_for_training()
    result: dict[tuple, dict] = {}

    for market, market_data in data.items():
        for category, category_data in market_data.items():
            for symbol, timeframes in category_data.items():
                for tf_name, tf_block in timeframes.items():
                    feat = compute_block(
                        tf_block["block"], tf_block["close"], engines, dtype
                    )
                    result[(market, category, symbol, tf_name)] = {
                        "names": names,
                        "feat": feat,
                        "timestamp": tf_block["timestamp"],
                    }
                    logger.info(
                        f"{market}/{category}/{symbol} @ {tf_name}: "
                        f"feat {feat.shape} {feat.dtype}"
                    )

    if not result:
        logger.warning("Không có block nào để tính feature. Fetch dữ liệu trước.")
    return result


# ── Torch hand-off ───────────────────────────────────────────────────────────

def _require_torch():
    """Import torch on demand — it is an optional dependency, not a core one.

    The ingestion and feature layers run on NumPy alone; pulling a CUDA wheel
    into the base install would cost gigabytes for users who only want data.
    """
    try:
        import torch
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "to_torch() requires PyTorch, an optional dependency. "
            "Install it with: pip install 'timeseries-ingestion-daemon[torch]'"
        ) from exc
    return torch


def to_torch(entry: dict, device=None) -> dict:
    """Hand one feature block to torch, sharing memory instead of copying it.

    ``torch.from_numpy`` aliases the NumPy buffer, so the tensors below start out
    pointing at the very bytes TileDB was read into — no second allocation, and a
    write through either side is visible from the other.

    The per-column tensors are the ones worth using. The block is Fortran-ordered
    (chosen so TileDB could take contiguous columns on write), which means:

    * every **column** view is both zero-copy *and* C-contiguous — torch can use
      it directly, with no hidden materialisation;
    * the **whole block** is zero-copy but reports ``is_contiguous() == False``
      (strides ``(1, N)``). Any op that needs C-contiguity will call
      ``.contiguous()`` internally and copy the lot. That copy is not avoided by
      handing the block over, only deferred — so prefer the columns, or accept
      the copy knowingly.

    For contrast, ``torch.tensor(feat)`` always copies.

    Args:
        entry: one value from ``run()`` — ``{"names", "feat", "timestamp"}``.
        device: optional torch device. Anything other than CPU **necessarily
            copies**, since the bytes have to cross to the device.

    Returns:
        dict with ``names``, ``block`` (N, F), ``timestamp`` (N,) and one
        ``(N,)`` tensor per feature name.
    """
    torch = _require_torch()

    feat, names = entry["feat"], entry["names"]
    reserved = {"names", "block", "timestamp"}
    clash = reserved.intersection(names)
    if clash:
        raise ValueError(f"Tên cột trùng khoá dành riêng của to_torch(): {sorted(clash)}")

    out: dict = {"names": names, "block": torch.from_numpy(feat)}
    out["timestamp"] = torch.from_numpy(entry["timestamp"])
    for j, name in enumerate(names):
        out[name] = torch.from_numpy(feat[:, j])       # F-order ⇒ contiguous view

    if device is not None:
        out = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in out.items()
        }
    return out
